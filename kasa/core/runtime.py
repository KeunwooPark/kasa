"""The path from a delivered event to an answer, assembled once.

Every surface needs the same three things between it and the agent: a durable
queue (#19), one actor per conversation (#20), and something that runs the turn
and sends the answer back where it came from. An adapter should not have to
wire those together correctly — it enqueues, and it says how to reply.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

from kasa.core.agent import Agent, AgentResult
from kasa.core.events import InboundEvent
from kasa.core.inbox import DEFAULT_LEASE_TTL, Dispatcher, Enqueued, Inbox
from kasa.core.session import IDLE_AFTER, SessionRouter, Turn
from kasa.llm.types import Delta
from kasa.store import Store

log = logging.getLogger(__name__)


class Reply(Protocol):
    """One turn's answer, on its way back to whoever asked for it.

    Opened before the model is called, so a surface that can show progress has
    somewhere to show it. A surface that cannot ignores `delta` entirely; see
    `one_message`.
    """

    async def delta(self, delta: Delta) -> None:
        """One fragment of the answer, as the model produces it.

        Called on the path that consumes the model's stream, so it must not
        wait on anything: whatever it does costs the turn its latency, once per
        token. Raising fails the turn, which is a heavy price for a progress
        indicator — swallow instead.
        """
        ...

    async def finish(self, result: AgentResult) -> None:
        """Deliver the finished answer.

        Raising means the turn failed, so the queue will run it again — model
        call included. A surface whose posting can fail transiently should
        retry inside this, not through it.
        """
        ...

    async def aclose(self) -> None:
        """The turn ended without an answer. Release anything `delta` held.

        Called instead of `finish` when the turn raised. The event goes back on
        the queue, so anything left half-written here is what the next attempt
        inherits.
        """
        ...


#: How a surface opens the reply for one turn. Called before the model is, and
#: on the turn path, so a failure here fails the turn like any other.
ReplyOpener = Callable[[InboundEvent], Awaitable[Reply]]

#: The whole of a surface that has nothing to say until the turn is over.
ReplySink = Callable[[InboundEvent, AgentResult], Awaitable[None]]

#: A last look at an event after it leaves the queue and before a session sees
#: it, for whatever a surface could not do at ingress. Slack's is resolving
#: `<@U0456>` to a name (#23): it needs a network call, and ingress has three
#: seconds and spends them on one INSERT.
#:
#: Off the ack path but *on* the turn path, so it inherits the turn's failure
#: semantics — raising here re-delivers the message. A hook whose work is an
#: improvement rather than a requirement should swallow its own failures and
#: return the event it was given, which is what `Directory.hydrate` does.
EventPreparer = Callable[[InboundEvent], Awaitable[InboundEvent]]
Scrubber = Callable[[str], str]

#: How many turns may be in flight at once, across all conversations.
DEFAULT_CONCURRENCY = 8


class _OneMessage:
    """A `Reply` that says nothing until it has the answer."""

    def __init__(self, event: InboundEvent, sink: ReplySink) -> None:
        self._event = event
        self._sink = sink

    async def delta(self, delta: Delta) -> None:
        pass

    async def finish(self, result: AgentResult) -> None:
        await self._sink(self._event, result)

    async def aclose(self) -> None:
        pass


def one_message(sink: ReplySink) -> ReplyOpener:
    """Turn a plain "post the answer" function into a `ReplyOpener`.

    For a surface with nothing to stream to — and for the Slack adapter with
    streaming switched off, which is what a workspace that would rather not
    watch a message rewrite itself gets.
    """

    async def open(event: InboundEvent) -> Reply:
        return _OneMessage(event, sink)

    return open


class Runtime:
    """Queue, actors and agent, wired the one way they work."""

    def __init__(
        self,
        agent: Agent,
        reply: ReplyOpener,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        lease_ttl: float = DEFAULT_LEASE_TTL,
        idle_after: float = IDLE_AFTER,
        prepare: EventPreparer | None = None,
        scrub: Scrubber | None = None,
    ) -> None:
        self._agent = agent
        self._reply = reply
        self._prepare = prepare
        self._scrub = scrub or (lambda text: text)
        self._ephemeral_originals: dict[tuple[str, str], str] = {}
        self.inbox = Inbox(agent.store, lease_ttl=lease_ttl)
        self._router = SessionRouter(agent.store, self._turn, idle_after=idle_after)
        self.dispatcher = Dispatcher(self.inbox, self._deliver, concurrency=concurrency)

    @property
    def store(self) -> Store:
        return self._agent.store

    async def submit(self, event: InboundEvent) -> Enqueued:
        """What an adapter calls at ingress. Durable by the time it returns."""
        original = event.text
        safe = self._scrub(original)
        changed = safe != original
        if changed:
            event = event.model_copy(update={"text": safe, "credential_scrubbed": True})
        enqueued = await self.inbox.enqueue(event)
        if changed and not enqueued.duplicate:
            self._ephemeral_originals[(event.source, event.external_id)] = original
        return enqueued

    async def run(self) -> None:
        """Answer whatever arrives, until `stop()`."""
        try:
            await self.dispatcher.run()
        finally:
            # After the dispatcher, so nothing is still being handed to an
            # actor that is closing.
            await self._router.aclose()

    def stop(self) -> None:
        self.dispatcher.stop()

    async def _deliver(self, event: InboundEvent) -> None:
        """Between the queue and the conversation.

        Before the router rather than inside the turn, so the session actor
        serializes an event that already says what it means. A scrubbed event's
        original is restored here only in memory; the agent's store boundary
        persists the redacted form while its first prompt sees the original.
        """
        if original := self._ephemeral_originals.pop((event.source, event.external_id), None):
            event = event.model_copy(update={"text": original})
        if self._prepare is not None:
            event = await self._prepare(event)
        await self._router.deliver(event)

    async def _turn(self, turn: Turn) -> None:
        reply = await self._reply(turn.event)
        try:
            result = await self._agent.respond(
                turn.event.session_id,
                turn.event.text,
                surface=turn.event.source,
                author=turn.event.author,
                # The session's scope, not the event's. They are derived the
                # same way and normally agree; when they do not, the record of
                # what this conversation has been under all along is the one to
                # trust, and an event cannot widen a session that was opened as
                # private.
                scope=turn.session.scope,
                on_delta=reply.delta,
                # So a later edit or deletion of this message can find what was
                # stored for it (#25).
                external_id=turn.event.external_id,
                credential_scrubbed=turn.event.credential_scrubbed,
                # Whether anybody actually said this just now (#179). The
                # session's is not an alternative here: origin is a property of
                # the event, and a thread can carry both kinds.
                origin=turn.event.origin,
                # Where this conversation's answers go, for the tools that
                # create something which will answer here later (#180). The
                # surface put them on the event; nothing in between edits them.
                channel=turn.event.channel,
                reply_to=turn.event.reply_to,
            )
        except BaseException:
            # Including cancellation, which is what a shutdown mid-turn is. A
            # reply left open holds a background task; the event is going back
            # on the queue either way.
            await reply.aclose()
            raise
        await reply.finish(result)

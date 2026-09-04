"""The path from a delivered event to an answer, assembled once.

Every surface needs the same three things between it and the agent: a durable
queue (#19), one actor per conversation (#20), and something that runs the turn
and sends the answer back where it came from. An adapter should not have to
wire those together correctly — it enqueues, and it says how to reply.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from kasa.core.agent import Agent, AgentResult
from kasa.core.events import InboundEvent
from kasa.core.inbox import DEFAULT_LEASE_TTL, Dispatcher, Enqueued, Inbox
from kasa.core.session import IDLE_AFTER, SessionRouter, Turn
from kasa.store import Store

log = logging.getLogger(__name__)

#: How an answer gets back to where the question came from. Raising means the
#: turn failed, so the queue will run it again — model call included. A surface
#: whose posting can fail transiently should retry inside this, not through it.
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

#: How many turns may be in flight at once, across all conversations.
DEFAULT_CONCURRENCY = 8


class Runtime:
    """Queue, actors and agent, wired the one way they work."""

    def __init__(
        self,
        agent: Agent,
        reply: ReplySink,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        lease_ttl: float = DEFAULT_LEASE_TTL,
        idle_after: float = IDLE_AFTER,
        prepare: EventPreparer | None = None,
    ) -> None:
        self._agent = agent
        self._reply = reply
        self._prepare = prepare
        self.inbox = Inbox(agent.store, lease_ttl=lease_ttl)
        self._router = SessionRouter(agent.store, self._turn, idle_after=idle_after)
        self.dispatcher = Dispatcher(self.inbox, self._deliver, concurrency=concurrency)

    @property
    def store(self) -> Store:
        return self._agent.store

    async def submit(self, event: InboundEvent) -> Enqueued:
        """What an adapter calls at ingress. Durable by the time it returns."""
        return await self.inbox.enqueue(event)

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
        serializes an event that already says what it means — and so what is
        stored as the user's message is the text the model was actually shown.
        """
        if self._prepare is not None:
            event = await self._prepare(event)
        await self._router.deliver(event)

    async def _turn(self, turn: Turn) -> None:
        result = await self._agent.respond(
            turn.event.session_id,
            turn.event.text,
            surface=turn.event.source,
            author=turn.event.author,
            # The session's scope, not the event's. They are derived the same
            # way and normally agree; when they do not, the record of what this
            # conversation has been under all along is the one to trust, and an
            # event cannot widen a session that was opened as private.
            scope=turn.session.scope,
        )
        await self._reply(turn.event, result)

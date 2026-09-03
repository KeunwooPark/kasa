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
    ) -> None:
        self._agent = agent
        self._reply = reply
        self.inbox = Inbox(agent.store, lease_ttl=lease_ttl)
        self._router = SessionRouter(agent.store, self._turn, idle_after=idle_after)
        self.dispatcher = Dispatcher(self.inbox, self._router.deliver, concurrency=concurrency)

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

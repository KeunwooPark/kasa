"""One actor per conversation, with a serialized mailbox.

Two messages arriving in the same thread while the agent is mid-turn must
queue, not interleave into one context window (`docs/DESIGN.md` §3.2). An actor
per session key is what enforces that, and it is the *only* ordering guarantee
in the system: within a session, strictly in order; across sessions, everything
at once.

Actors hold no durable state. Everything a turn needs is already in SQLite, so
an idle actor is dropped rather than kept warm, and the next message for that
session builds another one. That is what makes eviction free, and it is why
`_rehydrate` reads the session row every turn instead of caching it: a cached
copy is a second source of truth that goes stale the moment a background job
touches the same session.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from kasa.core.events import InboundEvent
from kasa.core.supervise import keep_running
from kasa.errors import KasaError
from kasa.store import Store

log = logging.getLogger(__name__)

#: How long an actor may sit with an empty mailbox before it is dropped.
IDLE_AFTER = 300.0

#: How often to look for actors to drop. Coarse on purpose: an idle actor costs
#: a dict entry and a parked task, so sweeping eagerly buys nothing.
SWEEP_INTERVAL = 30.0

#: How far behind one session may fall before it stops accepting messages.
#: The dispatcher bounds how many events are in flight, so reaching this means
#: something else is posting — and a queue that grows without limit inside one
#: thread is how a single busy channel takes the process down.
MAILBOX_LIMIT = 64


class SessionOverflow(KasaError):
    """A session is too far behind to accept another message."""


@dataclass(frozen=True, slots=True)
class SessionState:
    """What a session is, as of the start of the turn about to run."""

    id: str
    surface: str
    scope: str
    #: Turns already stored. The turns themselves are not here: the agent reads
    #: them from SQLite on every turn, and a second copy is a second truth.
    message_count: int
    #: The episode this turn belongs to, if one is open. Opening and closing
    #: them belongs to `episode_close` (#27); this only reports what is there.
    episode_id: str | None


@dataclass(frozen=True, slots=True)
class Turn:
    event: InboundEvent
    session: SessionState


#: What an actor does with a message. Raising means the turn failed, and the
#: caller — normally the dispatcher — decides whether to retry it.
TurnHandler = Callable[[Turn], Awaitable[None]]


@dataclass(slots=True)
class _Envelope:
    event: InboundEvent
    done: asyncio.Future[None]


class SessionActor:
    """A serialized mailbox for one session, and the task that drains it."""

    def __init__(
        self,
        session_id: str,
        *,
        store: Store,
        handler: TurnHandler,
        mailbox_limit: int = MAILBOX_LIMIT,
    ) -> None:
        self._id = session_id
        self._store = store
        self._handler = handler
        self._limit = mailbox_limit
        self._mailbox: asyncio.Queue[_Envelope | None] = asyncio.Queue()
        self._busy = False
        self._closing = False
        self._last_active = time.monotonic()
        self._task = asyncio.create_task(self._run(), name=f"session:{session_id}")

    @property
    def session_id(self) -> str:
        return self._id

    @property
    def waiting(self) -> int:
        return self._mailbox.qsize()

    @property
    def idle(self) -> bool:
        return not self._busy and self._mailbox.empty()

    @property
    def last_active(self) -> float:
        return self._last_active

    def post(self, event: InboundEvent) -> asyncio.Future[None]:
        """Queue an event; the returned awaitable completes with its turn.

        Synchronous up to the insert, deliberately. The dispatcher starts one
        task per event in arrival order and the first thing each one does is
        get here, so awaiting anything before the insert is how a later message
        in the same thread overtakes an earlier one.
        """
        if self._closing:
            raise SessionOverflow(f"session {self._id} is shutting down")
        if self._mailbox.qsize() >= self._limit:
            raise SessionOverflow(
                f"session {self._id} already has {self._mailbox.qsize()} message(s) waiting"
            )
        envelope = _Envelope(event, asyncio.get_running_loop().create_future())
        self._mailbox.put_nowait(envelope)
        self._last_active = time.monotonic()
        return envelope.done

    async def aclose(self) -> None:
        """Stop after the current turn and whatever is already queued."""
        if self._closing:
            await self._task
            return
        self._closing = True
        self._mailbox.put_nowait(None)
        await self._task

    # -- internals -----------------------------------------------------------

    async def _run(self) -> None:
        while (envelope := await self._mailbox.get()) is not None:
            # Whoever was waiting for this has gone — a dispatcher cancelled at
            # shutdown, most often. Running the turn anyway would answer a
            # message whose inbox row has already been handed back, which is
            # the one duplicate this layer can cheaply avoid.
            if envelope.done.cancelled():
                continue
            self._busy = True
            try:
                state = await self._rehydrate(envelope.event)
                await self._handler(Turn(event=envelope.event, session=state))
            except asyncio.CancelledError:
                if not envelope.done.done():
                    envelope.done.cancel()
                raise
            except Exception as exc:
                if not envelope.done.done():
                    envelope.done.set_exception(exc)
            else:
                if not envelope.done.done():
                    envelope.done.set_result(None)
            finally:
                self._busy = False
                self._last_active = time.monotonic()

    async def _rehydrate(self, event: InboundEvent) -> SessionState:
        """Read what this session is, from the only place that knows.

        Three small local queries against a database this process already has
        open, next to a model call that takes seconds. Paying them per turn is
        what buys the actor being free to evict.
        """
        await self._store.ensure_session(self._id, surface=event.source, scope=event.scope)
        row = await self._store.get_session(self._id)
        episode = await self._store.open_episode(self._id)
        return SessionState(
            id=self._id,
            surface=str(row["surface"]) if row else event.source,
            scope=str(row["scope"]) if row else event.scope,
            message_count=await self._store.message_count(self._id),
            episode_id=str(episode["id"]) if episode else None,
        )


class SessionRouter:
    """Delivers each event to the one actor that owns its session.

    Plug into the dispatcher as its handler: `Dispatcher(inbox, router.deliver)`.
    `deliver` returns when the turn is over, so the inbox row stays leased for
    as long as the work it describes is running.
    """

    def __init__(
        self,
        store: Store,
        handler: TurnHandler,
        *,
        idle_after: float = IDLE_AFTER,
        sweep_interval: float = SWEEP_INTERVAL,
        mailbox_limit: int = MAILBOX_LIMIT,
    ) -> None:
        self._store = store
        self._handler = handler
        self._idle_after = idle_after
        self._sweep_interval = sweep_interval
        self._mailbox_limit = mailbox_limit
        self._actors: dict[str, SessionActor] = {}
        self._sweeper: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def sessions(self) -> int:
        """How many actors are alive. Not how many sessions exist — that is a
        question for the database."""
        return len(self._actors)

    async def deliver(self, event: InboundEvent) -> None:
        """Hand an event to its session and wait for the turn to finish."""
        if self._closed:
            raise SessionOverflow("the router is closed")
        self._start_sweeper()
        return await self._actor_for(event.session_id).post(event)

    async def aclose(self) -> None:
        """Drain every actor. Queued messages are answered; nothing is dropped."""
        self._closed = True
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)
            self._sweeper = None
        actors = list(self._actors.values())
        self._actors.clear()
        await asyncio.gather(*(actor.aclose() for actor in actors), return_exceptions=True)

    async def evict_idle(self) -> int:
        """Drop actors that have been quiet. Free, because they hold nothing."""
        cutoff = time.monotonic() - self._idle_after
        stale = [
            session_id
            for session_id, actor in self._actors.items()
            if actor.idle and actor.last_active <= cutoff
        ]
        for session_id in stale:
            await self._actors.pop(session_id).aclose()
        if stale:
            log.debug("evicted %d idle session(s)", len(stale))
        return len(stale)

    # -- internals -----------------------------------------------------------

    def _actor_for(self, session_id: str) -> SessionActor:
        actor = self._actors.get(session_id)
        if actor is None:
            actor = SessionActor(
                session_id,
                store=self._store,
                handler=self._handler,
                mailbox_limit=self._mailbox_limit,
            )
            self._actors[session_id] = actor
        return actor

    def _start_sweeper(self) -> None:
        # Lazily, because a router is often built before there is a loop to
        # attach a task to, and because a router that never delivers anything
        # has nothing to sweep.
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(
                keep_running(self.evict_idle, every=self._sweep_interval, name="session sweeper"),
                name="session-sweeper",
            )

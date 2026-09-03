"""The durable ingress queue, and the loop that drains it.

Ingress is decoupled from processing (`docs/DESIGN.md` §3.1). An adapter
durably enqueues and acks; a `Dispatcher` leases rows back out and hands them
to a handler. Nothing an adapter does waits on a model call.

Delivery is at-least-once, and deliberately so. The alternative is to mark a
row done before the work happens, which turns every crash into a lost message —
and a chat assistant that silently ignores you is worse than one that answers
you twice. The dedupe that matters is at the *other* end: `UNIQUE (source,
external_id)` means a provider re-sending an event cannot produce a second
answer, which is the duplicate that actually happens in production.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kasa.core.backoff import Backoff
from kasa.core.drain import Drainer
from kasa.core.events import EventError, InboundEvent
from kasa.store import Store

log = logging.getLogger(__name__)

#: How long a leased row stays off-limits. The dispatcher renews the lease of
#: anything it is still working on, so this bounds how long a *dead* process's
#: work sits unclaimed — not how long a turn is allowed to take.
DEFAULT_LEASE_TTL = 60.0

#: Five leases, then a dead letter, waiting 2s, 4s, 8s … between them. Short at
#: the start because the common failure is a provider blip and the person is
#: still watching the thread. Leases rather than failures on purpose; see
#: `Store.reclaim_inbox`.
DEFAULT_BACKOFF = Backoff(max_attempts=5, base=2.0, cap=300.0)

#: How long a delivered row is kept. It is still the dedupe record for its
#: event id, so purging eagerly is how a late provider retry earns a second
#: answer. Slack re-sends an event for up to an hour; a week is cheap.
DONE_RETENTION = timedelta(days=7)

#: A handler that returns is a delivery; one that raises is a failed attempt.
EventHandler = Callable[[InboundEvent], Awaitable[None]]


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class Enqueued:
    """Where an event landed, and whether it was already there."""

    id: int
    duplicate: bool


@dataclass(frozen=True, slots=True)
class LeasedEvent:
    id: int
    event: InboundEvent
    #: Leases taken on this row so far, this one included. 1 is a first try.
    attempts: int


class Inbox:
    """The queue itself: enqueue, lease, and the outcome of a delivery."""

    def __init__(
        self,
        store: Store,
        *,
        lease_ttl: float = DEFAULT_LEASE_TTL,
        backoff: Backoff = DEFAULT_BACKOFF,
        retention: timedelta = DONE_RETENTION,
    ) -> None:
        self._store = store
        self._lease_ttl = lease_ttl
        self._backoff = backoff
        self._retention = retention
        self._subscribers: list[Callable[[], None]] = []

    @property
    def lease_ttl(self) -> float:
        return self._lease_ttl

    def subscribe(self, callback: Callable[[], None]) -> None:
        """Be told when something arrives, instead of waiting for the next poll.

        In-process only: a subscriber is a call on this event loop. An
        out-of-process drainer sees new work when it next polls, which is why
        the dispatcher polls at all rather than waiting on this alone.
        """
        self._subscribers.append(callback)

    async def enqueue(self, event: InboundEvent) -> Enqueued:
        """Durably record an event. Safe to call twice with the same one."""
        inbox_id, duplicate = await self._store.enqueue_inbox(
            source=event.source, external_id=event.external_id, payload=event.to_json()
        )
        if duplicate:
            log.debug("inbox %s: %s:%s already queued", inbox_id, event.source, event.external_id)
            return Enqueued(id=inbox_id, duplicate=True)
        for callback in self._subscribers:
            callback()
        return Enqueued(id=inbox_id, duplicate=False)

    async def lease(self, *, limit: int = 1) -> list[LeasedEvent]:
        """Claim up to `limit` deliverable events."""
        now = datetime.now(UTC)
        rows = await self._store.lease_inbox(
            limit=limit,
            now=_stamp(now),
            lease_until=_stamp(now + timedelta(seconds=self._lease_ttl)),
        )
        leased: list[LeasedEvent] = []
        for row in rows:
            try:
                event = InboundEvent.from_json(row["payload"])
            except EventError as exc:
                # Retrying will not help: the row is not going to start
                # parsing. Dead-letter it now rather than five times from now.
                log.error("inbox %s is undeliverable: %s", row["id"], exc)
                await self._store.fail_inbox(int(row["id"]), error=str(exc))
                continue
            leased.append(
                LeasedEvent(id=int(row["id"]), event=event, attempts=int(row["attempts"]))
            )
        return leased

    async def renew(self, ids: Sequence[int]) -> None:
        await self._store.renew_inbox(
            ids, lease_until=_stamp(datetime.now(UTC) + timedelta(seconds=self._lease_ttl))
        )

    async def complete(self, inbox_id: int) -> None:
        await self._store.complete_inbox(inbox_id)

    async def release(self, ids: Sequence[int]) -> None:
        """Hand unfinished work back, as a clean shutdown does."""
        await self._store.release_inbox(ids)

    async def fail(self, item: LeasedEvent, error: BaseException | str) -> bool:
        """Record a failed delivery. False once the row is dead-lettered."""
        reason = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error
        delay = self._backoff.delay_after(item.attempts)
        if delay is None:
            log.error("inbox %s failed %d time(s), giving up: %s", item.id, item.attempts, reason)
            await self._store.fail_inbox(item.id, error=reason)
            return False
        log.warning(
            "inbox %s failed (attempt %d), retrying in %.0fs: %s",
            item.id,
            item.attempts,
            delay,
            reason,
        )
        await self._store.retry_inbox(
            item.id,
            error=reason,
            not_before=_stamp(datetime.now(UTC) + timedelta(seconds=delay)),
        )
        return True

    async def reclaim(self, *, expired_only: bool = False) -> list[str]:
        """Make work a stopped process was holding deliverable again."""
        now = _stamp(datetime.now(UTC)) if expired_only else None
        return [
            f"{row['source']}:{row['external_id']} (attempt {row['attempts']})"
            for row in await self._store.reclaim_inbox(now=now)
        ]

    async def purge(self) -> int:
        """Drop delivered rows old enough that no provider will retry them."""
        return await self._store.purge_inbox(before=_stamp(datetime.now(UTC) - self._retention))

    async def counts(self) -> dict[str, int]:
        return await self._store.inbox_counts()

    async def dead_letters(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._store.inbox_failed(limit)

    async def revive(self) -> int:
        return await self._store.revive_inbox_failed()


class Dispatcher(Drainer[LeasedEvent]):
    """The drainer over the inbox — the design doc's word for it (§3.1).

    No loop of its own: that is generic, and the inbox is the durable half it
    drains. What it adds is the handler's signature. A handler here answers a
    *message*, and has no business knowing which row carried it; sessions
    serialize inside it (#20), and this layer only bounds how many are in
    flight at once.
    """

    def __init__(
        self,
        inbox: Inbox,
        handler: EventHandler,
        *,
        concurrency: int = 8,
        poll_interval: float = 1.0,
        reclaim_on_start: bool = True,
        shutdown_grace: float = 30.0,
        purge_interval: float = 3600.0,
    ) -> None:
        async def work(item: LeasedEvent) -> None:
            await handler(item.event)

        super().__init__(
            inbox,
            work,
            concurrency=concurrency,
            poll_interval=poll_interval,
            reclaim_on_start=reclaim_on_start,
            shutdown_grace=shutdown_grace,
            purge_interval=purge_interval,
        )

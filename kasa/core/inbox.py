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

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kasa.core.events import EventError, InboundEvent
from kasa.store import Store

log = logging.getLogger(__name__)

#: How long a leased row stays off-limits. The dispatcher renews the lease of
#: anything it is still working on, so this bounds how long a *dead* process's
#: work sits unclaimed — not how long a turn is allowed to take.
DEFAULT_LEASE_TTL = 60.0

#: Leases burned before a message stops being retried. It counts leases rather
#: than failures on purpose; see `Store.reclaim_inbox`.
DEFAULT_MAX_ATTEMPTS = 5

#: Retry backoff: 2s, 4s, 8s, … capped. Short at the start because the common
#: failure is a provider blip and the person is still watching the thread.
BACKOFF_BASE = 2.0
BACKOFF_CAP = 300.0

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
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retention: timedelta = DONE_RETENTION,
    ) -> None:
        self._store = store
        self._lease_ttl = lease_ttl
        self._max_attempts = max_attempts
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
        if item.attempts >= self._max_attempts:
            log.error("inbox %s failed %d time(s), giving up: %s", item.id, item.attempts, reason)
            await self._store.fail_inbox(item.id, error=reason)
            return False
        delay = min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (item.attempts - 1))
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

    async def reclaim(self, *, expired_only: bool = False) -> list[dict[str, Any]]:
        """Make work a stopped process was holding deliverable again."""
        now = _stamp(datetime.now(UTC)) if expired_only else None
        return await self._store.reclaim_inbox(now=now)

    async def purge(self) -> int:
        """Drop delivered rows old enough that no provider will retry them."""
        return await self._store.purge_inbox(before=_stamp(datetime.now(UTC) - self._retention))

    async def counts(self) -> dict[str, int]:
        return await self._store.inbox_counts()

    async def dead_letters(self, limit: int = 20) -> list[dict[str, Any]]:
        return await self._store.inbox_failed(limit)

    async def revive(self) -> int:
        return await self._store.revive_inbox_failed()


class Dispatcher:
    """Drains the inbox into a handler, holding each row's lease while it runs.

    One per process. Sessions serialize inside the handler (#20); this layer
    only bounds how many events are in flight at once, and guarantees every
    leased row reaches exactly one of `complete`, `fail` or `release`.
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
        self._inbox = inbox
        self._handler = handler
        self._concurrency = max(1, concurrency)
        self._poll_interval = poll_interval
        #: A sole drainer may assume every leased row belongs to the run that
        #: just died, and take it back immediately instead of waiting out a
        #: lease nobody is holding. Set False the day a second worker exists.
        self._reclaim_on_start = reclaim_on_start
        self._shutdown_grace = shutdown_grace
        self._purge_interval = purge_interval
        self._in_flight: dict[int, asyncio.Task[None]] = {}
        self._woken = asyncio.Event()
        self._stop = asyncio.Event()
        self._last_purge = 0.0
        inbox.subscribe(self.wake)

    def wake(self) -> None:
        """Look for work now rather than at the next poll."""
        self._woken.set()

    def stop(self) -> None:
        self._stop.set()
        self._woken.set()

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    async def run(self) -> None:
        """Drain until `stop()`, then let in-flight work finish."""
        if self._reclaim_on_start:
            for row in await self._inbox.reclaim():
                log.warning(
                    "replaying %s:%s, which a previous run was holding (attempt %s)",
                    row["source"],
                    row["external_id"],
                    row["attempts"],
                )
        keepalive = asyncio.create_task(self._keepalive(), name="inbox-keepalive")
        try:
            while not self._stop.is_set():
                self._woken.clear()
                if await self._start_batch() == 0:
                    await self._idle()
        finally:
            keepalive.cancel()
            await asyncio.gather(keepalive, return_exceptions=True)
            await self._settle()

    async def drain_once(self) -> int:
        """Deliver everything currently due and wait for it.

        The synchronous shape of the loop above, for callers that want the
        queue empty before they look at the result — tests, and `kasa` commands
        that are not the daemon.
        """
        started = await self._start_batch()
        if started:
            await asyncio.gather(*list(self._in_flight.values()))
        return started

    # -- internals -----------------------------------------------------------

    async def _start_batch(self) -> int:
        free = self._concurrency - len(self._in_flight)
        if free <= 0:
            return 0
        items = await self._inbox.lease(limit=free)
        for item in items:
            task = asyncio.create_task(self._deliver(item), name=f"inbox-{item.id}")
            self._in_flight[item.id] = task
            task.add_done_callback(functools.partial(self._forget, item.id))
        return len(items)

    def _forget(self, inbox_id: int, task: asyncio.Task[None]) -> None:
        self._in_flight.pop(inbox_id, None)

    async def _deliver(self, item: LeasedEvent) -> None:
        try:
            await self._handler(item.event)
        except asyncio.CancelledError:
            # Shutdown, not failure. Shielded because we are already being
            # cancelled, and an unshielded await here would leave the row
            # leased until it expired — the delay this whole path exists to
            # avoid. Same reasoning as `Agent._dispatch_all`.
            await asyncio.shield(self._inbox.release([item.id]))
            raise
        except Exception as exc:
            log.exception("handling inbox %s raised", item.id)
            await self._inbox.fail(item, exc)
        else:
            await self._inbox.complete(item.id)

    async def _idle(self) -> None:
        """Wait for an arrival, a stop, a slot to free up, or the poll timeout."""
        signals = [
            asyncio.create_task(self._woken.wait()),
            asyncio.create_task(self._stop.wait()),
        ]
        try:
            await asyncio.wait(
                {*signals, *self._in_flight.values()},
                timeout=self._poll_interval,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for signal in signals:
                signal.cancel()
            await asyncio.gather(*signals, return_exceptions=True)

    async def _keepalive(self) -> None:
        """Hold the leases of work still running, and retire old delivered rows."""
        interval = max(1.0, self._inbox.lease_ttl / 3)
        while True:
            await asyncio.sleep(interval)
            if self._in_flight:
                await self._inbox.renew(list(self._in_flight))
            now = time.monotonic()
            if now - self._last_purge >= self._purge_interval:
                self._last_purge = now
                if purged := await self._inbox.purge():
                    log.debug("purged %d delivered inbox row(s)", purged)

    async def _settle(self) -> None:
        """Let in-flight work finish; cancel what outstays the grace period."""
        tasks = list(self._in_flight.values())
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self._shutdown_grace)
        if pending:
            log.warning("cancelling %d event(s) still running at shutdown", len(pending))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

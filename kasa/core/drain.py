"""Leasing work out of a durable queue, and guaranteeing it comes back.

Two queues need this. The inbox (#19) hands out messages to answer; the job
queue (#26) hands out consolidation passes. What they share is not their SQL —
one dedupes on a provider's event id and never schedules, the other schedules
and never dedupes — but the loop over it: lease a bounded batch, hold the
leases for as long as the work runs, and make sure every leased row reaches
exactly one of complete, fail or release.

Keeping that in one place is also what makes the design's promise cheap: an
out-of-process worker is another `Drainer` over the same table, because
nothing here assumes it is the only one except `reclaim_on_start`, which says
so.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol

from kasa.core.supervise import keep_running

log = logging.getLogger(__name__)


class Leased(Protocol):
    """One unit of leased work. All the drainer needs is how to name it."""

    @property
    def id(self) -> Any: ...


class WorkQueue[ItemT: Leased](Protocol):
    """The durable half: what a drainer leases from and reports back to."""

    @property
    def lease_ttl(self) -> float: ...

    def subscribe(self, callback: Callable[[], None]) -> None: ...

    async def lease(self, *, limit: int) -> list[ItemT]: ...

    async def renew(self, ids: Sequence[Any]) -> None: ...

    async def complete(self, item_id: Any) -> None: ...

    async def release(self, ids: Sequence[Any]) -> None: ...

    async def fail(self, item: ItemT, error: BaseException | str) -> bool: ...

    async def reclaim(self, *, expired_only: bool = False) -> list[str]: ...

    async def purge(self) -> int: ...


class Drainer[ItemT: Leased]:
    """Runs leased work, bounded, and never loses a lease it took."""

    def __init__(
        self,
        queue: WorkQueue[ItemT],
        work: Callable[[ItemT], Awaitable[None]],
        *,
        concurrency: int = 8,
        poll_interval: float = 1.0,
        reclaim_on_start: bool = True,
        shutdown_grace: float = 30.0,
        purge_interval: float = 3600.0,
    ) -> None:
        self._queue = queue
        self._work = work
        self._concurrency = max(1, concurrency)
        self._poll_interval = poll_interval
        #: A sole drainer may assume every leased row belongs to the run that
        #: just died, and take it back immediately instead of waiting out a
        #: lease nobody is holding. Set False the day a second worker exists.
        self._reclaim_on_start = reclaim_on_start
        self._shutdown_grace = shutdown_grace
        self._purge_interval = purge_interval
        self._in_flight: dict[Any, asyncio.Task[None]] = {}
        self._woken = asyncio.Event()
        self._stop = asyncio.Event()
        self._last_purge = 0.0
        queue.subscribe(self.wake)

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
            for description in await self._queue.reclaim():
                log.warning("replaying %s, which a previous run was holding", description)
        keepalive = asyncio.create_task(
            keep_running(
                self._renew_and_purge,
                every=max(1.0, self._queue.lease_ttl / 3),
                name="drain keepalive",
            ),
            name="drain-keepalive",
        )
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
        """Run everything currently due and wait for it.

        The synchronous shape of the loop above, for callers that want the
        queue empty before they look at the result — tests, and the commands
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
        started = 0
        for item in await self._queue.lease(limit=free):
            if item.id in self._in_flight:
                # Our own lease, expired under us while the work is still
                # running: both queues' lease SQL matches `lease_until <= now`
                # on a leased row, which is what replays a dead process's work
                # and cannot tell that process from this one. Running it again
                # would answer the same message twice, and the second task
                # would evict the first from `_in_flight` — leaving it
                # unrenewed, unsettled at shutdown and uncancellable.
                #
                # Skipping is also the renewal that did not happen: the lease
                # just taken has already pushed `lease_until` out by a TTL.
                log.warning("re-leased %s while still running it; skipping", item.id)
                continue
            task = asyncio.create_task(self._deliver(item), name=f"work-{item.id}")
            self._in_flight[item.id] = task
            task.add_done_callback(functools.partial(self._forget, item.id))
            started += 1
        # Skipped rows are deliberately not counted: `run` reads a non-zero
        # return as "there was work", and a batch of nothing but our own
        # in-flight rows would then loop without ever reaching `_idle`.
        return started

    def _forget(self, item_id: Any, task: asyncio.Task[None]) -> None:
        self._in_flight.pop(item_id, None)

    async def _deliver(self, item: ItemT) -> None:
        try:
            await self._work(item)
        except asyncio.CancelledError:
            # Shutdown, not failure. Shielded because we are already being
            # cancelled, and an unshielded await here would leave the row
            # leased until it expired — the delay this path exists to avoid.
            await asyncio.shield(self._queue.release([item.id]))
            raise
        except Exception as exc:
            log.exception("work %s raised", item.id)
            await self._queue.fail(item, exc)
        else:
            await self._queue.complete(item.id)

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

    async def _renew_and_purge(self) -> None:
        """Hold the leases of work still running, and retire old finished rows.

        One tick of the keepalive. A `StoreError` out of either call is a
        transient the supervisor logs and retries; the cost of missing this
        tick is one lease renewal, which is bounded by the guard in
        `_start_batch` rather than by this loop staying alive.
        """
        if self._in_flight:
            await self._queue.renew(list(self._in_flight))
        now = time.monotonic()
        if now - self._last_purge >= self._purge_interval:
            self._last_purge = now
            if purged := await self._queue.purge():
                log.debug("purged %d finished row(s)", purged)

    async def _settle(self) -> None:
        """Let in-flight work finish; cancel what outstays the grace period."""
        tasks = list(self._in_flight.values())
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=self._shutdown_grace)
        if pending:
            log.warning("cancelling %d item(s) still running at shutdown", len(pending))
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

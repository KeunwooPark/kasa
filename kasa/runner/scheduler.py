"""The jobs table, the loop that drains it, and the clock that fills it.

Jobs are rows, not in-memory timers. A restart loses nothing, a job whose
process died runs again when its lease expires, and a second worker is another
drainer over the same table rather than a redesign.

The clock half is deliberately dumb: every tick, each recurring job's *next*
occurrence is inserted under an id derived from its fire time, which makes the
insert idempotent and the whole thing safe to run twice. It also means a
scheduler that was down over a fire time does not stampede on restart — the
occurrence it had already queued still runs, late, and the ones it never
queued never happened.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ulid import ULID

from kasa.core.backoff import Backoff
from kasa.core.drain import Drainer
from kasa.core.supervise import keep_running
from kasa.errors import KasaError
from kasa.runner.cron import Cron
from kasa.store import Store

log = logging.getLogger(__name__)

#: Long, because these are the jobs that call a frontier model several times.
#: The drainer renews while the work runs, so this bounds how long a *dead*
#: process's job waits, not how long a job may take.
DEFAULT_LEASE_TTL = 300.0

#: Three tries, half a minute apart and doubling to an hour. Slower than the
#: inbox's, because nobody is watching a thread waiting for `reflect`, and the
#: usual reason one of these fails is a rate limit or a busy remote.
DEFAULT_BACKOFF = Backoff(max_attempts=3, base=30.0, cap=3600.0)

#: How long a finished job is kept. It answers "when did this last run", which
#: is the first thing anybody asks about a background job.
DONE_RETENTION = timedelta(days=7)

#: How often the clock looks ahead. Well inside a minute, which is cron's own
#: resolution, so an occurrence is always queued before it is due.
TICK_INTERVAL = 30.0


class UnknownJob(KasaError):
    """Nothing is registered to run that kind of job."""


@dataclass(frozen=True, slots=True)
class Job:
    id: str
    kind: str
    payload: dict[str, Any]
    #: Tries so far, this one included. 1 is a first run.
    attempts: int


JobHandler = Callable[[Job], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class JobSpec:
    """A kind of job, what runs it, and when it runs on its own."""

    kind: str
    handler: JobHandler
    #: None means it only runs when something asks for it — `kasa job run`, or
    #: another job. `reindex` is the standing example.
    cron: Cron | None = None


@dataclass(frozen=True, slots=True)
class Queued:
    id: str
    duplicate: bool


def scheduled_id(kind: str, fire_at: datetime) -> str:
    """The id of one occurrence, derived from when it fires.

    This is the whole idempotency story for the clock: two schedulers racing on
    the same tick, or one ticking twice inside a minute, write the same row.
    """
    return f"{kind}@{fire_at.isoformat(timespec='minutes')}"


class JobQueue:
    """The durable half. Same shape as the inbox, over a table that schedules."""

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
        self._kinds: set[str] = set()
        self._subscribers: list[Callable[[], None]] = []

    @property
    def lease_ttl(self) -> float:
        return self._lease_ttl

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._kinds))

    def accept(self, kind: str) -> None:
        """Take jobs of this kind. A worker leases only what it can run."""
        self._kinds.add(kind)

    def subscribe(self, callback: Callable[[], None]) -> None:
        self._subscribers.append(callback)

    async def enqueue(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        run_after: datetime | None = None,
        job_id: str | None = None,
    ) -> Queued:
        identifier = job_id or str(ULID())
        inserted = await self._store.enqueue_job(
            job_id=identifier,
            kind=kind,
            payload=json.dumps(payload, sort_keys=True) if payload else None,
            run_after=_stamp(run_after or datetime.now(UTC)),
        )
        if inserted:
            for callback in self._subscribers:
                callback()
        return Queued(id=identifier, duplicate=not inserted)

    async def lease(self, *, limit: int = 1) -> list[Job]:
        now = datetime.now(UTC)
        rows = await self._store.lease_jobs(
            kinds=self.kinds,
            limit=limit,
            now=_stamp(now),
            lease_until=_stamp(now + timedelta(seconds=self._lease_ttl)),
        )
        return [
            Job(
                id=str(row["id"]),
                kind=str(row["kind"]),
                payload=json.loads(row["payload"]) if row["payload"] else {},
                attempts=int(row["attempts"]),
            )
            for row in rows
        ]

    async def renew(self, ids: Sequence[str]) -> None:
        await self._store.renew_jobs(
            ids, lease_until=_stamp(datetime.now(UTC) + timedelta(seconds=self._lease_ttl))
        )

    async def complete(self, item_id: str) -> None:
        await self._store.complete_job(item_id)

    async def release(self, ids: Sequence[str]) -> None:
        await self._store.release_jobs(ids)

    async def fail(self, item: Job, error: BaseException | str) -> bool:
        """Record a failed run. False once the job is dead-lettered."""
        reason = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else error
        delay = self._backoff.delay_after(item.attempts)
        if delay is None:
            log.error("job %s failed %d time(s), giving up: %s", item.id, item.attempts, reason)
            await self._store.fail_job(item.id, error=reason)
            return False
        log.warning(
            "job %s failed (attempt %d), retrying in %.0fs: %s",
            item.id,
            item.attempts,
            delay,
            reason,
        )
        await self._store.retry_job(
            item.id,
            error=reason,
            not_before=_stamp(datetime.now(UTC) + timedelta(seconds=delay)),
        )
        return True

    async def reclaim(self, *, expired_only: bool = False) -> list[str]:
        now = _stamp(datetime.now(UTC)) if expired_only else None
        return [
            f"{row['kind']} {row['id']} (attempt {row['attempts']})"
            for row in await self._store.reclaim_jobs(now=now)
        ]

    async def purge(self) -> int:
        return await self._store.purge_jobs(before=_stamp(datetime.now(UTC) - self._retention))


class Scheduler:
    """Keeps the jobs table filled from the clock, and empties it."""

    def __init__(
        self,
        store: Store,
        specs: Sequence[JobSpec] = (),
        *,
        concurrency: int = 2,
        poll_interval: float = 5.0,
        tick_interval: float = TICK_INTERVAL,
        lease_ttl: float = DEFAULT_LEASE_TTL,
        backoff: Backoff = DEFAULT_BACKOFF,
        reclaim_on_start: bool = True,
    ) -> None:
        self._store = store
        self._specs: dict[str, JobSpec] = {}
        self._schedule_failures: dict[str, int] = {}
        self.queue = JobQueue(store, lease_ttl=lease_ttl, backoff=backoff)
        self._tick_interval = tick_interval
        # Two at a time by default: these call a frontier model, and the point
        # of running them in the background is that they stay out of the way of
        # the conversation rather than that they finish quickly.
        self._drainer = Drainer(
            self.queue,
            self._run_job,
            concurrency=concurrency,
            poll_interval=poll_interval,
            reclaim_on_start=reclaim_on_start,
        )
        for spec in specs:
            self.register(spec)

    def register(self, spec: JobSpec) -> None:
        self._specs[spec.kind] = spec
        self.queue.accept(spec.kind)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    @property
    def in_flight(self) -> int:
        return self._drainer.in_flight

    async def run(self) -> None:
        """Fill the table from the clock and drain it, until `stop()`."""
        clock = asyncio.create_task(
            keep_running(
                self.schedule_due,
                every=self._tick_interval,
                name="scheduler clock",
                start_now=True,
            ),
            name="scheduler-clock",
        )
        try:
            await self._drainer.run()
        finally:
            clock.cancel()
            await asyncio.gather(clock, return_exceptions=True)

    def stop(self) -> None:
        self._drainer.stop()

    async def trigger(self, kind: str, payload: dict[str, Any] | None = None) -> str:
        """Queue a one-shot to run as soon as something picks it up."""
        self._require(kind)
        return (await self.queue.enqueue(kind, payload)).id

    async def run_now(self, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Queue it, run it here, and report on the row that was queued.

        What `kasa job run` does. The row is still a row, so a job run this way
        leases, retries and dead-letters exactly like one the scheduler picked
        up — the difference is only who is waiting for it.

        Which means draining until *this* row has been attempted, not draining
        once. `lease` takes the oldest runnable rows of the kind, ordered by
        `run_after`, so a row queued just now is behind anything of the same
        kind already due: one pass need not reach it, and reporting on a row
        nobody ran is worse than taking a moment longer. The rows ahead of it
        run on the way past, which is a drainer doing its job.
        """
        job_id = await self.trigger(kind, payload)
        while not await self._attempted(job_id):
            if await self._drainer.drain_once() == 0:
                # Nothing of this kind is runnable at all, so our row is not
                # ours to run — held by another process, or not yet due. Say
                # what is there rather than spin waiting for it.
                break
        return await self._row(job_id)

    async def schedule_due(self, *, now: datetime | None = None) -> list[str]:
        """Queue the next occurrence of every recurring job. Idempotent."""
        moment = now or datetime.now(UTC)
        queued = []
        for spec in self._specs.values():
            if spec.cron is None:
                continue
            # Per spec, because one that cannot be scheduled must not stop the
            # ones registered behind it. An expression that parses and never
            # fires is enough to reach here, and `_specs` is iterated in
            # registration order.
            try:
                fire_at = spec.cron.next_after(moment)
                result = await self.queue.enqueue(
                    spec.kind, run_after=fire_at, job_id=scheduled_id(spec.kind, fire_at)
                )
            except Exception:
                failures = self._schedule_failures.get(spec.kind, 0) + 1
                self._schedule_failures[spec.kind] = failures
                report = log.exception if failures == 1 else log.error
                report("could not schedule %s (%s)", spec.kind, spec.cron.expression)
                continue
            failures = self._schedule_failures.pop(spec.kind, 0)
            if failures:
                log.info(
                    "scheduling %s (%s) recovered after %d failed tick%s",
                    spec.kind,
                    spec.cron.expression,
                    failures,
                    "" if failures == 1 else "s",
                )
            if not result.duplicate:
                log.debug("queued %s for %s", spec.kind, fire_at)
                queued.append(result.id)
        return queued

    # -- internals -----------------------------------------------------------

    async def _row(self, job_id: str) -> dict[str, Any]:
        rows = await self._store.raw("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return rows[0]

    async def _attempted(self, job_id: str) -> bool:
        """Whether this row has been leased. A lease is what runs it, and a
        drainer that leased it also waited for it, so the row is settled."""
        return bool((await self._row(job_id))["attempts"])

    def _require(self, kind: str) -> JobSpec:
        if (spec := self._specs.get(kind)) is None:
            known = ", ".join(self.kinds) or "none"
            raise UnknownJob(f"no job named {kind!r}; registered: {known}")
        return spec

    async def _run_job(self, job: Job) -> None:
        await self._require(job.kind).handler(job)


def _stamp(moment: datetime) -> str:
    return moment.isoformat(timespec="milliseconds")

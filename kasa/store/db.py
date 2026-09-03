"""SQLite access and the forward-only migration runner."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import aiosqlite
from pydantic import TypeAdapter
from ulid import ULID

from kasa.errors import KasaError, StoreError
from kasa.llm.cost import CallRecord
from kasa.llm.types import ContentBlock, Message

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BLOCKS = TypeAdapter(tuple[ContentBlock, ...])


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class _Serial:
    """Mutual exclusion over the connection, re-entrant within one task.

    `aiosqlite` runs every statement on a single worker thread, which is easy
    to mistake for serialization. It is not: a coroutine yields between
    `execute` and `fetchall`, and again when a cursor closes, so two tasks
    sharing a connection interleave *inside* each other's statements. SQLite
    then refuses the commit that lands there —

        OperationalError: cannot commit transaction - SQL statements in progress

    — which is #101, and it is what a second concurrent conversation produced
    within a couple of hundred messages.

    It stayed hidden for as long as nothing ran two turns at once. It is also
    not only about commits: `append_message` reads `MAX(seq) + 1` and then
    inserts it, and that pair is correct only while nothing else appends to
    the same session in between.

    So the rule is the blunt one: every `Store` method is serialized against
    every other. The statements are microseconds against a model call that
    takes seconds, and WAL still lets a *separate* connection read throughout.

    Re-entrant because the compound methods are written in terms of the simple
    ones — `add_observation` looks up its session, `append_messages` appends
    one at a time — and a plain lock would deadlock on the first of them.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0

    async def __aenter__(self) -> None:
        task = asyncio.current_task()
        if task is not None and task is self._owner:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._depth = 1

    async def __aexit__(self, *exc: object) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


class Store:
    """Owns the connection. One instance per process."""

    def __init__(self, conn: aiosqlite.Connection, path: str) -> None:
        self._conn = conn
        self.path = path
        self._serial = _Serial()

    @classmethod
    async def open(cls, path: str | Path) -> Self:
        """Open the database, or raise `StoreError` having closed it again.

        The close is not tidiness. `aiosqlite.connect` starts a worker thread
        that is not a daemon, and opening the file is lazy, so the first
        statement is where a corrupt database announces itself — by which point
        the thread exists. Leaking it left the process alive at interpreter
        shutdown after the error had already been printed, so `kasa doctor` on
        a truncated database hung instead of failing (#87). `cli.py` guards the
        same hazard around the registry; this is the layer it starts at.

        The connect itself is inside the guard as well. It cleans up its own
        thread, but it can still fail — a directory where the file should be —
        and that failure should read like the other one rather than like a
        traceback. `close()` on a connection that never connected is a no-op.
        """
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        conn = aiosqlite.connect(target)
        try:
            await conn
            conn.row_factory = aiosqlite.Row
            if target != ":memory:":
                # WAL is what lets the scheduler read while a turn is writing.
                await conn.execute("PRAGMA journal_mode = WAL")
            await conn.execute("PRAGMA synchronous = NORMAL")
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA busy_timeout = 5000")
            await conn.commit()
            store = cls(conn, target)
            await store.migrate()
        except BaseException as exc:
            await conn.close()
            if isinstance(exc, sqlite3.Error):
                raise StoreError(
                    f"{target} is not a usable database ({exc}). It is derived from the "
                    "memory repo — delete it and run `kasa reindex` to rebuild it."
                ) from exc
            raise
        return store

    async def close(self) -> None:
        await self._conn.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- migrations ----------------------------------------------------------

    async def migrate(self) -> list[str]:
        """Apply pending migrations. Idempotent."""
        async with self._serial:
            await self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_version ("
                " name TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            await self._conn.commit()

            async with self._conn.execute("SELECT name FROM schema_version") as cur:
                applied = {row["name"] for row in await cur.fetchall()}

            pending = sorted(p for p in MIGRATIONS_DIR.glob("*.sql") if p.name not in applied)
            for migration in pending:
                try:
                    await self._conn.executescript(migration.read_text())
                    await self._conn.execute(
                        "INSERT INTO schema_version (name, applied_at) VALUES (?, ?)",
                        (migration.name, _now()),
                    )
                    await self._conn.commit()
                except Exception as exc:
                    await self._conn.rollback()
                    raise KasaError(f"migration {migration.name} failed: {exc}") from exc
            return [p.name for p in pending]

        # -- sessions ------------------------------------------------------------

    async def ensure_session(
        self, session_id: str, *, surface: str, scope: str = "workspace"
    ) -> None:
        async with self._serial:
            now = _now()
            await self._conn.execute(
                "INSERT INTO sessions (id, surface, scope, created_at, last_active)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET last_active = excluded.last_active",
                (session_id, surface, scope, now, now),
            )
            await self._conn.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self._serial:
            async with self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None

    async def clear_session(self, session_id: str) -> int:
        async with self._serial:
            cur = await self._conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            await self._conn.commit()
            return cur.rowcount

        # -- messages ------------------------------------------------------------

    async def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        author: str | None = None,
        tokens: int | None = None,
    ) -> str:
        async with self._serial:
            message_id = str(ULID())
            async with self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM messages WHERE session_id = ?",
                (session_id,),
            ) as cur:
                row = await cur.fetchone()
            seq = int(row["next"]) if row else 1

            await self._conn.execute(
                "INSERT INTO messages"
                " (id, session_id, seq, role, author, content, tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    seq,
                    message.role,
                    author,
                    _BLOCKS.dump_json(message.content).decode(),
                    tokens,
                    _now(),
                ),
            )
            await self._conn.execute(
                "UPDATE sessions SET last_active = ? WHERE id = ?", (_now(), session_id)
            )
            await self._conn.commit()
            return message_id

    async def append_messages(self, session_id: str, messages: Sequence[Message]) -> list[str]:
        async with self._serial:
            return [await self.append_message(session_id, m) for m in messages]

    async def recent_messages(self, session_id: str, limit: int = 100) -> list[Message]:
        """Most recent `limit` messages, oldest first."""
        async with self._serial:
            async with self._conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY seq DESC LIMIT ?",
                (session_id, limit),
            ) as cur:
                rows = list(await cur.fetchall())
            return [
                Message(role=row["role"], content=_BLOCKS.validate_json(row["content"]))
                for row in reversed(rows)
            ]

    async def message_count(self, session_id: str) -> int:
        async with self._serial:
            async with self._conn.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
            return int(row["n"]) if row else 0

        # -- accounting ----------------------------------------------------------

    async def record_call(self, record: CallRecord) -> None:
        async with self._serial:
            await self._conn.execute(
                "INSERT INTO llm_calls"
                " (created_at, role, provider, model, tag, input_tokens, output_tokens,"
                "  cache_read_tokens, cache_write_tokens, cost_usd, latency_ms, ok, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    record.role,
                    record.provider,
                    record.model,
                    record.tag,
                    record.usage.input_tokens,
                    record.usage.output_tokens,
                    record.usage.cache_read_tokens,
                    record.usage.cache_write_tokens,
                    record.cost_usd,
                    record.latency_ms,
                    int(record.ok),
                    record.error,
                ),
            )
            await self._conn.commit()

    async def cost_summary(self, *, since: str | None = None) -> list[dict[str, Any]]:
        async with self._serial:
            clause = "WHERE created_at >= ?" if since else ""
            params: tuple[Any, ...] = (since,) if since else ()
            async with self._conn.execute(
                f"SELECT role, model, COUNT(*) AS calls,"
                f" SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
                f" SUM(cache_read_tokens) AS cache_read_tokens,"
                f" SUM(cost_usd) AS cost_usd"
                f" FROM llm_calls {clause} GROUP BY role, model ORDER BY calls DESC",
                params,
            ) as cur:
                return [dict(row) for row in await cur.fetchall()]

    async def raw(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run a query and return its rows.

        Together with `write`, this is what lets the derived index own its own
        SQL in `kasa/memory/index.py` instead of pushing another dozen methods
        into this class. Durable state still goes through the typed methods
        above; these two are for the rebuildable half.
        """
        async with self._serial, self._conn.execute(sql, params) as cur:
            return [dict(row) for row in await cur.fetchall()]

    async def write(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        async with self._serial:
            cur = await self._conn.execute(sql, params)
            await self._conn.commit()
            return int(cur.rowcount)

    async def write_many(self, sql: str, rows: Sequence[tuple[Any, ...]]) -> None:
        async with self._serial:
            if not rows:
                return
            await self._conn.executemany(sql, rows)
            await self._conn.commit()

        # -- episodes ------------------------------------------------------------

    async def open_episode(self, session_id: str) -> dict[str, Any] | None:
        """The episode this session is currently accumulating into, if any.

        Read when an actor wakes for a session, so a turn knows what it is part
        of. Opening and closing episodes belongs to `episode_close` (#27); this
        only reports what is already there, and `None` is the normal answer
        until that job exists.
        """
        async with self._serial:
            async with self._conn.execute(
                "SELECT * FROM episodes WHERE session_id = ? AND state = 'open'"
                " ORDER BY started_at DESC LIMIT 1",
                (session_id,),
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None

        # -- observations --------------------------------------------------------

    async def add_observation(
        self,
        *,
        subject: str,
        claim: str,
        kind: str,
        scope: str,
        session_id: str | None = None,
        episode_id: str | None = None,
        confidence: float = 0.7,
        source_refs: Sequence[str] = (),
    ) -> str:
        """Record a candidate fact. Nothing durable happens until `promote` runs."""
        async with self._serial:
            observation_id = str(ULID())
            # An observation outlives the session that produced it, and losing the
            # fact because the bookkeeping link is missing would be the wrong trade.
            if session_id is not None and await self.get_session(session_id) is None:
                session_id = None
            await self._conn.execute(
                "INSERT INTO observations"
                " (id, episode_id, session_id, subject, claim, kind, confidence, scope,"
                "  source_refs, state, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    observation_id,
                    episode_id,
                    session_id,
                    subject,
                    claim,
                    kind,
                    confidence,
                    scope,
                    json_dumps(list(source_refs)),
                    _now(),
                ),
            )
            await self._conn.commit()
            return observation_id

    async def pending_observations(self, limit: int = 100) -> list[dict[str, Any]]:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM observations WHERE state = 'pending' ORDER BY created_at LIMIT ?",
                (limit,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

        # -- inbox ---------------------------------------------------------------
        #
        # The state machine, in one place, because every method below is a move in
        # it: pending -> leased -> done, with leased -> pending on expiry or a
        # retry, and pending -> failed once the attempts run out. `lease_until`
        # means "not before" in every state it is set in, which is what lets one
        # index answer "what is deliverable now".

    async def enqueue_inbox(
        self, *, source: str, external_id: str, payload: str
    ) -> tuple[int, bool]:
        """Queue an event, or report the id of the one already queued.

        The insert is the dedupe. Two adapters racing on the same provider
        retry both end up here; one inserts, the other conflicts, and both are
        told which row won, so neither has to decide what to do about it.
        """
        async with self._serial:
            cur = await self._conn.execute(
                "INSERT INTO inbox (source, external_id, payload, received_at, state)"
                " VALUES (?, ?, ?, ?, 'pending')"
                " ON CONFLICT (source, external_id) DO NOTHING",
                (source, external_id, payload, _now()),
            )
            await self._conn.commit()
            if cur.rowcount and cur.lastrowid is not None:
                return int(cur.lastrowid), False
            async with self._conn.execute(
                "SELECT id FROM inbox WHERE source = ? AND external_id = ?", (source, external_id)
            ) as existing:
                row = await existing.fetchone()
            if row is None:  # pragma: no cover - the conflict proves the row exists
                raise StoreError(f"inbox row for {source}:{external_id} vanished mid-enqueue")
            return int(row["id"]), True

    async def lease_inbox(
        self,
        *,
        limit: int,
        now: str,
        lease_until: str,
        exclude: Sequence[int] = (),
        only: Sequence[int] = (),
    ) -> list[dict[str, Any]]:
        """Claim up to `limit` deliverable rows, oldest first.

        One statement, so two drainers cannot claim the same row: SQLite makes
        the UPDATE atomic and RETURNING hands back exactly what it changed.

        `attempts` counts leases, not failures. A message that kills the process
        that is answering it leaves no failure behind to count, and counting
        only failures is how such a message loops forever.

        Which is why `exclude` lives here rather than in the drainer.
        `lease_until <= now` on a leased row is how a dead process's work
        replays, and it cannot tell that process from this one — so a caller
        passes the ids it is still running and those rows are not offered back.
        Dropping them from the result afterwards would be too late: this
        statement has already spent the attempt, and a caller re-leasing its
        own work is not a process that died holding it (#126).
        """
        skip = f" AND id NOT IN ({', '.join('?' for _ in exclude)})" if exclude else ""
        pick = f" AND id IN ({', '.join('?' for _ in only)})" if only else ""
        async with self._serial:
            async with self._conn.execute(
                "UPDATE inbox SET state = 'leased', lease_until = ?, attempts = attempts + 1"
                " WHERE id IN ("
                "   SELECT id FROM inbox"
                "   WHERE ((state = 'pending' AND (lease_until IS NULL OR lease_until <= ?))"
                "      OR (state = 'leased' AND lease_until <= ?))"
                f"{skip}{pick}"
                "   ORDER BY id LIMIT ?"
                " )"
                " RETURNING id, payload, attempts",
                (lease_until, now, now, *exclude, *only, limit),
            ) as cur:
                rows = [dict(row) for row in await cur.fetchall()]
            await self._conn.commit()
            return rows

    async def renew_inbox(self, ids: Sequence[int], *, lease_until: str) -> None:
        """Push the expiry out on rows still being worked on."""
        async with self._serial:
            if not ids:
                return
            placeholders = ", ".join("?" for _ in ids)
            await self._conn.execute(
                "UPDATE inbox SET lease_until = ? WHERE state = 'leased'"
                f" AND id IN ({placeholders})",
                (lease_until, *ids),
            )
            await self._conn.commit()

    async def complete_inbox(self, inbox_id: int) -> None:
        async with self._serial:
            await self._conn.execute(
                "UPDATE inbox SET state = 'done', lease_until = NULL, last_error = NULL"
                " WHERE id = ?",
                (inbox_id,),
            )
            await self._conn.commit()

    async def retry_inbox(self, inbox_id: int, *, error: str, not_before: str) -> None:
        async with self._serial:
            await self._conn.execute(
                "UPDATE inbox SET state = 'pending', lease_until = ?, last_error = ? WHERE id = ?",
                (not_before, error, inbox_id),
            )
            await self._conn.commit()

    async def fail_inbox(self, inbox_id: int, *, error: str) -> None:
        """Dead-letter a row. Nothing retries it until somebody says so."""
        async with self._serial:
            await self._conn.execute(
                "UPDATE inbox SET state = 'failed', lease_until = NULL, last_error = ?"
                " WHERE id = ?",
                (error, inbox_id),
            )
            await self._conn.commit()

    async def release_inbox(self, ids: Sequence[int]) -> None:
        """Hand rows back unfinished, as a clean shutdown does.

        The attempt is given back with them. Stopping the daemon is not the
        message's fault, and a queue that dead-letters its backlog after five
        deploys is worse than no bound at all.
        """
        async with self._serial:
            if not ids:
                return
            placeholders = ", ".join("?" for _ in ids)
            await self._conn.execute(
                "UPDATE inbox SET state = 'pending', lease_until = NULL,"
                " attempts = MAX(attempts - 1, 0)"
                f" WHERE state = 'leased' AND id IN ({placeholders})",
                tuple(ids),
            )
            await self._conn.commit()

    async def reclaim_inbox(self, *, now: str | None = None) -> list[dict[str, Any]]:
        """Make rows a stopped process was holding deliverable again.

        With `now`, only rows whose lease has already expired — the safe
        reading when somebody else may still be working. Without it, every
        leased row, which is what a sole drainer may assume at startup.

        The attempt is *not* given back. A message that kills whatever is
        answering it leaves no failure behind to count, so the lease it burned
        is the only record that it was tried, and giving it back is how such a
        message loops forever.
        """
        async with self._serial:
            clause = " AND lease_until <= ?" if now is not None else ""
            params: tuple[Any, ...] = (now,) if now is not None else ()
            async with self._conn.execute(
                "UPDATE inbox SET state = 'pending', lease_until = NULL"
                f" WHERE state = 'leased'{clause}"
                " RETURNING id, source, external_id, attempts",
                params,
            ) as cur:
                rows = [dict(row) for row in await cur.fetchall()]
            await self._conn.commit()
            return rows

    async def inbox_counts(self) -> dict[str, int]:
        async with (
            self._serial,
            self._conn.execute("SELECT state, COUNT(*) AS n FROM inbox GROUP BY state") as cur,
        ):
            return {row["state"]: int(row["n"]) for row in await cur.fetchall()}

    async def inbox_failed(self, limit: int = 20) -> list[dict[str, Any]]:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT id, source, external_id, received_at, attempts, last_error FROM inbox"
                " WHERE state = 'failed' ORDER BY id LIMIT ?",
                (limit,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def revive_inbox_failed(self) -> int:
        """Put every dead letter back in the queue with a fresh attempt budget."""
        async with self._serial:
            cur = await self._conn.execute(
                "UPDATE inbox SET state = 'pending', attempts = 0, lease_until = NULL"
                " WHERE state = 'failed'"
            )
            await self._conn.commit()
            return int(cur.rowcount)

    async def purge_inbox(self, *, before: str) -> int:
        """Drop delivered rows older than `before`.

        Only `done` rows, and only old ones. A delivered row is still the
        dedupe record for its event id, so purging eagerly is how a late
        provider retry gets answered a second time.
        """
        async with self._serial:
            cur = await self._conn.execute(
                "DELETE FROM inbox WHERE state = 'done' AND received_at < ?", (before,)
            )
            await self._conn.commit()
            return int(cur.rowcount)

        # -- jobs ----------------------------------------------------------------

    #
    # The same state machine as `inbox` above, over a table that schedules
    # instead of deduping. `run_after` carries both meanings a job needs — when
    # it first becomes due and when a failed one may be tried again — so the
    # drainer's query stays one statement.

    async def enqueue_job(
        self, *, job_id: str, kind: str, payload: str | None, run_after: str
    ) -> bool:
        """Queue a job. False when that exact id was already there.

        The id is the idempotency: a scheduled run is `kind@fire-time`, so two
        schedulers racing on the same tick — or one scheduler polling twice
        within a minute — insert the same row and only the first counts.
        """
        async with self._serial:
            cur = await self._conn.execute(
                "INSERT INTO jobs (id, kind, payload, run_after, state, created_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?)"
                " ON CONFLICT (id) DO NOTHING",
                (job_id, kind, payload, run_after, _now()),
            )
            await self._conn.commit()
            return bool(cur.rowcount)

    async def lease_jobs(
        self,
        *,
        kinds: Sequence[str],
        limit: int,
        now: str,
        lease_until: str,
        exclude: Sequence[str] = (),
        only: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Claim up to `limit` runnable jobs of the kinds this worker knows.

        The `kind` filter is what lets a second process take only the work it
        has handlers for, which is the out-of-process worker the design asks
        this model to reach without a redesign.

        `exclude` is the ids the caller is still running. See `lease_inbox` for
        why it belongs in the statement that spends the attempt rather than in
        the drainer that reads what comes back.

        `only` narrows the other way, to specific rows. `kasa job run` wants
        the one row it queued and not the backlog of its kind sitting in front
        of it, and the row still has to be *leased* to be run — so the
        narrowing belongs in the same statement rather than in a caller that
        would have to lease the others to find out what they were (#127).
        """
        if not kinds:
            return []
        placeholders = ", ".join("?" for _ in kinds)
        skip = f" AND id NOT IN ({', '.join('?' for _ in exclude)})" if exclude else ""
        pick = f" AND id IN ({', '.join('?' for _ in only)})" if only else ""
        async with self._serial:
            async with self._conn.execute(
                "UPDATE jobs SET state = 'leased', lease_until = ?, attempts = attempts + 1"
                " WHERE id IN ("
                "   SELECT id FROM jobs"
                f"   WHERE kind IN ({placeholders})"
                "     AND ((state = 'pending' AND run_after <= ?)"
                "          OR (state = 'leased' AND lease_until <= ?))"
                f"{skip}{pick}"
                "   ORDER BY run_after LIMIT ?"
                " )"
                " RETURNING id, kind, payload, attempts",
                (lease_until, *kinds, now, now, *exclude, *only, limit),
            ) as cur:
                rows = [dict(row) for row in await cur.fetchall()]
            await self._conn.commit()
            return rows

    async def renew_jobs(self, ids: Sequence[str], *, lease_until: str) -> None:
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        async with self._serial:
            await self._conn.execute(
                "UPDATE jobs SET lease_until = ? WHERE state = 'leased'"
                f" AND id IN ({placeholders})",
                (lease_until, *ids),
            )
            await self._conn.commit()

    async def complete_job(self, job_id: str) -> None:
        async with self._serial:
            await self._conn.execute(
                "UPDATE jobs SET state = 'done', lease_until = NULL, last_error = NULL,"
                " finished_at = ? WHERE id = ?",
                (_now(), job_id),
            )
            await self._conn.commit()

    async def retry_job(self, job_id: str, *, error: str, not_before: str) -> None:
        async with self._serial:
            await self._conn.execute(
                "UPDATE jobs SET state = 'pending', lease_until = NULL, run_after = ?,"
                " last_error = ? WHERE id = ?",
                (not_before, error, job_id),
            )
            await self._conn.commit()

    async def fail_job(self, job_id: str, *, error: str) -> None:
        """Dead-letter a job. Nothing retries it until somebody says so."""
        async with self._serial:
            await self._conn.execute(
                "UPDATE jobs SET state = 'failed', lease_until = NULL, last_error = ?,"
                " finished_at = ? WHERE id = ?",
                (error, _now(), job_id),
            )
            await self._conn.commit()

    async def release_jobs(self, ids: Sequence[str]) -> None:
        """Hand jobs back unfinished, as a clean shutdown does, attempt included."""
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        async with self._serial:
            await self._conn.execute(
                "UPDATE jobs SET state = 'pending', lease_until = NULL,"
                " attempts = MAX(attempts - 1, 0)"
                f" WHERE state = 'leased' AND id IN ({placeholders})",
                tuple(ids),
            )
            await self._conn.commit()

    async def reclaim_jobs(self, *, now: str | None = None) -> list[dict[str, Any]]:
        """Make jobs a stopped process was holding runnable again.

        The attempt is not given back, for the reason `reclaim_inbox` gives:
        a job that kills the process running it leaves no failure to count.
        """
        clause = " AND lease_until <= ?" if now is not None else ""
        params: tuple[Any, ...] = (now,) if now is not None else ()
        async with self._serial:
            async with self._conn.execute(
                "UPDATE jobs SET state = 'pending', lease_until = NULL"
                f" WHERE state = 'leased'{clause}"
                " RETURNING id, kind, attempts",
                params,
            ) as cur:
                rows = [dict(row) for row in await cur.fetchall()]
            await self._conn.commit()
            return rows

    async def job_overview(self) -> list[dict[str, Any]]:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT kind, state, COUNT(*) AS n, MAX(finished_at) AS last_run"
                " FROM jobs GROUP BY kind, state ORDER BY kind, state"
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def failed_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT id, kind, attempts, last_error FROM jobs"
                " WHERE state = 'failed' ORDER BY finished_at LIMIT ?",
                (limit,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def revive_failed_jobs(self) -> int:
        async with self._serial:
            cur = await self._conn.execute(
                "UPDATE jobs SET state = 'pending', attempts = 0, lease_until = NULL,"
                " run_after = ? WHERE state = 'failed'",
                (_now(),),
            )
            await self._conn.commit()
            return int(cur.rowcount)

    async def purge_jobs(self, *, before: str) -> int:
        """Drop finished jobs that ran before `before`.

        Only `done` rows. A dead letter is a thing somebody has to look at, and
        a scheduled id is not a dedupe record for anything once it has run —
        the next occurrence has a different id.
        """
        async with self._serial:
            cur = await self._conn.execute(
                "DELETE FROM jobs WHERE state = 'done' AND finished_at < ?", (before,)
            )
            await self._conn.commit()
            return int(cur.rowcount)

    # -- leases --------------------------------------------------------------

    async def take_lease(
        self, name: str, *, holder: str, job: str | None, ttl_seconds: float
    ) -> None:
        """Record that `holder` holds `name`. The flock is what enforces it."""
        async with self._serial:
            now = datetime.now(UTC)
            await self._conn.execute(
                "INSERT INTO leases (name, holder, job, acquired_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(name) DO UPDATE SET"
                " holder = excluded.holder, job = excluded.job,"
                " acquired_at = excluded.acquired_at, expires_at = excluded.expires_at",
                (
                    name,
                    holder,
                    job,
                    now.isoformat(timespec="milliseconds"),
                    (now + timedelta(seconds=ttl_seconds)).isoformat(timespec="milliseconds"),
                ),
            )
            await self._conn.commit()

    async def release_lease(self, name: str) -> bool:
        async with self._serial:
            cur = await self._conn.execute("DELETE FROM leases WHERE name = ?", (name,))
            await self._conn.commit()
            return cur.rowcount > 0

    async def get_lease(self, name: str) -> dict[str, Any] | None:
        async with self._serial:
            async with self._conn.execute("SELECT * FROM leases WHERE name = ?", (name,)) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

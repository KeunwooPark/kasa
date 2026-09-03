"""SQLite access and the forward-only migration runner."""

from __future__ import annotations

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


class Store:
    """Owns the connection. One instance per process."""

    def __init__(self, conn: aiosqlite.Connection, path: str) -> None:
        self._conn = conn
        self.path = path

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
        now = _now()
        await self._conn.execute(
            "INSERT INTO sessions (id, surface, scope, created_at, last_active)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET last_active = excluded.last_active",
            (session_id, surface, scope, now, now),
        )
        await self._conn.commit()

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self._conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None

    async def clear_session(self, session_id: str) -> int:
        cur = await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
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
        return [await self.append_message(session_id, m) for m in messages]

    async def recent_messages(self, session_id: str, limit: int = 100) -> list[Message]:
        """Most recent `limit` messages, oldest first."""
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
        async with self._conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
        return int(row["n"]) if row else 0

    # -- accounting ----------------------------------------------------------

    async def record_call(self, record: CallRecord) -> None:
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
        async with self._conn.execute(sql, params) as cur:
            return [dict(row) for row in await cur.fetchall()]

    async def write(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        cur = await self._conn.execute(sql, params)
        await self._conn.commit()
        return int(cur.rowcount)

    async def write_many(self, sql: str, rows: Sequence[tuple[Any, ...]]) -> None:
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
        async with self._conn.execute(
            "SELECT * FROM observations WHERE state = 'pending' ORDER BY created_at LIMIT ?",
            (limit,),
        ) as cur:
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

    async def lease_inbox(self, *, limit: int, now: str, lease_until: str) -> list[dict[str, Any]]:
        """Claim up to `limit` deliverable rows, oldest first.

        One statement, so two drainers cannot claim the same row: SQLite makes
        the UPDATE atomic and RETURNING hands back exactly what it changed.

        `attempts` counts leases, not failures. A message that kills the process
        that is answering it leaves no failure behind to count, and counting
        only failures is how such a message loops forever.
        """
        async with self._conn.execute(
            "UPDATE inbox SET state = 'leased', lease_until = ?, attempts = attempts + 1"
            " WHERE id IN (("
            "   SELECT id FROM inbox"
            "   WHERE (state = 'pending' AND (lease_until IS NULL OR lease_until <= ?))"
            "      OR (state = 'leased' AND lease_until <= ?)"
            "   ORDER BY id LIMIT ?"
            " ))"
            " RETURNING id, payload, attempts",
            (lease_until, now, now, limit),
        ) as cur:
            rows = [dict(row) for row in await cur.fetchall()]
        await self._conn.commit()
        return rows

    async def renew_inbox(self, ids: Sequence[int], *, lease_until: str) -> None:
        """Push the expiry out on rows still being worked on."""
        if not ids:
            return
        placeholders = ", ".join("?" for _ in ids)
        await self._conn.execute(
            f"UPDATE inbox SET lease_until = ? WHERE state = 'leased' AND id IN ({placeholders})",
            (lease_until, *ids),
        )
        await self._conn.commit()

    async def complete_inbox(self, inbox_id: int) -> None:
        await self._conn.execute(
            "UPDATE inbox SET state = 'done', lease_until = NULL, last_error = NULL WHERE id = ?",
            (inbox_id,),
        )
        await self._conn.commit()

    async def retry_inbox(self, inbox_id: int, *, error: str, not_before: str) -> None:
        await self._conn.execute(
            "UPDATE inbox SET state = 'pending', lease_until = ?, last_error = ? WHERE id = ?",
            (not_before, error, inbox_id),
        )
        await self._conn.commit()

    async def fail_inbox(self, inbox_id: int, *, error: str) -> None:
        """Dead-letter a row. Nothing retries it until somebody says so."""
        await self._conn.execute(
            "UPDATE inbox SET state = 'failed', lease_until = NULL, last_error = ? WHERE id = ?",
            (error, inbox_id),
        )
        await self._conn.commit()

    async def release_inbox(self, ids: Sequence[int]) -> None:
        """Hand rows back unfinished, as a clean shutdown does.

        The attempt is given back with them. Stopping the daemon is not the
        message's fault, and a queue that dead-letters its backlog after five
        deploys is worse than no bound at all.
        """
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
        async with self._conn.execute(
            "SELECT state, COUNT(*) AS n FROM inbox GROUP BY state"
        ) as cur:
            return {row["state"]: int(row["n"]) for row in await cur.fetchall()}

    async def inbox_failed(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._conn.execute(
            "SELECT id, source, external_id, received_at, attempts, last_error FROM inbox"
            " WHERE state = 'failed' ORDER BY id LIMIT ?",
            (limit,),
        ) as cur:
            return [dict(row) for row in await cur.fetchall()]

    async def revive_inbox_failed(self) -> int:
        """Put every dead letter back in the queue with a fresh attempt budget."""
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
        cur = await self._conn.execute(
            "DELETE FROM inbox WHERE state = 'done' AND received_at < ?", (before,)
        )
        await self._conn.commit()
        return int(cur.rowcount)

    # -- leases --------------------------------------------------------------

    async def take_lease(
        self, name: str, *, holder: str, job: str | None, ttl_seconds: float
    ) -> None:
        """Record that `holder` holds `name`. The flock is what enforces it."""
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
        cur = await self._conn.execute("DELETE FROM leases WHERE name = ?", (name,))
        await self._conn.commit()
        return cur.rowcount > 0

    async def get_lease(self, name: str) -> dict[str, Any] | None:
        async with self._conn.execute("SELECT * FROM leases WHERE name = ?", (name,)) as cur:
            row = await cur.fetchone()
        return dict(row) if row else None


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

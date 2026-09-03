"""SQLite access and the forward-only migration runner."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import aiosqlite
from pydantic import TypeAdapter
from ulid import ULID

from kasa.errors import KasaError
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
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(target)
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

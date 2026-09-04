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
from kasa.memory.observation import ObservationDraft
from kasa.memory.subject import normalize_subject

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BLOCKS = TypeAdapter(tuple[ContentBlock, ...])

#: One episode, with the two things every consolidation job asks about it that
#: are not on the row: how much conversation it holds, and who is allowed to
#: see what comes out of it. Written once because three queries select it, and
#: three hand-written joins are three chances to forget the scope.
_EPISODE_VIEW = """
SELECT e.id, e.session_id, e.started_at, e.ended_at, e.state, e.summary,
       e.signal_score, s.scope,
       COUNT(m.id) AS messages,
       COALESCE(MAX(m.created_at), e.started_at) AS last_message
FROM episodes e
JOIN sessions s ON s.id = e.session_id
LEFT JOIN messages m ON m.episode_id = e.id
"""


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
            # Every message belongs to an episode, and this is the only place
            # that can know which: the episode a message arrived during is the
            # open one at the moment it is written, and nothing downstream can
            # reconstruct that from timestamps once a later one has opened.
            episode_id = await self.ensure_episode(session_id)

            await self._conn.execute(
                "INSERT INTO messages"
                " (id, session_id, episode_id, seq, role, author, content, tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    session_id,
                    episode_id,
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
        of. `None` before the session's first message: episodes are opened by
        the write path, not by looking at one.
        """
        async with self._serial:
            async with self._conn.execute(
                "SELECT * FROM episodes WHERE session_id = ? AND state = 'open'"
                " ORDER BY started_at DESC LIMIT 1",
                (session_id,),
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None

    async def ensure_episode(self, session_id: str) -> str:
        """The session's open episode, opening one if it has none.

        Opening is cheap and unconditional; deciding when a segment has *ended*
        is the expensive judgement, and it belongs to `episode_close`. So there
        is no "start an episode" decision anywhere in the write path — a
        message arrives, and it lands in whatever is open.
        """
        async with self._serial:
            if (existing := await self.open_episode(session_id)) is not None:
                return str(existing["id"])
            episode_id = str(ULID())
            await self._conn.execute(
                "INSERT INTO episodes (id, session_id, started_at, state) VALUES (?, ?, ?, 'open')",
                (episode_id, session_id, _now()),
            )
            await self._conn.commit()
            return episode_id

    async def episode(self, episode_id: str) -> dict[str, Any] | None:
        """One episode, with the counts and the scope a consolidation job needs."""
        async with self._serial:
            async with self._conn.execute(
                f"{_EPISODE_VIEW} WHERE e.id = ? GROUP BY e.id", (episode_id,)
            ) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None

    async def due_episodes(
        self, *, idle_before: str, max_messages: int, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Open episodes that have gone quiet or grown long.

        `COALESCE(MAX(m.created_at), e.started_at)` is what closes an episode
        that never received a message: a session opened and abandoned leaves
        one behind, and an episode nothing can ever close is an episode every
        later sweep pays to look at.
        """
        async with (
            self._serial,
            self._conn.execute(
                f"{_EPISODE_VIEW} WHERE e.state = 'open' GROUP BY e.id"
                " HAVING COUNT(m.id) >= ? OR COALESCE(MAX(m.created_at), e.started_at) <= ?"
                " ORDER BY e.started_at LIMIT ?",
                (max_messages, idle_before, limit),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def open_episodes_of(self, session_id: str) -> list[dict[str, Any]]:
        """Every open episode of one session, however recent.

        What an explicit session end closes. There is normally exactly one; the
        list is because "normally" is not a guarantee worth writing a caller
        against.
        """
        async with (
            self._serial,
            self._conn.execute(
                f"{_EPISODE_VIEW} WHERE e.state = 'open' AND e.session_id = ?"
                " GROUP BY e.id ORDER BY e.started_at",
                (session_id,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def episode_messages(self, episode_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """The transcript of one episode, oldest first."""
        async with (
            self._serial,
            self._conn.execute(
                "SELECT id, seq, role, author, content, created_at FROM messages"
                " WHERE episode_id = ? ORDER BY seq LIMIT ?",
                (episode_id, limit),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def close_episode(
        self,
        episode_id: str,
        *,
        summary: str | None = None,
        signal_score: float | None = None,
        observations: Sequence[ObservationDraft] = (),
    ) -> list[str] | None:
        """Close an episode and record what was extracted from it, atomically.

        Returns the new observation ids, or None if something else closed the
        episode first. The guard is `state = 'open'`, so two sweeps racing over
        one episode produce one close and one None rather than two summaries
        and two sets of the same facts.

        `signal_score` is written whether or not the episode was gated on it.
        Only the scores of episodes that *were* gated would ever be surprising,
        and a threshold can only be tuned against the distribution it did not
        act on as well as the one it did.

        One transaction, because the two halves are only correct together. A
        close that commits without its observations discards what the episode
        was consolidated *for* and can never be retried — the episode is no
        longer open, so no sweep will look at it again.
        """
        async with self._serial:
            cur = await self._conn.execute(
                "UPDATE episodes SET state = 'closed', ended_at = ?,"
                " summary = COALESCE(?, summary), signal_score = COALESCE(?, signal_score)"
                " WHERE id = ? AND state = 'open'",
                (_now(), summary, signal_score, episode_id),
            )
            if cur.rowcount != 1:
                await self._conn.rollback()
                return None
            session_id = await self._session_of_episode(episode_id)
            ids = [
                await self._insert_observation(draft, session_id=session_id, episode_id=episode_id)
                for draft in observations
            ]
            await self._conn.commit()
            return ids

    async def _session_of_episode(self, episode_id: str) -> str | None:
        async with self._conn.execute(
            "SELECT session_id FROM episodes WHERE id = ?", (episode_id,)
        ) as cur:
            row = await cur.fetchone()
        return str(row["session_id"]) if row else None

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
            observation_id = await self._insert_observation(
                ObservationDraft(
                    subject=subject,
                    claim=claim,
                    kind=kind,
                    scope=scope,
                    confidence=confidence,
                    source_refs=tuple(source_refs),
                ),
                session_id=session_id,
                episode_id=episode_id,
            )
            await self._conn.commit()
            return observation_id

    async def _insert_observation(
        self, draft: ObservationDraft, *, session_id: str | None, episode_id: str | None
    ) -> str:
        """The one INSERT. Callers hold `_serial` and own the commit.

        `subject` is normalized here rather than trusted from the caller. It is
        the key `promote` groups by, so "Jane Doe" from a tool call and "Jane
        Doe's" from an extraction have to arrive as the same string, or the
        same person becomes two memories.
        """
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
                normalize_subject(draft.subject),
                draft.claim,
                draft.kind,
                draft.confidence,
                draft.scope,
                json_dumps(list(draft.source_refs)),
                _now(),
            ),
        )
        return observation_id

    async def pending_observations(self, limit: int = 100) -> list[dict[str, Any]]:
        """Candidate facts nobody has decided about yet, oldest first.

        Ordered by `(scope, subject)` before age so that `promote`'s grouping
        falls out of the read: the pair is what it reconciles in one call, and
        two visibility scopes must never meet inside one. `created_at` breaks
        the tie, so the limit still takes the oldest of whatever is waiting.
        """
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM observations WHERE state = 'pending'"
                " ORDER BY scope, subject, created_at LIMIT ?",
                (limit,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def episode_summaries(
        self, *, since: str, until: str, scope: str = "workspace"
    ) -> list[dict[str, Any]]:
        """Closed episodes from one window and one audience, oldest first.

        Scoped, and the caller says to what. The one reader is the nightly
        journal, which is a file in the repo: summarizing a DM into it would
        put a private conversation somewhere the whole workspace can read.
        """
        async with (
            self._serial,
            self._conn.execute(
                "SELECT e.id, e.summary, e.signal_score, e.ended_at, s.scope FROM episodes e"
                " JOIN sessions s ON s.id = e.session_id"
                " WHERE e.state != 'open' AND e.summary IS NOT NULL AND s.scope = ?"
                " AND e.ended_at >= ? AND e.ended_at < ? ORDER BY e.ended_at",
                (scope, since, until),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

        # -- recall telemetry ----------------------------------------------------

    async def record_memory_hits(self, memory_ids: Sequence[str]) -> None:
        """Note that these memories were recalled into a conversation."""
        if not memory_ids:
            return
        now = _now()
        async with self._serial:
            await self._conn.executemany(
                "INSERT INTO memory_hits (memory_id, hit_at) VALUES (?, ?)",
                [(memory_id, now) for memory_id in memory_ids],
            )
            await self._conn.commit()

    async def memory_hits_since(self, since: str) -> dict[str, int]:
        """How often each memory was recalled since `since`."""
        async with (
            self._serial,
            self._conn.execute(
                "SELECT memory_id, COUNT(*) AS hits FROM memory_hits WHERE hit_at >= ?"
                " GROUP BY memory_id",
                (since,),
            ) as cur,
        ):
            return {str(row["memory_id"]): int(row["hits"]) for row in await cur.fetchall()}

    async def purge_memory_hits(self, *, before: str) -> int:
        """Drop hits older than the window `reflect` reads. They have already
        been folded into a salience that lives in the repo."""
        async with self._serial:
            cur = await self._conn.execute("DELETE FROM memory_hits WHERE hit_at < ?", (before,))
            await self._conn.commit()
            return int(cur.rowcount)

    async def note_observation_attempt(self, ids: Sequence[str]) -> None:
        """Record that promotion was tried on these and did not land."""
        if not ids:
            return
        async with self._serial:
            await self._conn.execute(
                f"UPDATE observations SET attempts = attempts + 1 WHERE id IN ({_marks(ids)})",
                tuple(ids),
            )
            await self._conn.commit()

    async def resolve_observations(self, ids: Sequence[str], *, state: str, reason: str) -> int:
        """Move observations out of `pending`, recording why.

        The reason is not decoration. An observation that was discarded is a
        thing Kasa decided not to remember, and a person reading the table
        later has no other way to find out whether that was a judgement or a
        rejected patch plan.
        """
        if not ids:
            return 0
        async with self._serial:
            cur = await self._conn.execute(
                f"UPDATE observations SET state = ?, reason = ?, resolved_at = ?"
                f" WHERE id IN ({_marks(ids)}) AND state = 'pending'",
                (state, reason, _now(), *ids),
            )
            await self._conn.commit()
            return int(cur.rowcount)

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

    # -- slack identities ------------------------------------------------------

    async def upsert_slack_user(
        self,
        *,
        team_id: str,
        user_id: str,
        display_name: str,
        real_name: str = "",
        is_bot: bool = False,
        deleted: bool = False,
    ) -> None:
        """Record what `users.info` said about somebody, now.

        The link to a `people/` memory is not written here and not cleared
        here. A person who changed their display name is the same person, and
        an upsert that dropped `memory_id` would have the identity job write a
        second file for them every time they edited their profile — which is
        the one outcome #23 exists to prevent.
        """
        async with self._serial:
            await self._conn.execute(
                """
                INSERT INTO slack_users (
                    team_id, user_id, display_name, real_name, is_bot, deleted, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (team_id, user_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    real_name    = excluded.real_name,
                    is_bot       = excluded.is_bot,
                    deleted      = excluded.deleted,
                    fetched_at   = excluded.fetched_at
                """,
                (
                    team_id,
                    user_id,
                    display_name,
                    real_name,
                    int(is_bot),
                    int(deleted),
                    _now(),
                ),
            )
            await self._conn.commit()

    async def get_slack_user(self, team_id: str, user_id: str) -> dict[str, Any] | None:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM slack_users WHERE team_id = ? AND user_id = ?",
                (team_id, user_id),
            ) as cur,
        ):
            row = await cur.fetchone()
        return dict(row) if row else None

    async def slack_users_awaiting_memory(self, limit: int = 100) -> list[dict[str, Any]]:
        """Everyone the identity job still owes the corpus a write for.

        Two cases, one query: never linked at all, and linked under a name they
        no longer go by. Deleted accounts are included — somebody who has left
        is still who they were in every conversation already recorded.

        Bots are not, and not by filtering downstream: an app is nobody to
        remember, so a caller that skipped them would be handed the same rows
        every sweep for the life of the workspace.
        """
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM slack_users WHERE is_bot = 0"
                " AND (memory_id IS NULL OR memory_name IS NOT display_name)"
                " ORDER BY fetched_at LIMIT ?",
                (limit,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def link_slack_user(
        self, *, team_id: str, user_id: str, memory_id: str, memory_name: str
    ) -> None:
        """Point a uid at the `people/` memory that now records it."""
        async with self._serial:
            await self._conn.execute(
                "UPDATE slack_users SET memory_id = ?, memory_name = ?, linked_at = ?"
                " WHERE team_id = ? AND user_id = ?",
                (memory_id, memory_name, _now(), team_id, user_id),
            )
            await self._conn.commit()

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


def _marks(values: Sequence[Any]) -> str:
    """Placeholders for an `IN` clause. The values are still bound, never
    interpolated — this only sizes the list."""
    return ", ".join("?" * len(values))

"""SQLite access and the forward-only migration runner."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import aiosqlite
from pydantic import TypeAdapter
from ulid import ULID

from kasa.errors import KasaError, StoreError
from kasa.llm.cost import CallRecord
from kasa.llm.types import ContentBlock, Message, starts_turn
from kasa.memory.observation import ObservationDraft
from kasa.memory.subject import normalize_subject

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_BLOCKS = TypeAdapter(tuple[ContentBlock, ...])

Scrubber = Callable[[str], str]


def _scrub_message(message: Message, scrub: Scrubber) -> Message:
    content = _scrub_value([block.model_dump(mode="python") for block in message.content], scrub)
    return message.model_copy(update={"content": _BLOCKS.validate_python(content)})


def _scrub_value(value: Any, scrub: Scrubber) -> Any:
    if isinstance(value, dict):
        return {key: _scrub_value(item, scrub) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item, scrub) for item in value]
    return scrub(value) if isinstance(value, str) else value


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

    def __init__(
        self, conn: aiosqlite.Connection, path: str, *, scrub: Scrubber | None = None
    ) -> None:
        self._conn = conn
        self.path = path
        self._serial = _Serial()
        self._vectors_enabled = False
        self._scrub = scrub or (lambda text: text)

    async def enable_vectors(self) -> None:
        """Load sqlite-vec lazily; lexical-only installs need no extension."""
        try:
            import sqlite_vec
        except ImportError as exc:  # pragma: no cover - depends on installation extras
            raise StoreError("vector retrieval requires the 'embeddings' extra") from exc
        async with self._serial:
            if self._vectors_enabled:
                return
            await self._conn.enable_load_extension(True)
            try:
                await self._conn.load_extension(sqlite_vec.loadable_path())
            finally:
                await self._conn.enable_load_extension(False)
            self._vectors_enabled = True

    @classmethod
    async def open(cls, path: str | Path, *, scrub: Scrubber | None = None) -> Self:
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
            store = cls(conn, target, scrub=scrub)
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
        external_id: str | None = None,
    ) -> str:
        # The final common boundary before conversation content reaches disk.
        # Providers and tools can echo a credential, so every block is covered.
        message = _scrub_message(message, self._scrub)
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
                " (id, session_id, episode_id, seq, role, author, content, tokens, created_at,"
                "  external_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    external_id,
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
        """The most recent messages, oldest first, starting at a turn boundary.

        At least `limit` of them, and more where the limit fell inside a turn.
        A turn writes two rows per round of tool calls, so a long one runs past
        any limit, and a slice taken by count alone can begin with a
        `tool_result` whose `tool_use` was left behind. Both provider families
        reject that outright, and Anthropic additionally requires the first
        message to be a real user message — which is exactly what a turn head
        is (#204).

        Widened rather than trimmed. Cutting back to the next boundary would
        throw away the turn in flight, which is the one the model is in the
        middle of and the one whose tool results the answer depends on. The
        packer is what decides how much of the extra fits (§8.5): it can
        neither drop nor split the newest turn, so it shortens the turn's own
        older tool results instead.

        A revised message is served as it stands now, because `revise_message`
        rewrote the row: an edit replaces the words and a deletion replaces
        them with a tombstone. Doing it there rather than here is what stops a
        reader forgetting to — the packer, the extractor and this all read
        `content`, and one of them treating a deleted message as still said is
        the whole failure #25 is about.
        """
        async with self._serial:
            rows = await self._messages_before(session_id, before=None, limit=limit)
            # Every session opens with something somebody said, so this walks
            # back to the head of one turn at worst, never past the first row.
            while rows and not starts_turn(rows[0][1]):
                older = await self._messages_before(session_id, before=rows[0][0], limit=limit)
                if not older:
                    break
                # The newest head in the batch, not the oldest row in it: this
                # is reaching back for the turn already in hand, not for the
                # conversation before it.
                head = next((i for i in reversed(range(len(older))) if starts_turn(older[i][1])), 0)
                rows = older[head:] + rows
            return [message for _, message in rows]

    async def _messages_before(
        self, session_id: str, *, before: int | None, limit: int
    ) -> list[tuple[int, Message]]:
        """`(seq, message)` for the newest `limit` rows older than `before`, oldest first."""
        sql = "SELECT seq, role, content FROM messages WHERE session_id = ?"
        params: tuple[object, ...] = (session_id, limit)
        if before is not None:
            sql += " AND seq < ?"
            params = (session_id, before, limit)
        async with self._conn.execute(f"{sql} ORDER BY seq DESC LIMIT ?", params) as cur:
            rows = list(await cur.fetchall())
        return [
            (row["seq"], Message(role=row["role"], content=_BLOCKS.validate_json(row["content"])))
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
                " (created_at, role, provider, model, tag, session_id, input_tokens, output_tokens,"
                "  cache_read_tokens, cache_write_tokens, cost_usd, latency_ms, ok, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _now(),
                    record.role,
                    record.provider,
                    record.model,
                    record.tag,
                    record.session_id,
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
                f"SELECT substr(created_at, 1, 10) AS day, role,"
                f" CASE WHEN instr(COALESCE(tag, ''), '.') > 0"
                f" THEN substr(tag, 1, instr(tag, '.') - 1)"
                f" ELSE COALESCE(tag, 'untagged') END AS job_kind,"
                f" model, COUNT(*) AS calls,"
                f" SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,"
                f" SUM(cache_read_tokens) AS cache_read_tokens,"
                f" SUM(cost_usd) AS cost_usd"
                f" FROM llm_calls {clause}"
                f" GROUP BY day, role, job_kind, model"
                f" ORDER BY day DESC, role, job_kind, model",
                params,
            ) as cur:
                return [dict(row) for row in await cur.fetchall()]

    async def spend_since(self, day: str) -> float:
        """Priced spend since the start of a UTC date."""
        async with (
            self._serial,
            self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0.0) AS spent FROM llm_calls WHERE created_at >= ?",
                (day,),
            ) as cur,
        ):
            row = await cur.fetchone()
            return float(row["spent"]) if row else 0.0

    async def session_cost_summary(self, session_id: str) -> dict[str, Any]:
        """Token, spend, and prompt-cache health for one conversation."""
        async with (
            self._serial,
            self._conn.execute(
                "SELECT COUNT(*) AS calls,"
                " COALESCE(SUM(input_tokens), 0) AS input_tokens,"
                " COALESCE(SUM(output_tokens), 0) AS output_tokens,"
                " COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,"
                " COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,"
                " COALESCE(SUM(cost_usd), 0.0) AS cost_usd"
                " FROM llm_calls WHERE session_id = ?",
                (session_id,),
            ) as cur,
        ):
            result = await cur.fetchone()
            assert result is not None  # aggregate queries always return one row
            row = dict(result)
        eligible = row["cache_read_tokens"] + row["cache_write_tokens"]
        row["cache_hit_rate"] = row["cache_read_tokens"] / eligible if eligible else 0.0
        return row

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

    # -- tasks -----------------------------------------------------------------
    #
    # A standing task is the intent; the `jobs` rows it produces are the runs.
    # Nothing here schedules — the clock reads these rows and enqueues a job —
    # so this section is storage and counting, and every judgement about what a
    # task is allowed to be lives in `kasa/runner/tasks.py`.

    async def create_task(
        self,
        *,
        task_id: str,
        owner: str,
        surface: str,
        session_id: str,
        channel: str | None,
        reply_to: str | None,
        scope: str,
        prompt: str,
        cron: str,
        timezone: str | None,
        fire_once: bool,
    ) -> None:
        """Insert a task. The caller has already decided it is allowed."""
        async with self._serial:
            await self._conn.execute(
                "INSERT INTO tasks (id, owner, surface, session_id, channel, reply_to, scope,"
                " prompt, cron, timezone, fire_once, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    task_id,
                    owner,
                    surface,
                    session_id,
                    channel,
                    reply_to,
                    scope,
                    prompt,
                    cron,
                    timezone,
                    int(fire_once),
                    _now(),
                ),
            )
            await self._conn.commit()

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        async with self._serial:
            async with self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cur:
                row = await cur.fetchone()
            return dict(row) if row else None

    async def list_tasks(
        self,
        *,
        state: str | None = None,
        owner: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Tasks, oldest first, narrowed however the caller is allowed to see.

        `owner` and `session_id` are the narrowing the `schedule_*` tools use,
        and they are the reason this takes them rather than filtering in Python:
        a tool that read every row and then discarded the ones it should not
        show has already had them (§7.1).
        """
        narrowing = [
            (column, value)
            for column, value in (("state", state), ("owner", owner), ("session_id", session_id))
            if value is not None
        ]
        where = "".join(f" AND {column} = ?" for column, _ in narrowing)
        async with (
            self._serial,
            self._conn.execute(
                f"SELECT * FROM tasks WHERE 1 = 1{where} ORDER BY created_at, id",
                tuple(value for _, value in narrowing),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def count_owner_tasks(self, owner: str) -> int:
        """How many schedules this person has that will still fire.

        `done` is excluded: a fired one-shot is history, and counting it against
        somebody forever would mean a person who used the feature correctly runs
        out of it.
        """
        async with (
            self._serial,
            self._conn.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE owner = ? AND state IN ('active', 'paused')",
                (owner,),
            ) as cur,
        ):
            row = await cur.fetchone()
            return int(row["n"]) if row else 0

    async def set_task_state(self, task_id: str, *, state: str, error: str | None = None) -> bool:
        """Move a task between active, paused and done. False if it is gone.

        A task coming back to `active` has its failure count cleared: whoever
        resumed it is saying the reason it stopped has been dealt with, and
        resuming into one-failure-from-paused would be a trap.
        """
        async with self._serial:
            if state == "active":
                statement = (
                    "UPDATE tasks SET state = 'active', consecutive_failures = 0,"
                    " last_error = NULL WHERE id = ?"
                )
                params: tuple[Any, ...] = (task_id,)
            else:
                statement = (
                    "UPDATE tasks SET state = ?, last_error = COALESCE(?, last_error) WHERE id = ?"
                )
                params = (state, error, task_id)
            cur = await self._conn.execute(statement, params)
            await self._conn.commit()
            return bool(cur.rowcount)

    async def delete_task(self, task_id: str) -> bool:
        async with self._serial:
            cur = await self._conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            await self._conn.commit()
            return bool(cur.rowcount)

    async def record_task_run(self, task_id: str, *, job_id: str) -> None:
        """A run that reached the inbox. Clears whatever went wrong before it."""
        async with self._serial:
            await self._conn.execute(
                "UPDATE tasks SET last_run_at = ?, last_job_id = ?, last_error = NULL,"
                " consecutive_failures = 0 WHERE id = ?",
                (_now(), job_id, task_id),
            )
            await self._conn.commit()

    async def record_task_failure(self, task_id: str, *, error: str) -> int:
        """A run that did not. Returns the new consecutive-failure count."""
        async with self._serial:
            async with self._conn.execute(
                "UPDATE tasks SET last_run_at = ?, last_error = ?,"
                " consecutive_failures = consecutive_failures + 1"
                " WHERE id = ? RETURNING consecutive_failures",
                (_now(), error, task_id),
            ) as cur:
                row = await cur.fetchone()
            await self._conn.commit()
            return int(row["consecutive_failures"]) if row else 0

    # -- revisions -------------------------------------------------------------

    async def message_by_external_id(self, external_id: str) -> dict[str, Any] | None:
        """The stored message a surface's own id refers to, if Kasa kept one."""
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM messages WHERE external_id = ? ORDER BY seq LIMIT 1",
                (external_id,),
            ) as cur,
        ):
            row = await cur.fetchone()
        return dict(row) if row else None

    async def revise_message(self, message_id: str, *, content: Message, state: str) -> None:
        """Rewrite what a stored message says, and record that it was rewritten.

        The row keeps its place in the sequence and its timestamps. Deleting it
        would rewrite a conversation the assistant has already answered — the
        reply is still in the transcript, and a reply to nothing reads as the
        model having invented the question.
        """
        content = _scrub_message(content, self._scrub)
        async with self._serial:
            await self._conn.execute(
                "UPDATE messages SET content = ?, state = ?, revised_at = ? WHERE id = ?",
                (_BLOCKS.dump_json(content.content).decode(), state, _now(), message_id),
            )
            await self._conn.commit()

    async def observations_from(self, message_id: str) -> list[dict[str, Any]]:
        """Every candidate fact that cited this message as its source.

        `source_refs` holds message ids, which is what makes this answerable at
        all: the extractor resolves the line numbers a model cites back to the
        rows they came from, precisely so that provenance survives into here.
        """
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM observations o WHERE EXISTS"
                " (SELECT 1 FROM json_each(o.source_refs) WHERE value = ?)"
                " ORDER BY created_at",
                (message_id,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def weaken_observations(self, ids: Sequence[str], *, factor: float, reason: str) -> int:
        """Trust these less, without deciding they are false.

        A retracted message is evidence about a claim, not a verdict on it: the
        person may have deleted a typo, or thought better of saying it out
        loud, or been wrong. Scaling the confidence lets `promote` weigh it
        against everything else that was said instead of this one event
        settling the matter.
        """
        if not ids:
            return 0
        async with self._serial:
            cur = await self._conn.execute(
                f"UPDATE observations SET confidence = MAX(0.0, confidence * ?), reason = ?"
                f" WHERE id IN ({_marks(ids)}) AND state = 'pending'",
                (factor, reason, *ids),
            )
            await self._conn.commit()
            return cur.rowcount

    # -- answers and feedback --------------------------------------------------

    async def record_answer(
        self,
        *,
        source: str,
        external_id: str,
        memory_ids: Sequence[str],
        session_id: str | None = None,
        scope: str = "workspace",
    ) -> str:
        """Remember which memories produced one answer.

        Written even when nothing was recalled. A 👍 on an answer that used no
        memory is still a fact about the answer, and a row that exists with an
        empty list is how a later reaction can tell "nothing to boost" from
        "this reply is not one of ours".
        """
        answer_id = str(ULID())
        async with self._serial:
            cur = await self._conn.execute(
                "INSERT INTO answers"
                " (id, source, external_id, session_id, scope, memory_ids, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (source, external_id) DO NOTHING",
                (
                    answer_id,
                    source,
                    external_id,
                    session_id,
                    scope,
                    json_dumps(list(dict.fromkeys(memory_ids))),
                    _now(),
                ),
            )
            await self._conn.commit()
            if cur.rowcount:
                return answer_id
            async with self._conn.execute(
                "SELECT id FROM answers WHERE source = ? AND external_id = ?",
                (source, external_id),
            ) as existing:
                row = await existing.fetchone()
            return str(row["id"]) if row else answer_id

    async def answer_at(self, source: str, external_id: str) -> dict[str, Any] | None:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM answers WHERE source = ? AND external_id = ?",
                (source, external_id),
            ) as cur,
        ):
            row = await cur.fetchone()
        return dict(row) if row else None

    async def add_memory_feedback(
        self, *, memory_id: str, kind: str, answer_id: str, author: str = ""
    ) -> bool:
        """Record one person's verdict on one memory. False if already recorded."""
        async with self._serial:
            cur = await self._conn.execute(
                "INSERT INTO memory_feedback (memory_id, kind, answer_id, author, created_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT (memory_id, answer_id, author, kind) DO NOTHING",
                (memory_id, kind, answer_id, author, _now()),
            )
            await self._conn.commit()
            return cur.rowcount > 0

    async def drop_memory_feedback(self, *, answer_id: str, author: str, kind: str) -> int:
        """Take a verdict back, while it is still only a row.

        Somebody removing a reaction is retracting it, and until `reflect` has
        acted on it there is nothing to undo but this. One that has already
        been applied stays: the corpus has moved, and quietly un-applying it
        here would leave the row and the file disagreeing with nobody able to
        tell which was right.
        """
        async with self._serial:
            cur = await self._conn.execute(
                "DELETE FROM memory_feedback"
                " WHERE answer_id = ? AND author = ? AND kind = ? AND applied_at IS NULL",
                (answer_id, author, kind),
            )
            await self._conn.commit()
            return cur.rowcount

    async def endorsements_since(self, since: str) -> dict[str, int]:
        """How many people vouched for each memory since `since`.

        A window and a count, exactly like `memory_hits_since`, because it
        feeds the same recomputed number and has to leave it idempotent.
        """
        async with (
            self._serial,
            self._conn.execute(
                "SELECT memory_id, COUNT(*) AS votes FROM memory_feedback"
                " WHERE kind = 'up' AND created_at >= ? GROUP BY memory_id",
                (since,),
            ) as cur,
        ):
            return {str(row["memory_id"]): int(row["votes"]) for row in await cur.fetchall()}

    async def unapplied_feedback(self, kind: str, limit: int = 100) -> list[dict[str, Any]]:
        """Verdicts nothing has acted on yet, oldest first."""
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM memory_feedback WHERE kind = ? AND applied_at IS NULL"
                " ORDER BY created_at LIMIT ?",
                (kind, limit),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def mark_feedback_applied(self, ids: Sequence[int]) -> int:
        """Spend these verdicts. Called after the commit that acted on them."""
        if not ids:
            return 0
        async with self._serial:
            cur = await self._conn.execute(
                f"UPDATE memory_feedback SET applied_at = ?"
                f" WHERE id IN ({_marks(ids)}) AND applied_at IS NULL",
                (_now(), *ids),
            )
            await self._conn.commit()
            return cur.rowcount

    # -- reviews ---------------------------------------------------------------

    async def queue_review(
        self,
        *,
        kind: str,
        key: str,
        subject: str,
        detail: str = "",
        refs: Sequence[str] = (),
        scope: str = "workspace",
    ) -> str | None:
        """Ask for a person. Returns the id, or None if it was already asked.

        Idempotent on `(kind, key)`, because the paths that raise reviews are
        retried: Slack re-sends an event it did not hear an ack for, and three
        deliveries of one deletion is one thing to look at.
        """
        review_id = str(ULID())
        async with self._serial:
            cur = await self._conn.execute(
                "INSERT INTO reviews (id, kind, subject, detail, refs, scope, key, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (kind, key) DO NOTHING",
                (review_id, kind, subject, detail, json_dumps(list(refs)), scope, key, _now()),
            )
            await self._conn.commit()
            return review_id if cur.rowcount else None

    async def open_reviews(self, limit: int = 100) -> list[dict[str, Any]]:
        async with (
            self._serial,
            self._conn.execute(
                "SELECT * FROM reviews WHERE state = 'open' ORDER BY created_at LIMIT ?",
                (limit,),
            ) as cur,
        ):
            return [dict(row) for row in await cur.fetchall()]

    async def resolve_review(self, review_id: str) -> bool:
        async with self._serial:
            cur = await self._conn.execute(
                "UPDATE reviews SET state = 'done', resolved_at = ?"
                " WHERE id = ? AND state = 'open'",
                (_now(), review_id),
            )
            await self._conn.commit()
            return cur.rowcount > 0

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


def message_from_row(row: Mapping[str, Any]) -> Message:
    """One `messages` row as the thing the model would be shown.

    Here rather than at each caller because the content column's encoding is
    this module's business, and a second place that knew how to decode it is a
    second place to keep in step with the block types.
    """
    return Message(role=row["role"], content=_BLOCKS.validate_json(str(row["content"])))


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _marks(values: Sequence[Any]) -> str:
    """Placeholders for an `IN` clause. The values are still bound, never
    interpolated — this only sizes the list."""
    return ", ".join("?" * len(values))

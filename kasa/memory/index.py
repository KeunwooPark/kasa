"""Building the derived search index from the repository.

The invariant this module exists to preserve: **SQLite is disposable, the repo
is the truth.** `reindex --full` deletes every chunk and rebuilds from the files
on disk, and the result must be identical to what an incremental run would have
produced. Chunk ids are derived from the memory id and the ordinal for exactly
that reason — a rebuild that renumbered rows would be a different index wearing
the same name.

Incremental work is keyed on the git blob hash of each file. Content that has
not changed is neither re-chunked nor re-embedded, which is the expensive half.
"""

from __future__ import annotations

import hashlib
import logging
import re
import struct
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kasa.memory.chunk import Chunk, chunk_document
from kasa.memory.document import MemoryDoc, MemoryError_, Problem, read_memory_bytes
from kasa.memory.layout import MEMORY_DIR, is_memory_path
from kasa.memory.lease import INDEX_LEASE_NAME, INDEX_LOCK_SUFFIX, Lease
from kasa.store import Store

log = logging.getLogger(__name__)

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]
EMBED_BATCH_SIZE = 64
_TABLE = re.compile(r"^chunks_vec_[0-9a-f]{16}$")


@dataclass(slots=True)
class IndexResult:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    #: Files the parser refused, each with the reason. Reasons rather than bare
    #: paths because the caller prints these next to the manifest's problems,
    #: which have always carried one — so a file both halves refused was named
    #: twice, once uselessly (#77).
    problems: list[Problem] = field(default_factory=list)
    chunks: int = 0
    embedded: int = 0

    def summary(self) -> str:
        parts = [f"{len(self.indexed)} file(s) indexed", f"{self.chunks} chunk(s)"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} unchanged")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.problems:
            parts.append(f"{len(self.problems)} unreadable")
        if self.embedded:
            parts.append(f"{self.embedded} embedded")
        return ", ".join(parts)


@dataclass(slots=True)
class Freshness:
    """The gap between the repo and the index, split by what can close it.

    `changed` and `removed` are what a reindex would act on. `unreadable` is
    what it would refuse again — the same list every run, which is why it must
    not be reported as staleness.
    """

    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)

    @property
    def stale(self) -> bool:
        return bool(self.changed or self.removed)


def blob_sha(data: bytes) -> str:
    """Git's own hash for a blob.

    Computed here rather than shelled out to `git hash-object`: it is the same
    number, it costs no subprocess per file, and it works on a file that has not
    been committed yet.
    """
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


class MemoryIndex:
    """The chunk table and its FTS mirror, derived from a memory repo."""

    def __init__(
        self,
        store: Store,
        root: Path,
        *,
        embedder: Embedder | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._store = store
        self._root = root.expanduser()
        self._embedder = embedder
        self._embedding_model = embedding_model

    async def reindex(self, *, full: bool = False) -> IndexResult:
        """Bring the index in step with the repo, under the index lease.

        Serialized because `_replace` is delete-then-insert, and SQLite makes
        each statement atomic rather than the pair: two runs both saw a path as
        absent and both inserted, so the loser died on `UNIQUE constraint
        failed: chunks.id` (#95). Nothing was lost — the index is derived — but
        a command documented as safe at any time should not exit on a traceback.
        """
        lock = self._lock_path()
        if lock is None:
            return await self._reindex(full=full)
        async with Lease(self._store, lock, name=INDEX_LEASE_NAME):
            return await self._reindex(full=full)

    def _lock_path(self) -> Path | None:
        """Beside the database, not in the repo.

        The database is what is being protected — the repo is only read — and
        putting it in the repo meant fabricating a `.git` directory for a tree
        that had none, which git then reads as a broken repository.

        `None` for an in-memory store: each connection is its own database, so
        there is nothing to serialize and no file to put a lock beside.
        """
        if self._store.path == ":memory:":
            return None
        database = Path(self._store.path)
        return database.with_name(f".{database.name}{INDEX_LOCK_SUFFIX}")

    async def _reindex(self, *, full: bool = False) -> IndexResult:
        if full:
            await self._store.write("DELETE FROM chunks")
            await self._store.write("DELETE FROM index_state")

        result = IndexResult()
        state = await self._state()
        on_disk: set[str] = set()
        dirty_chunks: list[Chunk] = []

        for path in sorted((self._root / MEMORY_DIR).rglob("*.md")):
            relative = path.relative_to(self._root).as_posix()
            if not is_memory_path(relative):
                continue
            # Added before the read, so an entry that cannot be read is not
            # then treated as deleted — its rows would be dropped on every run
            # and `freshness` would report it as removed forever.
            on_disk.add(relative)

            try:
                raw = read_memory_bytes(path, source=relative)
                sha = blob_sha(raw)
                if state.get(relative) == sha:
                    result.skipped.append(relative)
                    continue
                doc = MemoryDoc.parse(raw.decode(), source=relative)
            except (MemoryError_, UnicodeDecodeError) as exc:
                # One file somebody broke by hand must not cost the whole index.
                # `UnicodeDecodeError` because a `.md` that is not text at all —
                # a stray binary, a bad `git add` — used to take the command
                # down before it reported any of the work it had already done;
                # `read_memory_bytes` covers the ways an entry can fail before
                # its contents are even reached.
                reason = exc.reason if isinstance(exc, MemoryError_) else str(exc)
                log.warning("index: %s: %s", relative, reason)
                result.problems.append(Problem(relative, reason))
                continue

            chunks = chunk_document(doc, relative)
            await self._replace(relative, chunks, sha)
            dirty_chunks.extend(chunks)
            result.indexed.append(relative)
            result.chunks += len(chunks)

        for gone in sorted(set(state) - on_disk):
            await self._forget(gone)
            result.removed.append(gone)

        if self._embedder is not None and self._embedding_model is not None:
            result.embedded = await self._update_vectors(dirty_chunks, full=full)

        log.info("reindex: %s", result.summary())
        return result

    async def stats(self) -> dict[str, int]:
        rows = await self._store.raw(
            "SELECT COUNT(*) AS chunks, COUNT(DISTINCT memory_id) AS memories FROM chunks"
        )
        files = await self._store.raw("SELECT COUNT(*) AS n FROM index_state")
        return {
            "chunks": int(rows[0]["chunks"]) if rows else 0,
            "memories": int(rows[0]["memories"]) if rows else 0,
            "files": int(files[0]["n"]) if files else 0,
        }

    async def freshness(self) -> Freshness:
        """What a reindex would do if it ran right now.

        Comparing hashes alone was not enough. A file the parser refuses is
        never written to `index_state`, so it differs from the index forever,
        and `doctor` reported a repo that "has moved on" and prescribed
        `kasa reindex` — a command that had already run and could not change it
        (#69). Staleness is "a reindex would fix this"; a file that cannot be
        parsed is a different fact, and it belongs in a different sentence.

        Files whose hash matches are not parsed. The set this has to open is
        the set that differs, which is normally empty.
        """
        state = await self._state()
        fresh = Freshness()
        on_disk: set[str] = set()

        for path in sorted((self._root / MEMORY_DIR).rglob("*.md")):
            relative = path.relative_to(self._root).as_posix()
            if not is_memory_path(relative):
                continue
            on_disk.add(relative)

            try:
                raw = read_memory_bytes(path, source=relative)
                if state.get(relative) == blob_sha(raw):
                    continue
                MemoryDoc.parse(raw.decode(), source=relative)
            except (MemoryError_, UnicodeDecodeError):
                fresh.unreadable.append(relative)
            else:
                fresh.changed.append(relative)

        fresh.removed.extend(sorted(set(state) - on_disk))
        return fresh

    async def is_stale(self) -> bool:
        """True when a reindex would change the index."""
        return (await self.freshness()).stale

    # -- internals -----------------------------------------------------------

    async def _state(self) -> dict[str, str]:
        rows = await self._store.raw("SELECT path, blob_sha FROM index_state")
        return {row["path"]: row["blob_sha"] for row in rows}

    async def _replace(self, path: str, chunks: list[Chunk], sha: str) -> None:
        # Delete-then-insert rather than upsert: a file that lost a section must
        # lose the chunks for it, and the FTS triggers only see what changes.
        await self._store.write("DELETE FROM chunks WHERE path = ?", (path,))
        await self._store.write_many(
            "INSERT INTO chunks"
            " (id, memory_id, path, ordinal, text, scope, salience, pinned, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    c.id,
                    c.memory_id,
                    c.path,
                    c.ordinal,
                    c.text,
                    c.scope,
                    c.salience,
                    int(c.pinned),
                    c.updated_at,
                )
                for c in chunks
            ],
        )
        await self._store.write(
            "INSERT INTO index_state (path, blob_sha, indexed_at) VALUES (?, ?, ?)"
            " ON CONFLICT(path) DO UPDATE SET"
            " blob_sha = excluded.blob_sha, indexed_at = excluded.indexed_at",
            (path, sha, datetime.now(UTC).isoformat(timespec="seconds")),
        )

    async def _forget(self, path: str) -> None:
        await self._store.write("DELETE FROM chunks WHERE path = ?", (path,))
        await self._store.write("DELETE FROM index_state WHERE path = ?", (path,))

    async def _update_vectors(self, dirty: list[Chunk], *, full: bool) -> int:
        await self._store.enable_vectors()
        current = await self._store.raw("SELECT * FROM vector_indexes WHERE active = 1")
        active = current[0] if current else None
        rebuild = full or active is None or active["model"] != self._embedding_model
        if rebuild:
            rows = await self._store.raw("SELECT id, text, scope FROM chunks ORDER BY id")
        else:
            assert active is not None
            dirty_ids = {chunk.id for chunk in dirty}
            rows = [{"id": chunk.id, "text": chunk.text, "scope": chunk.scope} for chunk in dirty]
            table = _safe_table(str(active["table_name"]))
            for chunk_id in sorted(dirty_ids):
                await self._store.write(f"DELETE FROM {table} WHERE chunk_id = ?", (chunk_id,))
            live = {str(row["id"]) for row in await self._store.raw("SELECT id FROM chunks")}
            vector_ids = {
                str(row["chunk_id"])
                for row in await self._store.raw(f"SELECT chunk_id FROM {table}")
            }
            for stale in sorted(vector_ids - live):
                await self._store.write(f"DELETE FROM {table} WHERE chunk_id = ?", (stale,))
        if not rows:
            return 0

        embedded: list[tuple[dict[str, object], list[float]]] = []
        assert self._embedder is not None
        for start in range(0, len(rows), EMBED_BATCH_SIZE):
            batch = rows[start : start + EMBED_BATCH_SIZE]
            vectors = await self._embedder([str(row["text"]) for row in batch])
            if len(vectors) != len(batch):
                raise ValueError(f"asked for {len(batch)} embeddings, got {len(vectors)}")
            embedded.extend(zip(batch, vectors, strict=True))
        dimensions = len(embedded[0][1])
        if not dimensions or any(len(vector) != dimensions for _, vector in embedded):
            raise ValueError("embedding provider returned inconsistent dimensions")

        if rebuild:
            version = hashlib.sha256(f"{self._embedding_model}\0{dimensions}".encode()).hexdigest()[
                :16
            ]
            table = f"chunks_vec_{version}"
            await self._store.write(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0("
                f"chunk_id TEXT PRIMARY KEY, scope TEXT, embedding float[{dimensions}])"
            )
            await self._store.write(f"DELETE FROM {table}")
        else:
            assert active is not None
            table = _safe_table(str(active["table_name"]))
            if dimensions != int(active["dimensions"]):
                raise ValueError("embedding dimensions changed without a model version change")

        await self._store.write_many(
            f"INSERT INTO {table} (chunk_id, scope, embedding) VALUES (?, ?, ?)",
            [(str(row["id"]), str(row["scope"]), _serialize(vector)) for row, vector in embedded],
        )
        if rebuild:
            await self._store.write("UPDATE vector_indexes SET active = 0 WHERE active = 1")
            await self._store.write(
                "INSERT INTO vector_indexes"
                " (version, model, dimensions, table_name, active, built_at)"
                " VALUES (?, ?, ?, ?, 1, ?)"
                " ON CONFLICT(version) DO UPDATE SET active = 1, built_at = excluded.built_at",
                (
                    table.removeprefix("chunks_vec_"),
                    self._embedding_model,
                    dimensions,
                    table,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
        return len(embedded)


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _safe_table(table: str) -> str:
    if not _TABLE.fullmatch(table):
        raise ValueError("invalid vector table name")
    return table

"""Building the derived search index from the repository.

The invariant this module exists to preserve: **SQLite is disposable, the repo
is the truth.** `reindex --full` deletes every chunk and rebuilds from the files
on disk, and the result must be identical to what an incremental run would have
produced. Chunk ids are derived from the memory id and the ordinal for exactly
that reason — a rebuild that renumbered rows would be a different index wearing
the same name.

Incremental work is keyed on the git blob hash of each file. Content that has
not changed is not re-chunked, and — once #31 lands — not re-embedded, which is
the expensive half.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from kasa.memory.chunk import Chunk, chunk_document
from kasa.memory.document import MemoryDoc, MemoryError_
from kasa.memory.layout import MEMORY_DIR, is_memory_path
from kasa.store import Store

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IndexResult:
    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    chunks: int = 0

    def summary(self) -> str:
        parts = [f"{len(self.indexed)} file(s) indexed", f"{self.chunks} chunk(s)"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} unchanged")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.problems:
            parts.append(f"{len(self.problems)} unreadable")
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

    def __init__(self, store: Store, root: Path) -> None:
        self._store = store
        self._root = root.expanduser()

    async def reindex(self, *, full: bool = False) -> IndexResult:
        """Bring the index in step with the repo. Returns what it did."""
        if full:
            await self._store.write("DELETE FROM chunks")
            await self._store.write("DELETE FROM index_state")

        result = IndexResult()
        state = await self._state()
        on_disk: set[str] = set()

        for path in sorted((self._root / MEMORY_DIR).rglob("*.md")):
            relative = path.relative_to(self._root).as_posix()
            if not is_memory_path(relative):
                continue
            on_disk.add(relative)

            raw = path.read_bytes()
            sha = blob_sha(raw)
            if state.get(relative) == sha:
                result.skipped.append(relative)
                continue

            try:
                doc = MemoryDoc.parse(raw.decode(), source=relative)
            except (MemoryError_, UnicodeDecodeError) as exc:
                # One file somebody broke by hand must not cost the whole index.
                # `UnicodeDecodeError` because a `.md` that is not text at all —
                # a stray binary, a bad `git add` — used to take the command
                # down before it reported any of the work it had already done.
                log.warning("index: %s: %s", relative, exc)
                result.problems.append(relative)
                continue

            chunks = chunk_document(doc, relative)
            await self._replace(relative, chunks, sha)
            result.indexed.append(relative)
            result.chunks += len(chunks)

        for gone in sorted(set(state) - on_disk):
            await self._forget(gone)
            result.removed.append(gone)

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

            raw = path.read_bytes()
            if state.get(relative) == blob_sha(raw):
                continue
            try:
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

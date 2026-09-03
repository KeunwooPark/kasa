"""`.kasa/manifest.json` — the id → path index, and link resolution through it.

Every durable reference in the corpus goes through here. That is the whole point
of the file: the reorganizer moves and merges memories every week, and links
that pointed at paths would break on every reorganization. Links point at ids,
ids are resolved here, and a file is then free to move.

Like everything else in SQLite-versus-repo terms, this is derived data. It is
committed because the repo has to be readable on its own, but `rebuild` walks
the tree and regenerates it, and that is the authority when the two disagree.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from kasa.memory.document import MemoryDoc, MemoryError_, Problem, is_memory_id
from kasa.memory.layout import MANIFEST_PATH, MEMORY_DIR, is_memory_path
from kasa.memory.schema import SCHEMA_VERSION

MANIFEST_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    title: str
    type: str
    visibility: str = "workspace"
    checksum: str
    last_touched: str
    #: Denormalized from the document so a link to a merged-away memory can be
    #: followed without opening every file in the repo.
    supersedes: list[str] = Field(default_factory=list)


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = MANIFEST_VERSION
    schema_version: int = SCHEMA_VERSION
    generated: str = Field(default_factory=_now)
    memories: dict[str, ManifestEntry] = Field(default_factory=dict)

    # -- disk ----------------------------------------------------------------

    @classmethod
    def load(cls, root: Path) -> Self:
        target = root.expanduser() / MANIFEST_PATH
        if not target.exists():
            return cls()
        try:
            return cls.model_validate_json(target.read_text())
        except Exception as exc:
            raise MemoryError_(f"{MANIFEST_PATH} is unreadable: {exc}") from exc

    def save(self, root: Path) -> Path:
        target = root.expanduser() / MANIFEST_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        # Sorted keys and a trailing newline: this file is committed on every
        # write, and an unstable key order would make every diff unreadable.
        payload = self.model_dump(mode="json")
        payload["memories"] = dict(sorted(payload["memories"].items()))
        target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
        return target

    @classmethod
    def rebuild(cls, root: Path) -> tuple[Self, list[Problem]]:
        """Walk the repo and derive the manifest from what is actually there."""
        root = root.expanduser()
        memories: dict[str, ManifestEntry] = {}
        problems: list[Problem] = []

        for path in sorted((root / MEMORY_DIR).rglob("*.md")):
            relative = path.relative_to(root).as_posix()
            if not is_memory_path(relative):
                continue
            try:
                doc = MemoryDoc.parse(path.read_bytes().decode(), source=relative)
            except (MemoryError_, UnicodeDecodeError) as exc:
                # Decoding explicitly, and counted as a problem rather than
                # raised: a `.md` that is not text is one broken file, and it
                # must not cost the manifest for the whole repo.
                #
                # `.reason`, not `str(exc)`: a `Problem` already carries the
                # path, and every caller prefixes it. `str(exc)` named the file
                # a second time inside the reason (#70).
                reason = exc.reason if isinstance(exc, MemoryError_) else str(exc)
                problems.append(Problem(relative, reason))
                continue
            if (existing := memories.get(doc.id)) is not None:
                # Two files claiming one id is a merge gone wrong. Neither is
                # silently dropped; the first by path wins and the clash is named.
                problems.append(
                    Problem(relative, f"duplicate id {doc.id}, already used by {existing.path}")
                )
                continue
            memories[doc.id] = _entry(relative, doc, path)

        return cls(memories=memories), problems

    def accounts_for(self, root: Path) -> bool:
        """Whether this manifest describes exactly the memory files on disk.

        A path-set comparison rather than a full rebuild, because it is what
        resolution actually depends on and it costs a directory walk instead of
        parsing the corpus. The denormalized fields on an entry can still drift
        — a title edited by hand, say — which is why every consumer re-reads the
        document rather than trusting the copy.
        """
        root = root.expanduser()
        on_disk = set()
        for path in (root / MEMORY_DIR).rglob("*.md"):
            relative = path.relative_to(root).as_posix()
            if is_memory_path(relative):
                on_disk.add(relative)
        return on_disk == {entry.path for entry in self.memories.values()}

    # -- resolution ----------------------------------------------------------

    def __contains__(self, memory_id: object) -> bool:
        return memory_id in self.memories

    def __len__(self) -> int:
        return len(self.memories)

    def entry(self, memory_id: str) -> ManifestEntry | None:
        return self.memories.get(memory_id)

    def path_of(self, memory_id: str) -> str | None:
        entry = self.memories.get(memory_id)
        return entry.path if entry else None

    def id_at(self, path: str) -> str | None:
        for memory_id, entry in self.memories.items():
            if entry.path == path:
                return memory_id
        return None

    def successor_of(self, memory_id: str) -> str | None:
        """The memory that replaced `memory_id`, if one did."""
        for candidate, entry in self.memories.items():
            if memory_id in entry.supersedes:
                return candidate
        return None

    def resolve(self, target: str) -> ManifestEntry | None:
        """Resolve a wikilink target: an id, or a path with or without `memory/`.

        Ids win, and a superseded id follows the chain to whatever replaced it —
        that is what keeps a link alive across a merge. Path targets are a
        convenience for people writing links by hand, so they fall back to
        matching on filename, which survives the file being moved.
        """
        target = target.strip()
        if not target:
            return None
        if is_memory_id(target):
            if (entry := self.memories.get(target)) is not None:
                return entry
            successor = self.successor_of(target)
            return self.memories.get(successor) if successor else None
        return self._by_path(target)

    def dangling(self, doc: MemoryDoc) -> list[str]:
        """Link targets in `doc` that resolve to nothing."""
        return [target for target in doc.links() if self.resolve(target) is None]

    # -- mutation ------------------------------------------------------------

    def record(self, relative_path: str, doc: MemoryDoc, *, checksum: str) -> ManifestEntry:
        entry = ManifestEntry(
            path=relative_path,
            title=doc.frontmatter.title,
            type=doc.frontmatter.type,
            visibility=doc.frontmatter.visibility,
            checksum=checksum,
            last_touched=_now(),
            supersedes=list(doc.frontmatter.supersedes),
        )
        self.memories[doc.id] = entry
        self.generated = entry.last_touched
        return entry

    def forget(self, memory_id: str) -> bool:
        removed = self.memories.pop(memory_id, None) is not None
        if removed:
            self.generated = _now()
        return removed

    def move(self, memory_id: str, new_path: str) -> bool:
        """Point an existing id at a new path, leaving inbound links resolving."""
        entry = self.memories.get(memory_id)
        if entry is None:
            return False
        self.memories[memory_id] = entry.model_copy(
            update={"path": new_path, "last_touched": _now()}
        )
        self.generated = _now()
        return True

    # -- internals -----------------------------------------------------------

    def _by_path(self, target: str) -> ManifestEntry | None:
        candidates = {target, f"{target}.md"}
        candidates |= {f"{MEMORY_DIR}/{c}" for c in tuple(candidates)}
        for entry in self.memories.values():
            if entry.path in candidates:
                return entry
        # Nothing at that exact path. Fall back to the filename, so a hand-written
        # `[[people/jane]]` keeps working after the reorganizer moves the file.
        stem = Path(target).name.removesuffix(".md")
        for entry in self.memories.values():
            if Path(entry.path).stem == stem:
                return entry
        return None


def checksum_of(data: bytes | str) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _entry(relative: str, doc: MemoryDoc, path: Path) -> ManifestEntry:
    return ManifestEntry(
        path=relative,
        title=doc.frontmatter.title,
        type=doc.frontmatter.type,
        visibility=doc.frontmatter.visibility,
        checksum=checksum_of(path.read_bytes()),
        last_touched=doc.frontmatter.updated.astimezone(UTC).isoformat(timespec="seconds"),
        supersedes=list(doc.frontmatter.supersedes),
    )

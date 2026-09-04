"""Repairing wikilinks through the manifest.

A memory that was merged away keeps its id: the successor records it in
`supersedes`, and `Manifest.resolve` follows the chain, so a link to the old id
still lands somewhere at read time. What it does not do is fix the file. Six
months of reorganizing leaves a corpus whose links all still work and none of
which say what they point at, which is a corpus nobody can read.

This is the repair, and it is deliberately narrow. A link is rewritten only
when the manifest can say what it *became* — the successor's own id. A link
that resolves to nothing at all is not repairable through the manifest, so it
is reported and left exactly as somebody wrote it. Guessing at it would mean
editing prose to remove a pointer that may be the only record that the thing
ever existed.
"""

from __future__ import annotations

from dataclasses import dataclass

from kasa.memory.document import MemoryDoc, is_memory_id, rewrite_links
from kasa.memory.layout import ARCHIVE_DIR
from kasa.memory.manifest import Manifest, ManifestEntry


@dataclass(frozen=True, slots=True)
class Repaired:
    """One memory whose links now say what they point at."""

    memory_id: str
    path: str
    body: str
    #: Old target → new target, for the log and for the commit message.
    rewrites: dict[str, str]


@dataclass(frozen=True, slots=True)
class Broken:
    """A link the manifest cannot account for. Reported, never rewritten."""

    path: str
    target: str


def repair(doc: MemoryDoc, path: str, manifest: Manifest) -> tuple[Repaired | None, list[Broken]]:
    """Follow every link in `doc` through the manifest.

    Returns the repair this memory needs, if any, and the links nothing can
    repair.
    """
    rewrites: dict[str, str] = {}
    broken: list[Broken] = []

    for target in doc.links():
        entry = manifest.resolve(target)
        if entry is None:
            broken.append(Broken(path=path, target=target))
            continue
        # Only an id-shaped link can be stale in the way this fixes. A
        # `[[people/jane]]` link names a place rather than a thing, and the
        # manifest resolving it is the whole of what it claims.
        if not is_memory_id(target) or target == doc.id:
            continue
        if (successor := _successor(target, entry, manifest)) is not None:
            rewrites[target] = successor

    if not rewrites:
        return None, broken
    return (
        Repaired(
            memory_id=doc.id,
            path=path,
            body=rewrite_links(doc.body, lambda t: rewrites.get(t)),
            rewrites=rewrites,
        ),
        broken,
    )


def _successor(target: str, entry: ManifestEntry, manifest: Manifest) -> str | None:
    """What `target` became, or None if it is still the live answer.

    Two shapes of staleness, and they are not the same.

    The target may be gone from the manifest entirely — `resolve` found it only
    by following a `supersedes` chain — in which case the link already points
    at nothing on its own and the id it resolved *to* is the repair.

    Or the target still exists and has been archived. `resolve` returns the
    archived file, so the link works and lands a reader in the soft-delete
    tier, next to a memory that stopped being the current answer. If something
    supersedes it, that something is where the reader wanted to go.
    """
    current = manifest.id_at(entry.path)
    if current is not None and current != target:
        return current
    if entry.path.startswith(f"{ARCHIVE_DIR}/"):
        successor = manifest.successor_of(target)
        return successor if successor and successor != target else None
    return None

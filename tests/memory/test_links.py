"""Repairing wikilinks through the manifest."""

from __future__ import annotations

from pathlib import Path

from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.links import repair
from kasa.memory.manifest import Manifest


def corpus(root: Path, *docs: tuple[MemoryDoc, str | None]) -> Manifest:
    bootstrap(root)
    for doc, path in docs:
        target = root / (path or doc.suggested_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.render())
    manifest, problems = Manifest.rebuild(root)
    assert problems == []
    return manifest


def fact(title: str, body: str, **fields: object) -> MemoryDoc:
    return MemoryDoc.new(type="fact", title=title, body=body, **fields)  # type: ignore[arg-type]


def test_a_link_into_the_archive_is_pointed_at_the_successor(tmp_path: Path) -> None:
    """The link works, and it lands a reader in the soft-delete tier next to a
    memory that stopped being the current answer."""
    old = fact("Old", "Superseded.")
    new = fact("New", "Current.", supersedes=[old.id])
    pointer = fact("Pointer", f"See [[{old.id}]].")
    manifest = corpus(tmp_path, (old, "memory/archive/old.md"), (new, None), (pointer, None))

    repaired, broken = repair(pointer, "memory/facts/pointer.md", manifest)

    assert repaired is not None
    assert repaired.rewrites == {old.id: new.id}
    assert f"[[{new.id}]]" in repaired.body
    assert broken == []


def test_a_link_to_an_id_the_manifest_lost_follows_the_chain(tmp_path: Path) -> None:
    """The file is gone entirely — hand-deleted, or collected — and `resolve`
    finds it only through `supersedes`."""
    gone = fact("Gone", "")
    new = fact("New", "Current.", supersedes=[gone.id])
    pointer = fact("Pointer", f"See [[{gone.id}]].")
    manifest = corpus(tmp_path, (new, None), (pointer, None))

    repaired, _ = repair(pointer, "memory/facts/pointer.md", manifest)

    assert repaired is not None and repaired.rewrites == {gone.id: new.id}


def test_a_live_link_is_left_exactly_as_it_was(tmp_path: Path) -> None:
    live = fact("Live", "Still current.")
    pointer = fact("Pointer", f"See [[{live.id}]] and [[facts/live]].")
    manifest = corpus(tmp_path, (live, None), (pointer, None))

    repaired, broken = repair(pointer, "memory/facts/pointer.md", manifest)

    assert repaired is None
    assert broken == []


def test_an_archived_memory_with_no_successor_is_not_rewritten(tmp_path: Path) -> None:
    """Archived is not the same as replaced. There is nowhere better to send
    the reader, and inventing one would be worse than the archive."""
    old = fact("Old", "Archived, but nothing took its place.")
    pointer = fact("Pointer", f"See [[{old.id}]].")
    manifest = corpus(tmp_path, (old, "memory/archive/old.md"), (pointer, None))

    repaired, broken = repair(pointer, "memory/facts/pointer.md", manifest)

    assert repaired is None and broken == []


def test_a_link_to_nothing_at_all_is_reported_and_left_alone(tmp_path: Path) -> None:
    """Not repairable through the manifest. The bracketed text may be the only
    record that the thing ever existed."""
    pointer = fact("Pointer", "See [[mem_01ZZZZZZZZZZZZZZZZZZZZZZZZ]].")
    manifest = corpus(tmp_path, (pointer, None))

    repaired, broken = repair(pointer, "memory/facts/pointer.md", manifest)

    assert repaired is None
    assert [b.target for b in broken] == ["mem_01ZZZZZZZZZZZZZZZZZZZZZZZZ"]


def test_a_memory_that_links_to_itself_is_left_alone(tmp_path: Path) -> None:
    """It happens when a split hands the parts each other's ids and one of
    them keeps the original's. Rewriting it would chase its own tail."""
    doc = fact("Self", "placeholder")
    doc = doc.model_copy(update={"body": f"\nSee [[{doc.id}]]."})
    manifest = corpus(tmp_path, (doc, None))

    repaired, broken = repair(doc, "memory/facts/self.md", manifest)

    assert repaired is None and broken == []

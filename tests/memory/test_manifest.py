from __future__ import annotations

from pathlib import Path

import pytest

from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc, MemoryError_, new_memory_id
from kasa.memory.layout import MANIFEST_PATH
from kasa.memory.manifest import Manifest, checksum_of


def write(root: Path, relative: str, doc: MemoryDoc) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc.render())
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    bootstrap(tmp_path)
    return tmp_path


@pytest.fixture
def jane() -> MemoryDoc:
    return MemoryDoc.new(type="person", title="Jane", body="Owns deploys. See [[projects/deploy]].")


@pytest.fixture
def deploy() -> MemoryDoc:
    return MemoryDoc.new(type="project", title="Deploy pipeline", body="Owned by [[people/jane]].")


# -- building ----------------------------------------------------------------


def test_rebuild_indexes_every_memory(repo: Path, jane: MemoryDoc, deploy: MemoryDoc) -> None:
    write(repo, "memory/people/jane.md", jane)
    write(repo, "memory/projects/deploy.md", deploy)

    manifest, problems = Manifest.rebuild(repo)

    assert not problems
    assert len(manifest) == 2
    assert manifest.path_of(jane.id) == "memory/people/jane.md"
    assert manifest.entry(jane.id) is not None
    assert manifest.entry(jane.id).title == "Jane"  # type: ignore[union-attr]


def test_rebuild_ignores_machinery(repo: Path, jane: MemoryDoc) -> None:
    """README.md and everything under .kasa/ are generated, not memories."""
    write(repo, "memory/people/jane.md", jane)
    manifest, problems = Manifest.rebuild(repo)

    assert not problems
    assert len(manifest) == 1
    assert all(".kasa" not in e.path for e in manifest.memories.values())


def test_a_malformed_file_is_reported_rather_than_crashing(repo: Path, jane: MemoryDoc) -> None:
    """One file a person broke by hand must not cost the whole index."""
    write(repo, "memory/people/jane.md", jane)
    (repo / "memory/facts/broken.md").write_text("this has no frontmatter\n")

    manifest, problems = Manifest.rebuild(repo)

    assert len(manifest) == 1, "the good file is still indexed"
    assert [p.path for p in problems] == ["memory/facts/broken.md"]
    assert "frontmatter" in problems[0].reason


def test_two_files_claiming_one_id_are_reported(repo: Path, jane: MemoryDoc) -> None:
    write(repo, "memory/people/jane.md", jane)
    write(repo, "memory/people/jane-copy.md", jane)

    manifest, problems = Manifest.rebuild(repo)

    assert len(manifest) == 1
    assert "duplicate id" in problems[0].reason


def test_checksums_track_content(repo: Path, jane: MemoryDoc) -> None:
    path = write(repo, "memory/people/jane.md", jane)
    before = Manifest.rebuild(repo)[0].entry(jane.id)

    path.write_text(path.read_text() + "\nA new sentence.\n")
    after = Manifest.rebuild(repo)[0].entry(jane.id)

    assert before is not None and after is not None
    assert before.checksum != after.checksum
    assert after.checksum == checksum_of(path.read_bytes())


# -- persistence -------------------------------------------------------------


def test_save_and_load_round_trip(repo: Path, jane: MemoryDoc) -> None:
    write(repo, "memory/people/jane.md", jane)
    manifest = Manifest.rebuild(repo)[0]
    manifest.save(repo)

    assert Manifest.load(repo) == manifest


def test_saved_manifests_have_a_stable_key_order(repo: Path) -> None:
    """It is committed on every write; an unstable order makes every diff unreadable."""
    for name in ("c", "a", "b"):
        write(repo, f"memory/facts/{name}.md", MemoryDoc.new(type="fact", title=name))
    Manifest.rebuild(repo)[0].save(repo)
    first = (repo / MANIFEST_PATH).read_text()

    Manifest.load(repo).save(repo)
    assert (repo / MANIFEST_PATH).read_text() == first

    ids = [line.split('"')[1] for line in first.splitlines() if line.strip().startswith('"mem_')]
    assert ids == sorted(ids)


def test_a_missing_manifest_loads_as_empty(tmp_path: Path) -> None:
    """SQLite is disposable and so is this: the repo is the truth."""
    assert len(Manifest.load(tmp_path)) == 0


def test_a_corrupt_manifest_says_so(repo: Path) -> None:
    (repo / MANIFEST_PATH).write_text("{not json")
    with pytest.raises(MemoryError_, match="unreadable"):
        Manifest.load(repo)


# -- resolution: the point of the whole file ---------------------------------


def test_links_resolve_by_id_and_by_path(repo: Path, jane: MemoryDoc, deploy: MemoryDoc) -> None:
    write(repo, "memory/people/jane.md", jane)
    write(repo, "memory/projects/deploy.md", deploy)
    manifest = Manifest.rebuild(repo)[0]

    assert manifest.resolve(jane.id) is not None
    assert manifest.resolve("people/jane") is not None
    assert manifest.resolve("memory/people/jane.md") is not None
    assert manifest.resolve("people/nobody") is None
    assert manifest.resolve("") is None


def test_moving_a_file_leaves_every_inbound_link_resolving(
    repo: Path, jane: MemoryDoc, deploy: MemoryDoc
) -> None:
    """The acceptance criterion for #12, and the reason ids exist at all."""
    write(repo, "memory/people/jane.md", jane)
    write(repo, "memory/projects/deploy.md", deploy)
    manifest = Manifest.rebuild(repo)[0]

    linked_by_id = MemoryDoc.new(type="fact", title="pointer", body=f"See [[{jane.id}]].")
    write(repo, "memory/facts/pointer.md", linked_by_id)
    manifest = Manifest.rebuild(repo)[0]
    assert manifest.dangling(linked_by_id) == []

    # The weekly reorganizer moves the file and updates the manifest.
    (repo / "memory/people/jane.md").rename(repo / "memory/archive/jane.md")
    manifest.move(jane.id, "memory/archive/jane.md")

    assert manifest.dangling(linked_by_id) == [], "an id link survives the move"
    assert manifest.path_of(jane.id) == "memory/archive/jane.md"
    # The hand-written path link survives too, via the filename fallback.
    assert manifest.resolve("people/jane") is not None


def test_a_link_to_a_merged_away_memory_follows_the_supersedes_chain(repo: Path) -> None:
    old_id = new_memory_id()
    successor = MemoryDoc.new(type="topic", title="Merged", supersedes=[old_id])
    write(repo, "memory/topics/merged.md", successor)
    manifest = Manifest.rebuild(repo)[0]

    resolved = manifest.resolve(old_id)
    assert resolved is not None and resolved.path == "memory/topics/merged.md"
    assert manifest.successor_of(old_id) == successor.id


def test_an_unknown_id_resolves_to_nothing(repo: Path) -> None:
    assert Manifest.rebuild(repo)[0].resolve(new_memory_id()) is None


def test_dangling_links_are_listed(repo: Path, jane: MemoryDoc) -> None:
    write(repo, "memory/people/jane.md", jane)
    manifest = Manifest.rebuild(repo)[0]

    doc = MemoryDoc.new(type="fact", title="p", body="[[people/jane]] and [[people/ghost]]")
    assert manifest.dangling(doc) == ["people/ghost"]


# -- mutation ----------------------------------------------------------------


def test_record_then_forget(repo: Path, jane: MemoryDoc) -> None:
    manifest = Manifest()
    manifest.record("memory/people/jane.md", jane, checksum=checksum_of(jane.render()))

    assert jane.id in manifest
    assert manifest.id_at("memory/people/jane.md") == jane.id
    assert manifest.forget(jane.id) is True
    assert manifest.forget(jane.id) is False


def test_moving_an_unknown_id_is_a_no_op() -> None:
    assert Manifest().move(new_memory_id(), "memory/facts/x.md") is False


def test_accounts_for_compares_the_manifest_against_the_files_on_disk(tmp_path: Path) -> None:
    """The cheap staleness check id resolution leans on."""
    bootstrap(tmp_path)
    doc = MemoryDoc.new(type="fact", title="A thing", body="It is so.")
    target = tmp_path / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())

    assert not Manifest().accounts_for(tmp_path), "a file it has never seen"

    rebuilt, _ = Manifest.rebuild(tmp_path)
    assert rebuilt.accounts_for(tmp_path)

    target.unlink()
    assert not rebuilt.accounts_for(tmp_path), "and a file that went away"


def test_a_markdown_file_that_is_not_text_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    """One broken file must not cost the manifest for the whole repo."""
    bootstrap(tmp_path)
    doc = MemoryDoc.new(type="fact", title="Readable", body="Fine.")
    target = tmp_path / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    (tmp_path / "memory" / "facts" / "binary.md").write_bytes(b"\xff\xfe\x00not utf-8")

    manifest, problems = Manifest.rebuild(tmp_path)

    assert manifest.resolve(doc.id) is not None
    assert [p.path for p in problems] == ["memory/facts/binary.md"]


def test_a_problem_names_the_file_once(tmp_path: Path) -> None:
    """#70. The reason is bare; the path is the other field. Every caller
    prefixes it, so a reason that repeated it rendered the path twice."""
    bootstrap(tmp_path)
    (tmp_path / "memory" / "facts" / "broken.md").write_text("no frontmatter here at all")

    _, problems = Manifest.rebuild(tmp_path)

    assert [p.path for p in problems] == ["memory/facts/broken.md"]
    assert problems[0].reason == "no YAML frontmatter (a file must open with `---`)"
    rendered = f"{problems[0].path}: {problems[0].reason}"
    assert rendered.count("memory/facts/broken.md") == 1

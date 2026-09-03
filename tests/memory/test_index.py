from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from kasa.memory.bootstrap import bootstrap
from kasa.memory.chunk import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, chunk_document, split_body
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import run_git
from kasa.memory.index import MemoryIndex, blob_sha
from kasa.memory.lease import INDEX_LEASE_NAME, Lease, LeaseError
from kasa.memory.manifest import Manifest
from kasa.store import Store


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    bootstrap(tmp_path)
    return tmp_path


def add(root: Path, doc: MemoryDoc, path: str | None = None) -> str:
    relative = path or doc.suggested_path()
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    return relative


async def chunks_of(store: Store) -> list[dict[str, object]]:
    return await store.raw(
        "SELECT id, memory_id, path, ordinal, text, scope, salience FROM chunks ORDER BY id"
    )


async def search(store: Store, query: str) -> list[str]:
    rows = await store.raw(
        "SELECT c.memory_id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid"
        " WHERE chunks_fts MATCH ? ORDER BY rank",
        (query,),
    )
    return list(dict.fromkeys(str(row["memory_id"]) for row in rows))


# -- chunking ----------------------------------------------------------------


def test_the_first_chunk_carries_the_title_and_tags() -> None:
    """A memory whose body never repeats its own title still has to be findable."""
    doc = MemoryDoc.new(
        type="person", title="Deploy pipeline ownership", tags=["infra"], body="She does it."
    )
    chunks = chunk_document(doc, "memory/people/jane.md")

    assert chunks[0].ordinal == 0
    assert "Deploy pipeline ownership" in chunks[0].text
    assert "infra" in chunks[0].text


def test_chunks_carry_scope_and_salience_for_filtering() -> None:
    doc = MemoryDoc.new(type="fact", title="T", body="b", visibility="private:U01", salience=0.9)
    for chunk in chunk_document(doc, "memory/facts/t.md"):
        assert chunk.scope == "private:U01"
        assert chunk.salience == 0.9


def test_chunk_ids_are_deterministic() -> None:
    doc = MemoryDoc.new(type="fact", title="T", body="b")
    assert [c.id for c in chunk_document(doc, "p")] == [f"{doc.id}:0", f"{doc.id}:1"]


def test_prose_is_cut_on_headings() -> None:
    section = "a" * 300
    body = f"intro {section}\n\n## First\n\n{section}\n\n## Second\n\n{section}\n"
    pieces = split_body(body)

    assert len(pieces) == 3
    assert pieces[1].startswith("## First")
    assert pieces[2].startswith("## Second")


def test_a_short_memory_stays_one_chunk() -> None:
    """Splitting a two-sentence memory into three pieces makes recall worse, not better."""
    body = "intro\n\n## First\n\nalpha\n\n## Second\n\nbeta\n"
    assert len(split_body(body)) == 1


def test_a_long_section_falls_back_to_paragraphs() -> None:
    paragraph = "x" * 500
    body = "\n\n".join([paragraph] * 6)
    pieces = split_body(body)

    assert len(pieces) > 1
    assert all(len(piece) <= MAX_CHUNK_CHARS for piece in pieces)


def test_a_single_paragraph_with_no_boundary_is_still_split() -> None:
    pieces = split_body("y" * (MAX_CHUNK_CHARS * 3))
    assert all(len(piece) <= MAX_CHUNK_CHARS for piece in pieces)
    assert "".join(pieces) == "y" * (MAX_CHUNK_CHARS * 3)


def test_a_short_tail_is_folded_back_rather_than_left_as_a_fragment() -> None:
    """A two-word chunk retrieved on its own reads as a non-sequitur."""
    body = "## A\n\n" + "x" * 400 + "\n\n## B\n\nshort\n"
    pieces = split_body(body)

    assert all(len(piece) >= MIN_CHUNK_CHARS or len(pieces) == 1 for piece in pieces)
    assert "short" in pieces[-1]


def test_an_empty_body_yields_only_the_header_chunk() -> None:
    doc = MemoryDoc.new(type="fact", title="T")
    assert len(chunk_document(doc, "p")) == 1


# -- indexing ----------------------------------------------------------------


async def test_reindex_indexes_every_memory(repo: Path, store: Store) -> None:
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    add(repo, MemoryDoc.new(type="project", title="Deploy pipeline", body="Runs nightly."))

    result = await MemoryIndex(store, repo).reindex()

    assert len(result.indexed) == 2
    assert result.chunks == 4  # header + body, twice
    assert len(await chunks_of(store)) == 4


async def test_the_index_is_searchable(repo: Path, store: Store) -> None:
    jane = MemoryDoc.new(type="person", title="Jane", body="Owns the deploy pipeline.")
    bob = MemoryDoc.new(type="person", title="Bob", body="Runs the incident rota.")
    add(repo, jane)
    add(repo, bob)
    await MemoryIndex(store, repo).reindex()

    assert await search(store, "deploy") == [jane.id]
    assert await search(store, "incident") == [bob.id]


async def test_machinery_is_not_indexed(repo: Path, store: Store) -> None:
    """README.md and .kasa/ are generated; indexing them pollutes every search."""
    await MemoryIndex(store, repo).reindex()
    assert await chunks_of(store) == []


async def test_a_broken_file_is_reported_and_skipped(repo: Path, store: Store) -> None:
    good = MemoryDoc.new(type="person", title="Jane", body="Fine.")
    add(repo, good)
    (repo / "memory/facts/broken.md").write_text("no frontmatter\n")

    result = await MemoryIndex(store, repo).reindex()

    assert result.indexed == ["memory/people/jane.md"]
    assert [p.path for p in result.problems] == ["memory/facts/broken.md"]


# -- acceptance: incremental work --------------------------------------------


async def test_reindex_after_a_one_file_change_touches_only_that_file(
    repo: Path, store: Store
) -> None:
    """The acceptance criterion. blob_sha is what keeps this cheap."""
    jane = add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    add(repo, MemoryDoc.new(type="project", title="Deploys", body="Nightly."))
    index = MemoryIndex(store, repo)
    await index.reindex()

    changed = MemoryDoc.parse((repo / jane).read_text())
    (repo / jane).write_text(
        MemoryDoc(frontmatter=changed.frontmatter, body="\nOwns deploys and the rota.\n").render()
    )
    result = await index.reindex()

    assert result.indexed == [jane]
    assert result.skipped == ["memory/projects/deploys.md"]


async def test_an_unchanged_repo_reindexes_nothing(repo: Path, store: Store) -> None:
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    index = MemoryIndex(store, repo)
    await index.reindex()

    result = await index.reindex()
    assert result.indexed == []
    assert len(result.skipped) == 1


async def test_a_deleted_file_loses_its_chunks(repo: Path, store: Store) -> None:
    jane = add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    index = MemoryIndex(store, repo)
    await index.reindex()

    (repo / jane).unlink()
    result = await index.reindex()

    assert result.removed == [jane]
    assert await chunks_of(store) == []
    assert await search(store, "deploys") == [], "and it stops being findable"


async def test_a_shrinking_file_loses_the_chunks_it_no_longer_has(repo: Path, store: Store) -> None:
    long_body = "\n\n".join(f"## Section {i}\n\ncontent {i} " + "x" * 300 for i in range(5))
    doc = MemoryDoc.new(type="topic", title="Big", body=long_body)
    path = add(repo, doc)
    index = MemoryIndex(store, repo)
    await index.reindex()
    assert len(await chunks_of(store)) > 3

    (repo / path).write_text(MemoryDoc(frontmatter=doc.frontmatter, body="\nsmall now\n").render())
    await index.reindex()

    assert len(await chunks_of(store)) == 2, "header plus one body chunk, nothing stale"
    assert await search(store, "Section") == []


# -- acceptance: a full rebuild reproduces the same index --------------------


async def test_a_full_rebuild_reproduces_an_identical_index(repo: Path, store: Store) -> None:
    """`rm index.db && kasa reindex --full` must land in the same place.

    This is the invariant the whole design rests on: SQLite is disposable, and
    the repo is the truth.
    """
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    add(
        repo, MemoryDoc.new(type="topic", title="Rota", body="## How\n\nWeekly.\n\n## Why\n\nFair.")
    )
    index = MemoryIndex(store, repo)
    await index.reindex()
    before = await chunks_of(store)

    await index.reindex(full=True)

    assert await chunks_of(store) == before
    assert await search(store, "deploys")


async def test_a_full_rebuild_drops_rows_for_files_that_are_gone(repo: Path, store: Store) -> None:
    path = add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    index = MemoryIndex(store, repo)
    await index.reindex()

    (repo / path).unlink()
    await index.reindex(full=True)

    assert await chunks_of(store) == []


async def test_the_fts_mirror_stays_in_step_with_the_table(repo: Path, store: Store) -> None:
    """The triggers exist so no code path can insert a chunk and forget to index it."""
    doc = MemoryDoc.new(type="person", title="Jane", body="Owns the deploy pipeline.")
    path = add(repo, doc)
    index = MemoryIndex(store, repo)
    await index.reindex()
    assert await search(store, "pipeline")

    (repo / path).write_text(
        MemoryDoc(frontmatter=doc.frontmatter, body="\nRuns the incident rota.\n").render()
    )
    await index.reindex()

    assert await search(store, "pipeline") == [], "the old text is gone from FTS too"
    assert await search(store, "rota") == [doc.id]


# -- freshness ---------------------------------------------------------------


async def test_staleness_is_detected(repo: Path, store: Store) -> None:
    index = MemoryIndex(store, repo)
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))

    assert await index.is_stale() is True
    await index.reindex()
    assert await index.is_stale() is False

    add(repo, MemoryDoc.new(type="person", title="Bob", body="Runs the rota."))
    assert await index.is_stale() is True


async def test_a_file_the_indexer_refuses_is_not_staleness(repo: Path, store: Store) -> None:
    """#69. A broken file is never written to `index_state`, so hash comparison
    alone said the repo had moved on — forever, and `kasa reindex` could not
    change it."""
    index = MemoryIndex(store, repo)
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    await index.reindex()

    (repo / "memory" / "facts" / "broken.md").write_text("no frontmatter here at all")

    for _ in range(3):
        assert [p.path for p in (await index.reindex()).problems] == ["memory/facts/broken.md"]
        fresh = await index.freshness()
        assert fresh.stale is False, "reindex cannot fix it, so it is not staleness"
        assert fresh.unreadable == ["memory/facts/broken.md"]
        assert fresh.changed == []


async def test_a_file_that_was_indexed_and_then_broken_is_still_not_staleness(
    repo: Path, store: Store
) -> None:
    """The index keeps the old chunks, and a reindex would not replace them —
    which is exactly why the answer has to name the file rather than the index."""
    index = MemoryIndex(store, repo)
    path = add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    await index.reindex()

    (repo / path).write_text("mangled by hand")
    fresh = await index.freshness()

    assert fresh.stale is False
    assert fresh.unreadable == [path]


async def test_a_deleted_file_is_staleness(repo: Path, store: Store) -> None:
    index = MemoryIndex(store, repo)
    path = add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    await index.reindex()

    (repo / path).unlink()
    fresh = await index.freshness()

    assert fresh.stale is True
    assert fresh.removed == [path]


async def test_a_markdown_file_that_is_not_text_does_not_take_the_run_down(
    repo: Path, store: Store
) -> None:
    """A stray binary or a bad `git add`. It used to raise `UnicodeDecodeError`
    out of `reindex`, losing the report of everything already indexed."""
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    (repo / "memory" / "facts" / "binary.md").write_bytes(b"\xff\xfe\x00not utf-8")

    result = await MemoryIndex(store, repo).reindex()

    assert result.indexed, "the readable file was still indexed"
    assert [p.path for p in result.problems] == ["memory/facts/binary.md"]
    assert (await MemoryIndex(store, repo).freshness()).stale is False


async def test_stats_report_what_is_indexed(repo: Path, store: Store) -> None:
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    await MemoryIndex(store, repo).reindex()

    stats = await MemoryIndex(store, repo).stats()
    assert stats == {"chunks": 2, "memories": 1, "files": 1}


# -- the blob hash -----------------------------------------------------------


def test_blob_sha_matches_git(tmp_path: Path) -> None:
    """It is git's number, so it can be compared against a tree listing directly."""
    content = b"some memory content\n"
    path = tmp_path / "f.md"
    path.write_bytes(content)

    expected = run_git("hash-object", str(path)).stdout.strip()
    assert blob_sha(content) == expected


# -- one rebuild at a time (#95) ----------------------------------------------


async def test_a_second_reindex_is_refused_rather_than_interleaved(
    repo: Path, store: Store
) -> None:
    """#95. `_replace` is delete-then-insert, and SQLite makes each statement
    atomic rather than the pair — so two runs both saw a path as absent and
    both inserted, and the loser died on `UNIQUE constraint failed: chunks.id`.
    """
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    index = MemoryIndex(store, repo)

    held = await Lease(store, index._lock_path(), name=INDEX_LEASE_NAME).acquire()
    try:
        with pytest.raises(LeaseError) as caught:
            await index.reindex()
    finally:
        await held.release()

    assert "index rebuild lease" in str(caught.value)
    assert "memory write lease" not in str(caught.value), "it names the lease it is about"


async def test_the_lease_is_released_and_the_next_run_works(repo: Path, store: Store) -> None:
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    index = MemoryIndex(store, repo)

    assert (await index.reindex()).indexed
    assert await store.get_lease(INDEX_LEASE_NAME) is None
    assert (await index.reindex(full=True)).indexed, "and again"


async def test_the_lock_is_not_put_in_the_repo(repo: Path, store: Store) -> None:
    """It guards the database, which is what a second run corrupts. Putting it
    in the repo meant fabricating a `.git` directory for a tree that had none,
    which git then reads as a broken repository."""
    await MemoryIndex(store, repo).reindex()

    assert not (repo / ".git").exists()
    assert MemoryIndex(store, repo)._lock_path() == Path(store.path).parent / (
        f".{Path(store.path).name}.index.lock"
    )


# -- entries under memory/ that are not readable files (#97) ------------------


def a_directory(root: Path) -> None:
    (root / "memory" / "facts" / "adir.md").mkdir(parents=True)


def a_symlink_loop(root: Path) -> None:
    target = root / "memory" / "facts" / "loop.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(target)


def a_dangling_symlink(root: Path) -> None:
    facts = root / "memory" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    (facts / "dangling.md").symlink_to(facts / "nope.md")


def a_fifo(root: Path) -> None:
    facts = root / "memory" / "facts"
    facts.mkdir(parents=True, exist_ok=True)
    os.mkfifo(facts / "pipe.md")


def an_unreadable_file(root: Path) -> None:
    path = root / "memory" / "facts" / "locked.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MemoryDoc.new(type="fact", title="Locked", body="b").render())
    path.chmod(0o000)


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (a_directory, "adir.md"),
        (a_symlink_loop, "loop.md"),
        (a_dangling_symlink, "dangling.md"),
        # A fifo is the one that could not be fixed by catching anything: the
        # open blocks forever with no writer. Hence the shape check.
        (a_fifo, "pipe.md"),
        (an_unreadable_file, "locked.md"),
    ],
)
async def test_an_entry_that_is_not_a_readable_file_costs_only_itself(
    repo: Path, store: Store, make: Callable[[Path], None], name: str
) -> None:
    """#97. `rglob("*.md")` matches on the name, and every one of these raised
    an `OSError` nobody caught — so one entry created by accident cost the whole
    index and the whole manifest."""
    add(repo, MemoryDoc.new(type="person", title="Jane", body="Owns deploys."))
    make(repo)

    result = await MemoryIndex(store, repo).reindex()

    assert result.indexed, "the readable memory was still indexed"
    assert [p.path for p in result.problems] == [f"memory/facts/{name}"]
    assert await search(store, "deploys")

    manifest, problems = Manifest.rebuild(repo)
    assert len(manifest) == 1
    assert [p.path for p in problems] == [f"memory/facts/{name}"]


async def test_a_symlink_to_a_real_file_is_still_indexed(repo: Path, store: Store) -> None:
    """The guard is "not a readable file", not "not a plain path"."""
    doc = MemoryDoc.new(type="fact", title="Elsewhere", body="Kept outside the tree.")
    outside = repo.parent / "outside.md"
    outside.write_text(doc.render())
    (repo / "memory" / "facts").mkdir(parents=True, exist_ok=True)
    (repo / "memory" / "facts" / "link.md").symlink_to(outside)

    result = await MemoryIndex(store, repo).reindex()

    assert result.problems == []
    assert "memory/facts/link.md" in result.indexed

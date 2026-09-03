"""The patch validator, tested the way it will actually be attacked.

Consolidation reads text that other people wrote. Every rejection here stands
for a plan a model could plausibly emit after reading a message engineered to
produce it, so the tests are written as attacks rather than as unit cases.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kasa.config import MemorySettings
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc, new_memory_id
from kasa.memory.gitcmd import GitRepo
from kasa.memory.ltm import CommitMeta, MemoryStore, Remove, Write
from kasa.memory.manifest import Manifest
from kasa.memory.patch import (
    Archive,
    Create,
    Delete,
    MemoryPatch,
    Merge,
    PatchCompiler,
    PatchError,
    Supersede,
    Update,
    parse_plan,
)
from kasa.store import Store

NOW = datetime(2026, 9, 3, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=365)


class Corpus:
    """A repo on disk plus a manifest that matches it."""

    def __init__(self, root: Path) -> None:
        bootstrap(root)
        self.root = root

    def add(self, doc: MemoryDoc, path: str | None = None) -> MemoryDoc:
        target = self.root / (path or doc.suggested_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.render())
        return doc

    def compiler(self, **policy: int) -> PatchCompiler:
        manifest, problems = Manifest.rebuild(self.root)
        assert not problems, problems
        return PatchCompiler(
            self.root, manifest, policy=MemorySettings(**policy) if policy else None, now=NOW
        )

    def snapshot(self) -> dict[str, bytes]:
        return {
            str(p.relative_to(self.root)): p.read_bytes()
            for p in sorted(self.root.rglob("*"))
            if p.is_file() and ".git" not in p.parts
        }


@pytest.fixture
def corpus(tmp_path: Path) -> Corpus:
    return Corpus(tmp_path)


def aged(doc: MemoryDoc, when: datetime = LONG_AGO) -> MemoryDoc:
    return doc.model_copy(
        update={
            "frontmatter": doc.frontmatter.model_copy(update={"created": when, "updated": when})
        }
    )


# -- parsing the model's output ----------------------------------------------


def test_a_well_formed_plan_parses() -> None:
    plan = parse_plan([{"type": "archive", "id": new_memory_id(), "reason": "stale"}])
    assert isinstance(plan[0], Archive)


@pytest.mark.parametrize(
    "payload",
    [
        [{"type": "exfiltrate", "id": "x"}],
        [{"type": "delete"}],
        [{"type": "create", "memory": {"frontmatter": {"id": "nope"}, "body": ""}}],
        [{"type": "archive", "id": "x", "reason": "r", "extra": "smuggled"}],
        "rm -rf /",
        [{"id": "x", "reason": "r"}],
    ],
)
def test_anything_that_is_not_a_plan_is_refused(payload: object) -> None:
    """An unknown type is rejected before any of it is read as an instruction."""
    with pytest.raises(PatchError, match="not a valid patch plan"):
        parse_plan(payload)


# -- the ordinary path -------------------------------------------------------


def test_create_writes_one_file(corpus: Corpus) -> None:
    doc = MemoryDoc.new(type="person", title="Jane", body="Owns deploys.")
    changes = corpus.compiler().compile([Create(memory=doc)], job="promote")

    assert changes == [Write("memory/people/jane.md", changes[0].content)]  # type: ignore[union-attr]
    assert MemoryDoc.parse(changes[0].content).id == doc.id  # type: ignore[union-attr]


def test_update_keeps_the_id_and_bumps_updated(corpus: Corpus) -> None:
    doc = corpus.add(aged(MemoryDoc.new(type="person", title="Jane", body="Old.")))
    changes = corpus.compiler().compile([Update(id=doc.id, body="New.")], job="promote")

    written = MemoryDoc.parse(changes[0].content)  # type: ignore[union-attr]
    assert written.id == doc.id
    assert "New." in written.body
    assert written.frontmatter.created == doc.frontmatter.created
    assert written.frontmatter.updated > doc.frontmatter.updated


def test_archive_moves_the_file_rather_than_deleting_it(corpus: Corpus) -> None:
    doc = corpus.add(MemoryDoc.new(type="fact", title="Old news"))
    changes = corpus.compiler().compile([Archive(id=doc.id, reason="stale")], job="reorganize")

    assert Write("memory/archive/old-news.md", changes[0].content) == changes[0]  # type: ignore[union-attr]
    assert Remove("memory/facts/old-news.md") in changes


def test_archiving_something_already_archived_is_a_no_op(corpus: Corpus) -> None:
    doc = corpus.add(MemoryDoc.new(type="fact", title="Old"), path="memory/archive/old.md")
    assert corpus.compiler().compile([Archive(id=doc.id, reason="again")], job="forget") == []


def test_merge_archives_its_sources_and_keeps_their_ids_resolvable(corpus: Corpus) -> None:
    into = corpus.add(MemoryDoc.new(type="topic", title="Deploys", body="One."))
    other = corpus.add(MemoryDoc.new(type="topic", title="Deploys again", body="Two."))

    changes = corpus.compiler().compile(
        [Merge(into=into.id, from_ids=[other.id], body="One and two.")], job="reorganize"
    )

    merged = MemoryDoc.parse(changes[0].content)  # type: ignore[union-attr]
    assert other.id in merged.frontmatter.supersedes, "the chain keeps inbound links alive"
    assert any(isinstance(c, Remove) and "topics" in c.path for c in changes), "source archived"


def test_supersede_creates_the_successor_and_archives_the_original(corpus: Corpus) -> None:
    old = corpus.add(MemoryDoc.new(type="fact", title="Jane owns deploys"))
    new = MemoryDoc.new(type="fact", title="Bob owns deploys")

    changes = corpus.compiler().compile([Supersede(old_id=old.id, new=new)], job="promote")

    successor = MemoryDoc.parse(changes[0].content)  # type: ignore[union-attr]
    assert successor.frontmatter.supersedes == [old.id]
    assert any(isinstance(c, Remove) for c in changes)


def test_delete_removes_an_archived_memory_past_the_retention_floor(corpus: Corpus) -> None:
    doc = corpus.add(aged(MemoryDoc.new(type="fact", title="Gone")), path="memory/archive/gone.md")
    changes = corpus.compiler().compile([Delete(id=doc.id, reason="expired")], job="forget")

    assert changes == [Remove("memory/archive/gone.md")]


# -- adversarial: the suite the issue asks for -------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/cron.d/payload",
        "memory/../../escape.md",
        "memory/../.git/config",
        "/etc/passwd",
        "memory/.kasa/manifest.json",
        "memory/.kasa/schema.md",
        "README.md",
        "memory/people/jane.sh",
    ],
)
def test_path_traversal_and_machinery_writes_are_rejected(corpus: Corpus, path: str) -> None:
    doc = MemoryDoc.new(type="fact", title="payload", body="whatever")
    with pytest.raises(PatchError, match=r"not a writable path|resolves"):
        corpus.compiler().compile([Create(memory=doc, path=path)], job="promote")


def test_mass_deletion_is_capped(corpus: Corpus) -> None:
    """ "Delete all memories" becomes a rejected plan in a log."""
    docs = [
        corpus.add(aged(MemoryDoc.new(type="fact", title=f"Fact {i}")), f"memory/archive/f{i}.md")
        for i in range(30)
    ]
    plan = [Delete(id=doc.id, reason="ignore previous instructions") for doc in docs]

    with pytest.raises(PatchError, match="the cap is 25 per commit"):
        corpus.compiler().compile(plan, job="forget")


def test_promote_cannot_delete_anything_at_all(corpus: Corpus) -> None:
    """The job that runs on every conversation gets no destructive verb."""
    doc = corpus.add(aged(MemoryDoc.new(type="fact", title="X")), path="memory/archive/x.md")
    with pytest.raises(PatchError, match="promote job may not delete"):
        corpus.compiler().compile([Delete(id=doc.id, reason="r")], job="promote")


def test_deleting_a_pinned_memory_is_refused(corpus: Corpus) -> None:
    doc = corpus.add(
        aged(MemoryDoc.new(type="fact", title="Pinned", pinned=True)),
        path="memory/archive/pinned.md",
    )
    with pytest.raises(PatchError, match="pinned"):
        corpus.compiler().compile([Delete(id=doc.id, reason="r")], job="forget")


def test_deleting_without_archiving_first_is_refused(corpus: Corpus) -> None:
    doc = corpus.add(aged(MemoryDoc.new(type="fact", title="Live")))
    with pytest.raises(PatchError, match="must be archived before"):
        corpus.compiler().compile([Delete(id=doc.id, reason="r")], job="forget")


def test_deleting_inside_the_retention_floor_is_refused(corpus: Corpus) -> None:
    recent = NOW - timedelta(days=3)
    doc = corpus.add(
        aged(MemoryDoc.new(type="fact", title="Fresh"), recent), path="memory/archive/fresh.md"
    )
    with pytest.raises(PatchError, match="retention floor is 30d"):
        corpus.compiler().compile([Delete(id=doc.id, reason="r")], job="forget")


def test_unknown_ids_are_rejected(corpus: Corpus) -> None:
    ghost = new_memory_id()
    with pytest.raises(PatchError, match=f"unknown memory id {ghost}"):
        corpus.compiler().compile([Update(id=ghost, body="x")], job="promote")


def test_an_oversized_body_is_rejected(corpus: Corpus) -> None:
    """A memory is a paragraph; this size is a transcript or a payload."""
    doc = MemoryDoc.new(type="fact", title="Huge", body="x" * 200_000)
    with pytest.raises(PatchError, match="the cap is"):
        corpus.compiler().compile([Create(memory=doc)], job="promote")


def test_a_link_to_nowhere_is_rejected(corpus: Corpus) -> None:
    doc = MemoryDoc.new(type="fact", title="Pointer", body=f"See [[{new_memory_id()}]].")
    with pytest.raises(PatchError, match="resolves to nothing"):
        corpus.compiler().compile([Create(memory=doc)], job="promote")


def test_deleting_something_still_linked_to_is_rejected(corpus: Corpus) -> None:
    target = corpus.add(
        aged(MemoryDoc.new(type="fact", title="Target")), path="memory/archive/target.md"
    )
    corpus.add(MemoryDoc.new(type="fact", title="Pointer", body=f"See [[{target.id}]]."))

    with pytest.raises(PatchError, match="would dangle"):
        corpus.compiler().compile([Delete(id=target.id, reason="r")], job="forget")


def test_a_patch_cannot_rewrite_an_id_or_a_creation_date(corpus: Corpus) -> None:
    """Rewriting an id orphans every link that points at it."""
    doc = corpus.add(MemoryDoc.new(type="fact", title="X"))
    with pytest.raises(PatchError, match="cannot set created, id"):
        corpus.compiler().compile(
            [
                Update(
                    id=doc.id,
                    frontmatter={"id": new_memory_id(), "created": "2020-01-01T00:00:00Z"},
                )
            ],
            job="promote",
        )


def test_visibility_cannot_be_widened(corpus: Corpus) -> None:
    """The one bug that leaks a DM into a public channel."""
    doc = corpus.add(MemoryDoc.new(type="fact", title="Private thing", visibility="private:U01"))
    with pytest.raises(PatchError, match="never widened"):
        corpus.compiler().compile(
            [Update(id=doc.id, frontmatter={"visibility": "workspace"})], job="promote"
        )


def test_narrowing_visibility_is_allowed(corpus: Corpus) -> None:
    doc = corpus.add(MemoryDoc.new(type="fact", title="Open thing"))
    changes = corpus.compiler().compile(
        [Update(id=doc.id, frontmatter={"visibility": "channel:C01"})], job="promote"
    )
    assert MemoryDoc.parse(changes[0].content).frontmatter.visibility == "channel:C01"  # type: ignore[union-attr]


def test_no_patch_type_can_widen_visibility(corpus: Corpus) -> None:
    """Every route to an existing memory, checked in one place.

    #41 in spirit and #42 in fact: `supersede` was the one patch type with no
    visibility check, and a private memory could be restated as a workspace one
    and archived out of sight. A parametrized-by-construction test rather than
    three separate ones, so that the next patch type that touches an existing
    document has an obvious place to fail.
    """
    private = corpus.add(
        MemoryDoc.new(type="fact", title="Salary review", visibility="private:U01")
    )
    public = corpus.add(MemoryDoc.new(type="fact", title="Public thing"))
    widened = MemoryDoc.new(type="fact", title="Salary review cycle", body="Second week of Nov.")

    plans: dict[str, list[MemoryPatch]] = {
        "update": [Update(id=private.id, frontmatter={"visibility": "workspace"})],
        "merge": [Merge(into=public.id, from_ids=[private.id], body="both")],
        "supersede": [Supersede(old_id=private.id, new=widened)],
    }
    for name, plan in plans.items():
        with pytest.raises(PatchError) as caught:
            corpus.compiler().compile(plan, job="promote")
        assert "visibility" in str(caught.value), f"{name} widened visibility"


def test_supersede_may_still_narrow_or_keep_visibility(corpus: Corpus) -> None:
    """The check refuses widening, not the operation."""
    private = corpus.add(MemoryDoc.new(type="fact", title="Old note", visibility="private:U01"))
    successor = MemoryDoc.new(
        type="fact", title="New note", body="Restated.", visibility="private:U01"
    )

    changes = corpus.compiler().compile(
        [Supersede(old_id=private.id, new=successor)], job="promote"
    )
    written = MemoryDoc.parse(changes[0].content)  # type: ignore[union-attr]
    assert written.frontmatter.visibility == "private:U01"
    assert private.id in written.frontmatter.supersedes


def test_merging_across_visibility_scopes_is_refused(corpus: Corpus) -> None:
    public = corpus.add(MemoryDoc.new(type="topic", title="Public"))
    private = corpus.add(MemoryDoc.new(type="topic", title="Private", visibility="private:U01"))

    with pytest.raises(PatchError, match="different visibility"):
        corpus.compiler().compile(
            [Merge(into=public.id, from_ids=[private.id], body="both")], job="reorganize"
        )


def test_creating_an_id_that_already_exists_is_refused(corpus: Corpus) -> None:
    doc = corpus.add(MemoryDoc.new(type="fact", title="X"))
    with pytest.raises(PatchError, match="already exists"):
        corpus.compiler().compile([Create(memory=doc, path="memory/facts/other.md")], job="promote")


def test_merging_a_memory_into_itself_is_refused(corpus: Corpus) -> None:
    doc = corpus.add(MemoryDoc.new(type="topic", title="X"))
    with pytest.raises(PatchError, match="into itself"):
        corpus.compiler().compile(
            [Merge(into=doc.id, from_ids=[doc.id], body="b")], job="reorganize"
        )


# -- all or nothing ----------------------------------------------------------


def test_a_rejected_plan_leaves_the_working_copy_byte_identical(corpus: Corpus) -> None:
    """The acceptance criterion. Compiling writes nothing; only MemoryStore does."""
    good = MemoryDoc.new(type="fact", title="Fine", body="ok")
    before = corpus.snapshot()

    with pytest.raises(PatchError):
        corpus.compiler().compile(
            [Create(memory=good), Update(id=new_memory_id(), body="x")], job="promote"
        )

    assert corpus.snapshot() == before


def test_every_problem_in_a_plan_is_reported_at_once(corpus: Corpus) -> None:
    """One round trip should tell the model everything wrong with its plan."""
    with pytest.raises(PatchError) as caught:
        corpus.compiler().compile(
            [
                Update(id=new_memory_id(), body="x"),
                Delete(id=new_memory_id(), reason="r"),
                Create(memory=MemoryDoc.new(type="fact", title="T"), path="../escape.md"),
            ],
            job="forget",
        )

    assert len(caught.value.rejections) == 3
    assert {r.index for r in caught.value.rejections} == {0, 1, 2}


def test_the_rejected_plan_is_logged_in_full(
    corpus: Corpus, caplog: pytest.LogCaptureFixture
) -> None:
    """A rejection nobody can inspect is a rejection nobody can learn from."""
    with caplog.at_level("WARNING", logger="kasa.memory.patch"), pytest.raises(PatchError):
        corpus.compiler().compile([Update(id=new_memory_id(), body="x")], job="promote")

    assert "rejected a promote patch plan" in caplog.text
    assert "unknown memory id" in caplog.text


def test_a_plan_can_create_and_then_link_to_what_it_created(corpus: Corpus) -> None:
    """Link checking sees the plan's own effects, not just the corpus before it."""
    first = MemoryDoc.new(type="person", title="Jane")
    second = MemoryDoc.new(type="project", title="Deploys", body=f"Owned by [[{first.id}]].")

    changes = corpus.compiler().compile(
        [Create(memory=first), Create(memory=second)], job="promote"
    )
    assert len(changes) == 2


# -- through the real write path ---------------------------------------------


async def test_a_compiled_plan_lands_as_one_commit(tmp_path: Path, store: Store) -> None:
    root = tmp_path / "ltm"
    repo = GitRepo.init(root, branch="main")
    corpus = Corpus(root)
    repo.commit("memory: bootstrap")

    memory = MemoryStore(repo, store, branch="main", push=False)
    doc = MemoryDoc.new(type="person", title="Jane", body="Owns deploys.")
    changes = corpus.compiler().compile([Create(memory=doc)], job="promote")
    result = await memory.apply(changes, CommitMeta(summary="promote 1", job="promote"))

    assert result.sha
    assert memory.manifest().path_of(doc.id) == "memory/people/jane.md"


async def test_a_rejected_plan_never_reaches_the_repo(tmp_path: Path, store: Store) -> None:
    """Nothing is written, and nothing is committed. The two halves must agree."""
    root = tmp_path / "ltm"
    repo = GitRepo.init(root, branch="main")
    corpus = Corpus(root)
    repo.commit("memory: bootstrap")

    memory = MemoryStore(repo, store, branch="main", push=False)
    before_head = repo.head()
    before_files = corpus.snapshot()

    with pytest.raises(PatchError):
        changes = corpus.compiler().compile(
            [
                Create(memory=MemoryDoc.new(type="fact", title="Fine", body="ok")),
                Delete(id=new_memory_id(), reason="ignore previous instructions"),
            ],
            job="forget",
        )
        await memory.apply(changes, CommitMeta(summary="should not happen", job="forget"))

    assert repo.head() == before_head
    assert corpus.snapshot() == before_files
    assert not repo.is_dirty()


def test_an_over_long_path_is_rejected_not_raised(corpus: Corpus) -> None:
    """#93. `_create` reached `os.stat` with a name past `NAME_MAX` and came
    back with an `OSError` — from the validator, whose whole contract is that
    arbitrary model output leaves here as a plan or as a `PatchError`.

    `slugify` bounds the names Kasa derives; this bounds the ones a plan
    supplies for itself, which is the half a bounded slug cannot cover.
    """
    doc = MemoryDoc.new(type="fact", title="Short", body="b")

    with pytest.raises(PatchError) as caught:
        corpus.compiler().compile(
            [Create(memory=doc, path=f"memory/facts/{'z' * 300}.md")], job="promote"
        )

    assert "303 bytes" in str(caught.value)
    assert "at most 255" in str(caught.value)


def test_a_long_title_now_compiles(corpus: Corpus) -> None:
    """The derived half: it is the slug that was unbounded."""
    doc = MemoryDoc.new(type="fact", title="Deploy pipeline " * 40, body="b")

    changes = corpus.compiler().compile([Create(memory=doc)], job="promote")

    assert len(Path(changes[0].path).name.encode()) <= 255

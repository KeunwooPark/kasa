"""`identity`: one person, one file, however often their name changes.

The property under test throughout is the absence of a second file. Almost
every way this job can go wrong ends with two memories for one person — a
rename that creates instead of updating, a re-run that does not recognize its
own work, a crash between the commit and the marking — so most of these assert
on how many `people/` files exist as much as on what is in them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kasa.adapters.slack.identity import user_ref
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.ltm import MemoryStore
from kasa.memory.manifest import Manifest
from kasa.runner.identity import CLOSE, OPEN, Registrar
from kasa.store import Store

TEAM = "T0TEAM"
JANE = "U0JANE"
RAJ = "U0RAJ"


# -- fixtures and helpers ----------------------------------------------------


def clone_at(tmp_path: Path) -> Path:
    repo = tmp_path / "ltm"
    GitRepo.init(repo, branch="main")
    bootstrap(repo)
    Manifest.rebuild(repo)[0].save(repo)
    GitRepo.at(repo).commit("memory: bootstrap")
    return repo


def registrar(clone: Path, store: Store, **kwargs: Any) -> Registrar:
    memory = MemoryStore(GitRepo.at(clone), store, branch="main", push=False)
    return Registrar(store, memory, **kwargs)


async def seen(store: Store, user_id: str, display_name: str, **fields: Any) -> None:
    await store.upsert_slack_user(
        team_id=TEAM, user_id=user_id, display_name=display_name, **fields
    )


def people(clone: Path) -> list[str]:
    directory = clone / "memory/people"
    if not directory.exists():
        return []
    return sorted(p.name for p in directory.glob("*.md") if p.name != "README.md")


def read(clone: Path, name: str) -> MemoryDoc:
    return MemoryDoc.parse((clone / "memory/people" / name).read_text())


def retitled(doc: MemoryDoc, title: str) -> MemoryDoc:
    return doc.model_copy(
        update={"frontmatter": doc.frontmatter.model_copy(update={"title": title})}
    )


def write(clone: Path, doc: MemoryDoc, *, path: str | None = None) -> str:
    target = clone / (path or doc.suggested_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit(f"memory: seed {doc.id}")
    return str(target.relative_to(clone))


# -- the first sighting ------------------------------------------------------


async def test_a_new_person_becomes_one_memory(tmp_path: Path, store: Store) -> None:
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane", real_name="Jane Doe")

    result = await registrar(clone, store).run()

    assert people(clone) == ["jane.md"]
    doc = read(clone, "jane.md")
    assert doc.frontmatter.title == "jane"
    assert doc.frontmatter.source_refs == [user_ref(TEAM, JANE)]
    assert doc.frontmatter.visibility == "workspace"
    assert "Jane Doe" in doc.body
    assert result.created == 1


async def test_the_row_is_pointed_at_what_was_written(tmp_path: Path, store: Store) -> None:
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")

    await registrar(clone, store).run()

    row = await store.get_slack_user(TEAM, JANE)
    assert row is not None
    assert row["memory_id"] == read(clone, "jane.md").id
    assert row["memory_name"] == "jane"


async def test_a_second_run_finds_nothing_to_do(tmp_path: Path, store: Store) -> None:
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    await registrar(clone, store).run()

    again = await registrar(clone, store).run()

    assert again.linked == []
    assert people(clone) == ["jane.md"]


async def test_a_bot_is_never_written(tmp_path: Path, store: Store) -> None:
    clone = clone_at(tmp_path)
    await seen(store, "B0DEPLOY", "deploybot", is_bot=True)

    result = await registrar(clone, store).run()

    assert people(clone) == []
    assert result.linked == []


async def test_a_deactivated_account_is_still_somebody(tmp_path: Path, store: Store) -> None:
    """They are in every conversation already recorded, and a reader of those
    should be able to find out why they went quiet."""
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane", deleted=True)

    await registrar(clone, store).run()

    assert "deactivated" in read(clone, "jane.md").body


# -- names that change -------------------------------------------------------


async def test_a_rename_updates_the_memory_instead_of_forking_it(
    tmp_path: Path, store: Store
) -> None:
    """The whole issue, in one test."""
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    await registrar(clone, store).run()
    first = read(clone, "jane.md").id

    await seen(store, JANE, "jane-doe")
    result = await registrar(clone, store).run()

    assert people(clone) == ["jane.md"], "the path never moves; the manifest maps id to path"
    doc = read(clone, "jane.md")
    assert doc.id == first
    assert doc.frontmatter.title == "jane-doe"
    assert doc.frontmatter.source_refs == [user_ref(TEAM, JANE)], "the ref is not duplicated"
    assert "@jane-doe" in doc.body and "@jane`" not in doc.body
    assert result.updated == 1


async def test_a_title_somebody_else_chose_is_left_alone(tmp_path: Path, store: Store) -> None:
    """`promote` and people both write these files. A display name changing is
    not a reason to overwrite a title neither of them derived from it."""
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    await registrar(clone, store).run()

    doc = read(clone, "jane.md")
    write(clone, retitled(doc, "Jane Doe (infra)"), path="memory/people/jane.md")
    await seen(store, JANE, "jane-doe")
    await registrar(clone, store).run()

    assert read(clone, "jane.md").frontmatter.title == "Jane Doe (infra)"
    assert "@jane-doe" in read(clone, "jane.md").body, "the block still tracks the handle"


async def test_prose_around_the_block_survives_a_rename(tmp_path: Path, store: Store) -> None:
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    await registrar(clone, store).run()

    doc = read(clone, "jane.md")
    write(
        clone,
        doc.model_copy(update={"body": f"{doc.body}\nRuns the deploy rota.\n"}),
        path="memory/people/jane.md",
    )
    await seen(store, JANE, "jane-doe")
    await registrar(clone, store).run()

    body = read(clone, "jane.md").body
    assert "Runs the deploy rota." in body
    assert body.count(OPEN) == 1 and body.count(CLOSE) == 1


# -- linking to what is already there ----------------------------------------


async def test_a_person_the_corpus_already_knows_is_adopted(tmp_path: Path, store: Store) -> None:
    """`promote` writes about people who have not spoken yet. Keying only on
    the uid would put a second Jane beside the one already there."""
    clone = clone_at(tmp_path)
    existing = MemoryDoc.new(type="person", title="jane", body="Owns the deploy pipeline.")
    write(clone, existing)
    await seen(store, JANE, "jane")

    await registrar(clone, store).run()

    assert people(clone) == ["jane.md"]
    doc = read(clone, "jane.md")
    assert doc.id == existing.id
    assert doc.frontmatter.source_refs == [user_ref(TEAM, JANE)]
    assert "Owns the deploy pipeline." in doc.body


async def test_a_namesake_already_claimed_by_someone_else_is_not_adopted(
    tmp_path: Path, store: Store
) -> None:
    """Two people with one name is a duplicate file. Two people in one file is
    a memory that says false things about both of them."""
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    await registrar(clone, store).run()

    await seen(store, RAJ, "jane")
    await registrar(clone, store).run()

    assert people(clone) == ["jane-2.md", "jane.md"]
    assert read(clone, "jane.md").frontmatter.source_refs == [user_ref(TEAM, JANE)]
    assert read(clone, "jane-2.md").frontmatter.source_refs == [user_ref(TEAM, RAJ)]


async def test_a_commit_that_never_landed_is_not_reported_as_mapped(
    tmp_path: Path, store: Store, monkeypatch: Any
) -> None:
    """A row pointed at a file that was never written is the one state this
    job cannot recover from: the sweep would never return that uid again."""
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    job = registrar(clone, store)

    async def refuse(*_: object, **__: object) -> Any:
        from kasa.memory.ltm import ApplyResult

        return ApplyResult()

    monkeypatch.setattr(job._memory, "apply", refuse)
    result = await job.run()

    assert result.linked == []
    row = await store.get_slack_user(TEAM, JANE)
    assert row is not None and row["memory_id"] is None, "still due a write"


async def test_a_run_that_died_before_marking_links_without_a_second_commit(
    tmp_path: Path, store: Store
) -> None:
    """The crash window: committed, not yet marked. The corpus already says the
    right thing, so the fix is a link and no write — not another file, and not
    a row that is swept forever."""
    clone = clone_at(tmp_path)
    await seen(store, JANE, "jane")
    await registrar(clone, store).run()
    before = GitRepo.at(clone).head()

    await store.write(
        "UPDATE slack_users SET memory_id = NULL, memory_name = NULL WHERE user_id = ?", (JANE,)
    )
    result = await registrar(clone, store).run()

    assert people(clone) == ["jane.md"]
    assert GitRepo.at(clone).head() == before, "nothing needed writing"
    row = await store.get_slack_user(TEAM, JANE)
    assert row is not None and row["memory_id"] == read(clone, "jane.md").id
    assert len(result.linked) == 1

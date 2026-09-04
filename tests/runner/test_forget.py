"""`forget`, tested the way the one job that removes things has to be.

The acceptance criterion is adversarial: no configuration of inputs causes a
pinned or recent memory to be removed. So the settings here are deliberately
absurd — every threshold turned up to the point where the policy would take the
whole corpus — and the assertion is that the guards hold anyway.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from kasa.config import ForgetSettings, MemorySettings
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.ltm import MemoryStore
from kasa.memory.manifest import Manifest
from kasa.runner.forget import Collector
from kasa.store import Store

NOW = datetime(2026, 9, 4, tzinfo=UTC)

#: Everything the policy would take if nothing stopped it.
GREEDY = {"archive_below": 1.0, "archive_grace_days": 0, "max_per_run": 1_000}


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / "ltm"
    GitRepo.init(repo, branch="main")
    bootstrap(repo)
    Manifest.rebuild(repo)[0].save(repo)
    GitRepo.at(repo).commit("memory: bootstrap")
    return repo


def aged(doc: MemoryDoc, days: float) -> MemoryDoc:
    when = NOW - timedelta(days=days)
    return doc.model_copy(
        update={
            "frontmatter": doc.frontmatter.model_copy(update={"created": when, "updated": when})
        }
    )


def write(clone: Path, doc: MemoryDoc, *, path: str | None = None) -> str:
    target = clone / (path or doc.suggested_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit(f"memory: seed {doc.id}")
    return str(target.relative_to(clone))


def memory(title: str, body: str = "Something.", **fields: Any) -> MemoryDoc:
    return MemoryDoc.new(type="fact", title=title, body=body, **fields)


def collector_for(
    clone: Path, store: Store, *, policy: MemorySettings | None = None, **settings: Any
) -> Collector:
    return Collector(
        store,
        MemoryStore(GitRepo.at(clone), store, branch="main", push=False),
        settings=ForgetSettings(**settings),
        policy=policy,
        now=NOW,
    )


def live(clone: Path) -> list[str]:
    return sorted(p.name for p in (clone / "memory/facts").glob("*.md") if p.name != "README.md")


def archived(clone: Path) -> list[str]:
    return sorted(p.name for p in (clone / "memory/archive").glob("*.md"))


# -- the two transitions -----------------------------------------------------


async def test_a_cold_memory_is_archived(clone: Path, store: Store) -> None:
    write(clone, aged(memory("Forgotten", salience=0.02), days=200))

    result = await collector_for(clone, store).run()

    assert [d.title for d in result.archived] == ["Forgotten"]
    assert live(clone) == []
    assert archived(clone) == ["forgotten.md"]


async def test_a_warm_memory_is_left_alone(clone: Path, store: Store) -> None:
    write(clone, aged(memory("Still wanted", salience=0.6), days=200))

    result = await collector_for(clone, store).run()

    assert result.archived == []
    assert live(clone) == ["still-wanted.md"]


async def test_an_archived_memory_past_the_grace_period_is_collected(
    clone: Path, store: Store
) -> None:
    doc = aged(memory("Long gone", salience=0.02), days=200)
    write(clone, doc, path="memory/archive/long-gone.md")

    result = await collector_for(clone, store).run()

    assert [d.title for d in result.collected] == ["Long gone"]
    assert archived(clone) == []


async def test_an_archived_memory_inside_the_grace_period_is_not(clone: Path, store: Store) -> None:
    doc = aged(memory("Recently archived", salience=0.02), days=40)
    write(clone, doc, path="memory/archive/recent.md")

    result = await collector_for(clone, store, archive_grace_days=60).run()

    assert result.collected == []
    assert archived(clone) == ["recent.md"]


async def test_archiving_and_collecting_happen_in_one_commit(clone: Path, store: Store) -> None:
    write(clone, aged(memory("Cold", salience=0.02), days=200))
    write(clone, aged(memory("Older", salience=0.02), days=300), path="memory/archive/older.md")
    before = GitRepo.at(clone).head()

    result = await collector_for(clone, store).run()

    assert len(result.archived) == 1 and len(result.collected) == 1
    commits = GitRepo.at(clone).run("log", "--format=%H", f"{before}..HEAD").split()
    assert len(commits) == 1
    assert "Kasa-Job: forget" in GitRepo.at(clone).run("log", "-1", "--format=%B")


# -- the acceptance criterion ------------------------------------------------


@pytest.mark.parametrize(
    ("pinned", "days", "linked"),
    list(itertools.product([True, False], [1, 10, 29, 31, 200], [True, False])),
)
async def test_nothing_protected_is_ever_removed(
    clone: Path, store: Store, pinned: bool, days: float, linked: bool
) -> None:
    """The adversarial matrix. Every combination of the three guards, against a
    policy configured to take the entire corpus, at both transitions."""
    subject = aged(memory("Subject", salience=0.0, pinned=pinned), days=days)
    write(clone, subject)
    if linked:
        write(clone, memory("Pointer", f"See [[{subject.id}]]."))

    result = await collector_for(clone, store, **GREEDY).run()

    protected = pinned or days < 30 or linked
    gone = [d.memory_id for d in (*result.archived, *result.collected)]
    if protected:
        assert subject.id not in gone
        assert live(clone) == sorted({"subject.md", *(["pointer.md"] if linked else [])})
    else:
        assert subject.id in gone


@pytest.mark.parametrize("days", [1, 10, 29])
async def test_a_recent_memory_in_the_archive_is_not_collected(
    clone: Path, store: Store, days: float
) -> None:
    """The retention floor guards the second transition too. Something archived
    by hand this morning is not something to `git rm` this afternoon."""
    doc = aged(memory("Just archived", salience=0.0), days=days)
    write(clone, doc, path="memory/archive/just.md")

    result = await collector_for(clone, store, **GREEDY).run()

    assert result.collected == []
    assert archived(clone) == ["just.md"]


async def test_a_pinned_memory_survives_the_archive_too(clone: Path, store: Store) -> None:
    """`pinned` outranks every number at both transitions, including on
    something a person archived by hand."""
    doc = aged(memory("Pinned and archived", salience=0.0, pinned=True), days=500)
    write(clone, doc, path="memory/archive/pinned.md")

    result = await collector_for(clone, store, **GREEDY).run()

    assert result.collected == []
    assert archived(clone) == ["pinned.md"]


async def test_a_memory_linked_by_path_is_protected_too(clone: Path, store: Store) -> None:
    """Half the links in a hand-written corpus are paths. A version of this
    that only understood ids would collect the memories people link to most
    readably."""
    subject = aged(memory("Subject", salience=0.0), days=200)
    write(clone, subject)
    write(clone, memory("Pointer", "See [[facts/subject]]."))

    result = await collector_for(clone, store, **GREEDY).run()

    assert result.archived == []
    assert "subject.md" in live(clone)


async def test_a_link_from_the_archive_does_not_keep_a_memory_alive(
    clone: Path, store: Store
) -> None:
    """The archive is where things go to stop being the current answer.
    Letting one dead reference protect another is how a corpus never shrinks."""
    subject = aged(memory("Subject", salience=0.0), days=200)
    write(clone, subject)
    write(
        clone,
        aged(memory("Dead pointer", f"See [[{subject.id}]].", salience=0.0), days=200),
        path="memory/archive/dead.md",
    )

    result = await collector_for(clone, store, **GREEDY).run()

    assert [d.title for d in result.archived] == ["Subject"]


async def test_a_memory_that_links_to_itself_does_not_protect_itself(
    clone: Path, store: Store
) -> None:
    doc = memory("Self", "placeholder", salience=0.0)
    doc = aged(doc.model_copy(update={"body": f"\nSee [[{doc.id}]]."}), days=200)
    write(clone, doc)

    assert len((await collector_for(clone, store, **GREEDY).run()).archived) == 1


async def test_the_retention_floor_cannot_be_configured_away(clone: Path, store: Store) -> None:
    """A grace period shorter than the floor does not shorten the floor. The
    validator refuses the delete, and this refuses to propose it."""
    doc = aged(memory("Recently archived", salience=0.0), days=5)
    write(clone, doc, path="memory/archive/recent.md")

    result = await collector_for(
        clone, store, archive_grace_days=0, policy=MemorySettings(retention_floor_days=30)
    ).run()

    assert result.collected == []
    assert archived(clone) == ["recent.md"]


async def test_there_is_no_path_from_live_to_deleted_in_one_run(clone: Path, store: Store) -> None:
    """Archive-before-delete is mandatory. A memory archived by this very run
    is not collected by it — it has to sit out the grace period first, which
    starts now."""
    write(clone, aged(memory("Cold", salience=0.0), days=500))

    result = await collector_for(clone, store, **GREEDY).run()

    assert len(result.archived) == 1
    assert result.collected == []
    assert archived(clone) == ["cold.md"], "moved, not removed"


# -- bounds and recoverability -----------------------------------------------


async def test_a_run_is_bounded(clone: Path, store: Store) -> None:
    for n in range(6):
        write(clone, aged(memory(f"Cold {n}", salience=0.01 * n), days=200))

    result = await collector_for(clone, store, archive_below=1.0, max_per_run=2).run()

    assert len(result.archived) == 2
    assert result.protected["over this run's budget"] == 4


async def test_the_coldest_go_first(clone: Path, store: Store) -> None:
    write(clone, aged(memory("Coldest", salience=0.01), days=200))
    write(clone, aged(memory("Cool", salience=0.09), days=200))

    result = await collector_for(clone, store, max_per_run=1).run()

    assert [d.title for d in result.archived] == ["Coldest"]


async def test_the_reversible_half_of_a_full_week_is_the_half_that_happens(
    clone: Path, store: Store
) -> None:
    """Archiving leaves the file in the tree and its id resolving. When a week
    cannot do everything, that is the half to do."""
    write(clone, aged(memory("Cold", salience=0.0), days=200))
    write(clone, aged(memory("Older", salience=0.0), days=300), path="memory/archive/older.md")

    result = await collector_for(clone, store, max_per_run=1).run()

    assert len(result.archived) == 1
    assert result.collected == []


async def test_a_collected_memory_is_still_in_the_history(clone: Path, store: Store) -> None:
    """Delete is `git rm`. The blob stays reachable forever, which is what
    makes this safe to run at all."""
    doc = aged(memory("Long gone", "The thing it said.", salience=0.0), days=300)
    write(clone, doc, path="memory/archive/long-gone.md")

    await collector_for(clone, store, **GREEDY).run()

    assert archived(clone) == []
    log = GitRepo.at(clone).run("log", "--all", "-p", "--", "memory/archive/long-gone.md")
    assert "The thing it said." in log


async def test_a_quiet_week_writes_no_commit(clone: Path, store: Store) -> None:
    write(clone, aged(memory("Still wanted", salience=0.9), days=200))
    before = GitRepo.at(clone).head()

    result = await collector_for(clone, store).run()

    assert result.sha is None
    assert GitRepo.at(clone).head() == before
    assert "nothing forgotten" in result.summary()


async def test_what_was_spared_is_reported(clone: Path, store: Store) -> None:
    """ "Nothing was forgotten this week" and "everything was pinned" are
    different facts about a run."""
    write(clone, aged(memory("Pinned", salience=0.0, pinned=True), days=200))
    write(clone, aged(memory("New", salience=0.0), days=2))

    result = await collector_for(clone, store, **GREEDY).run()

    assert result.protected == {
        "pinned": 1,
        "younger than the retention floor": 1,
    }

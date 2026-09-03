"""The jobs this build knows how to run, and what they do when they collide."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kasa.config import Config, LTMSettings
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.index import MemoryIndex
from kasa.memory.lease import INDEX_LEASE_NAME, Lease
from kasa.memory.manifest import Manifest
from kasa.runner.jobs import default_specs
from kasa.runner.scheduler import Job, JobSpec, Scheduler
from kasa.store import Store


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / "ltm"
    GitRepo.init(repo, branch="main")
    bootstrap(repo)
    doc = MemoryDoc.new(type="person", title="Jane", body="Owns deploys.")
    target = repo / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    Manifest.rebuild(repo)[0].save(repo)
    GitRepo.at(repo).commit("memory: seed")
    return repo


def config_for(clone: Path) -> Config:
    return Config(ltm=LTMSettings(repo=str(clone), clone_path=str(clone), branch="main"))


#: The states a job row cannot leave again. `leased` is not one of them, which
#: is what made the wait below break while both jobs were still running.
TERMINAL = frozenset({"done", "failed"})


def only_spec(cfg: Config, store: Store) -> JobSpec:
    specs = default_specs(cfg, store)
    assert [spec.kind for spec in specs] == ["reindex"]
    return specs[0]


async def test_a_reindex_that_loses_the_lease_is_done_rather_than_failed(
    clone: Path, store: Store
) -> None:
    """#96 gave the rebuild a lease, which is right. A job that loses it has
    nothing left to do — the holder is doing exactly this job's work — so
    raising made it a failed attempt, and three of those a dead letter."""
    cfg = config_for(clone)
    index = MemoryIndex(store, clone)
    held = await Lease(store, index._lock_path(), name=INDEX_LEASE_NAME).acquire()
    try:
        await only_spec(cfg, store).handler(Job(id="j1", kind="reindex", payload={}, attempts=1))
    finally:
        await held.release()


async def test_two_reindex_jobs_at_once_do_not_dead_letter_each_other(
    clone: Path, store: Store
) -> None:
    """The default `concurrency=2` and one registered kind is all it takes:
    any two runnable rows are two concurrent passes.

    Read after `stop()`, not before it. The wait above says the work reached a
    state it cannot leave, and shutdown is where the drainer settles anything
    still in flight — so a snapshot taken inside the poll is a guess about the
    state under test, and this test used to assert on one.
    """
    cfg = config_for(clone)
    scheduler = Scheduler(store, default_specs(cfg, store), concurrency=2, poll_interval=0.01)
    await scheduler.queue.enqueue("reindex")
    await scheduler.queue.enqueue("reindex")

    task = asyncio.create_task(scheduler.run())
    try:
        for _ in range(2000):
            states = [row["state"] for row in await store.raw("SELECT state FROM jobs")]
            if all(state in TERMINAL for state in states):
                break
            await asyncio.sleep(0.01)
    finally:
        scheduler.stop()
        await asyncio.wait_for(task, timeout=10.0)

    rows = await store.raw("SELECT state, last_error FROM jobs")
    assert [row["state"] for row in rows] == ["done", "done"]
    assert [row["last_error"] for row in rows] == [None, None]


async def test_the_one_that_won_the_lease_still_did_the_work(clone: Path, store: Store) -> None:
    """A no-op for the loser only holds up if the winner indexed the repo."""
    cfg = config_for(clone)
    handler = only_spec(cfg, store).handler
    job = Job(id="j1", kind="reindex", payload={}, attempts=1)

    await asyncio.gather(handler(job), handler(job))

    assert (await store.raw("SELECT COUNT(*) AS n FROM chunks"))[0]["n"] > 0

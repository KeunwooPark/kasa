"""The jobs this build knows how to run, and what they do when they collide."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import os
from pathlib import Path
from typing import Any

import pytest

from kasa.config import Config, LTMSettings, ProviderConfig
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.index import MemoryIndex
from kasa.memory.lease import INDEX_LEASE_NAME, Lease
from kasa.memory.manifest import Manifest
from kasa.runner.jobs import EVERY_FIVE_MINUTES, default_specs
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
    """The reindex spec, from a config that registers nothing else.

    `config_for` deliberately has no model configured, so `episode_close` does
    not register and this stays the one spec these tests are about.
    """
    specs = default_specs(cfg, store)
    assert [spec.kind for spec in specs] == ["reindex"]
    return specs[0]


def test_reindex_polls_for_merged_supervised_prs(clone: Path, store: Store) -> None:
    spec = only_spec(config_for(clone), store)
    assert spec.cron is not None
    assert spec.cron.expression == "* * * * *"


# -- what registers, and on what ---------------------------------------------


def with_model(cfg: Config) -> Config:
    return cfg.model_copy(
        update={"llm": {"chat": ProviderConfig(kind="anthropic", model="claude-opus-5")}}
    )


def test_episode_close_registers_wherever_there_is_a_model(store: Store) -> None:
    """No repo, and it still registers: it writes to SQLite, and it is what
    fills the queue `promote` will later drain."""
    specs = default_specs(with_model(Config()), store)

    assert [spec.kind for spec in specs] == ["episode_close"]
    assert specs[0].cron is not None
    assert specs[0].cron.expression == EVERY_FIVE_MINUTES


def test_a_build_with_no_model_registers_no_consolidation(clone: Path, store: Store) -> None:
    """`kasa job list` has to work on a machine with no API key exported, and a
    job that fails on every tick is worse than one that is not there."""
    assert "episode_close" not in [spec.kind for spec in default_specs(config_for(clone), store)]


async def test_the_job_closes_the_session_its_payload_names(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit session end, end to end through the queue. The episode is
    empty, so it closes without ever reaching a provider — which is the only
    reason this can run without one."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    await store.ensure_session("cli:1", surface="cli")
    episode_id = await store.ensure_episode("cli:1")
    scheduler = Scheduler(store, default_specs(with_model(Config()), store))

    row = await scheduler.run_now("episode_close", {"session_id": "cli:1"})

    assert row["state"] == "done", row["last_error"]
    episode = await store.episode(episode_id)
    assert episode is not None and episode["state"] == "closed"


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


async def test_a_reindex_that_cannot_lock_at_all_is_not_reported_as_done(
    clone: Path, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#116 reads a lost index lease as "another rebuild is doing this work".

    On a filesystem where `flock` fails rather than blocks — NFS with no lock
    daemon, some FUSE mounts — nothing was holding it and nothing else was
    going to do the work. The row said `done`, the index stayed empty, and the
    explanation was logged at INFO, which `kasa job run` does not print
    without `-v`: broken, silent, and reporting success.
    """
    real_flock = fcntl.flock

    def enolck(fd: int, operation: int) -> None:
        if operation & fcntl.LOCK_EX:
            raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", enolck)
    cfg = config_for(clone)

    row = await Scheduler(store, default_specs(cfg, store)).run_now("reindex")

    assert row["state"] != "done", "nothing was indexed; saying otherwise is the bug"
    assert "No locks available" in str(row["last_error"])
    assert (await store.raw("SELECT COUNT(*) AS n FROM chunks"))[0]["n"] == 0


async def test_the_skip_for_a_lease_someone_else_holds_is_said_out_loud(
    clone: Path, store: Store, caplog: Any
) -> None:
    """Doing nothing is the right call there, but it is still a pass that did
    not run, and INFO is below what `kasa job run` prints without `-v`."""
    cfg = config_for(clone)
    index = MemoryIndex(store, clone)
    held = await Lease(store, index._lock_path(), name=INDEX_LEASE_NAME).acquire()
    try:
        job = Job(id="j1", kind="reindex", payload={}, attempts=1)
        with caplog.at_level("WARNING", logger="kasa.runner.jobs"):
            await only_spec(cfg, store).handler(job)
    finally:
        await held.release()

    assert "another rebuild already holds the lease" in caplog.text


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
            states = [
                row["state"]
                for row in await store.raw("SELECT state FROM jobs WHERE id NOT LIKE 'reindex@%'")
            ]
            if all(state in TERMINAL for state in states):
                break
            await asyncio.sleep(0.01)
    finally:
        scheduler.stop()
        await asyncio.wait_for(task, timeout=10.0)

    rows = await store.raw("SELECT state, last_error FROM jobs WHERE id NOT LIKE 'reindex@%'")
    assert [row["state"] for row in rows] == ["done", "done"]
    assert [row["last_error"] for row in rows] == [None, None]


async def test_the_one_that_won_the_lease_still_did_the_work(clone: Path, store: Store) -> None:
    """A no-op for the loser only holds up if the winner indexed the repo."""
    cfg = config_for(clone)
    handler = only_spec(cfg, store).handler
    job = Job(id="j1", kind="reindex", payload={}, attempts=1)

    await asyncio.gather(handler(job), handler(job))

    assert (await store.raw("SELECT COUNT(*) AS n FROM chunks"))[0]["n"] > 0

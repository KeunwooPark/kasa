"""The git write path, against real repositories on disk.

Nothing here is mocked out. The remote is a bare repo, the lease is a real
flock, and the pushes are real pushes — the failures this code exists to survive
(a lost race, a crash mid-write, a dirty working copy) only happen in git.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from kasa.config import Config
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo, run_git
from kasa.memory.layout import MANIFEST_PATH
from kasa.memory.lease import Lease, LeaseError, stale_lease
from kasa.memory.ltm import CommitMeta, MemoryStore, MemoryStoreError, Remove, Write
from kasa.memory.manifest import Manifest
from kasa.store import Store

META = CommitMeta(summary="promote 1 observation", job="promote", job_id="job_01", memory_ids=["m"])


def a_memory(title: str = "Jane", body: str = "Owns deploys.") -> tuple[str, str]:
    doc = MemoryDoc.new(type="person", title=title, body=body)
    return doc.suggested_path(), doc.render()


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    path = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(path))
    return path


@pytest.fixture
def clone(tmp_path: Path, remote: Path) -> Path:
    """A bootstrapped working copy with its bootstrap commit already pushed."""
    path = tmp_path / "ltm"
    repo = GitRepo.init(path, branch="main")
    bootstrap(path)
    repo.commit("memory: bootstrap")
    repo.set_remote(str(remote))
    repo.push("main", set_upstream=True)
    return path


def config_for(clone: Path, tmp_path: Path) -> Config:
    return Config.model_validate(
        {
            "ltm": {"repo": "someone/mem", "clone_path": str(clone), "branch": "main"},
            "store": {"path": str(tmp_path / "kasa.db")},
        }
    )


@pytest.fixture
async def memory(clone: Path, tmp_path: Path, store: Store) -> AsyncIterator[MemoryStore]:
    yield await MemoryStore.open(config_for(clone, tmp_path), store)


# -- the ordinary path -------------------------------------------------------


async def test_apply_writes_commits_and_pushes(memory: MemoryStore, remote: Path) -> None:
    path, content = a_memory()
    result = await memory.apply([Write(path, content)], META)

    assert result.sha
    assert result.pushed
    assert path in result.changed
    assert MANIFEST_PATH in result.changed, "the manifest is rewritten with every change"
    assert memory.read(path) == content

    landed = run_git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    assert path in landed


async def test_the_commit_message_carries_machine_readable_trailers(
    memory: MemoryStore,
) -> None:
    path, content = a_memory()
    await memory.apply([Write(path, content)], META)

    message = memory._repo.run("log", "-1", "--format=%B")
    assert message.startswith("memory: promote 1 observation")
    assert "Kasa-Job: promote" in message
    assert "Kasa-Job-Id: job_01" in message
    assert "Kasa-Memory-Ids: m" in message


async def test_the_manifest_is_rebuilt_from_what_landed(memory: MemoryStore) -> None:
    doc = MemoryDoc.new(type="person", title="Jane")
    await memory.apply([Write(doc.suggested_path(), doc.render())], META)

    manifest = memory.manifest()
    assert manifest.path_of(doc.id) == doc.suggested_path()


async def test_an_empty_plan_does_nothing(memory: MemoryStore) -> None:
    before = memory._repo.head()
    result = await memory.apply([], META)

    assert result.sha is None
    assert memory._repo.head() == before


async def test_delete_is_git_rm_so_the_blob_stays_in_history(memory: MemoryStore) -> None:
    path, content = a_memory()
    await memory.apply([Write(path, content)], META)
    await memory.apply([Remove(path)], CommitMeta(summary="forget one", job="forget"))

    assert not (memory.path / path).exists()
    # The whole undo story rests on this: the content is still reachable.
    assert content.splitlines()[2] in memory._repo.run("show", f"HEAD~1:{path}")


async def test_removing_something_absent_is_an_error_not_a_silent_pass(
    memory: MemoryStore,
) -> None:
    with pytest.raises(MemoryStoreError, match="does not exist"):
        await memory.apply([Remove("memory/facts/ghost.md")], META)


# -- refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "../../etc/passwd",
        "memory/../../escape.md",
        "/etc/passwd",
        MANIFEST_PATH.replace("manifest.json", "schema.md"),
        "README.md",
        "memory/people/jane.txt",
    ],
)
async def test_paths_outside_the_memory_tree_are_refused(memory: MemoryStore, path: str) -> None:
    with pytest.raises(MemoryStoreError):
        await memory.apply([Write(path, "x")], META)


async def test_a_refused_path_is_caught_before_the_lease_is_taken(
    memory: MemoryStore, store: Store
) -> None:
    with pytest.raises(MemoryStoreError):
        await memory.apply([Write("../escape.md", "x")], META)
    assert await store.get_lease("ltm") is None


async def test_a_failing_plan_leaves_the_tree_byte_identical(memory: MemoryStore) -> None:
    """All or nothing: a half-applied plan is a corpus nobody can reason about."""
    good, content = a_memory()
    before = memory._repo.head()

    with pytest.raises(MemoryStoreError):
        await memory.apply([Write(good, content), Remove("memory/facts/ghost.md")], META)

    assert not (memory.path / good).exists(), "the first write was rolled back"
    assert memory._repo.head() == before
    assert not memory._repo.is_dirty()


# -- the lease ---------------------------------------------------------------


async def test_a_second_writer_is_refused_not_merged(memory: MemoryStore, store: Store) -> None:
    """The acceptance criterion: concurrent writers are serialized."""
    held = memory.lease()
    await held.acquire(job="promote")
    try:
        with pytest.raises(LeaseError, match="promote"):
            await Lease(store, memory.lock_path).acquire(job="forget")
    finally:
        await held.release()

    # Once released, the next writer proceeds normally.
    path, content = a_memory()
    assert (await memory.apply([Write(path, content)], META)).sha


async def test_the_lease_is_released_even_when_the_write_fails(
    memory: MemoryStore, store: Store
) -> None:
    with pytest.raises(MemoryStoreError):
        await memory.apply([Remove("memory/facts/ghost.md")], META)
    assert await store.get_lease("ltm") is None
    assert (await memory.apply([Write(*a_memory())], META)).sha, "not wedged"


async def test_the_lease_row_names_who_holds_it(memory: MemoryStore, store: Store) -> None:
    async with memory.lease():
        row = await store.get_lease("ltm")
    assert row is not None and ":" in row["holder"], "host:pid"


# -- crash recovery ----------------------------------------------------------


async def test_a_crash_leaves_a_lease_row_that_the_next_run_reports(
    clone: Path, tmp_path: Path, store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    """The acceptance criterion: killing the process mid-apply stays recoverable."""
    # Exactly what a killed process leaves: a row, and no flock, because the
    # kernel released that when it died.
    await store.take_lease("ltm", holder="somehost:999", job="promote", ttl_seconds=900)

    with caplog.at_level("WARNING", logger="kasa.memory.ltm"):
        recovered = await MemoryStore.open(config_for(clone, tmp_path), store)

    assert "stopped mid-write" in caplog.text
    assert await stale_lease(store, recovered.lock_path) is not None
    # And the next write simply takes it over.
    assert (await recovered.apply([Write(*a_memory())], META)).sha


async def test_a_dirty_working_copy_is_stashed_not_destroyed(memory: MemoryStore) -> None:
    """A person's hand edits are indistinguishable from a crashed job's leftovers."""
    hand_edit = memory.path / "memory/facts/mine.md"
    hand_edit.write_text("something a person was in the middle of writing\n")

    result = await memory.apply([Write(*a_memory())], META)

    assert result.stashed
    assert not hand_edit.exists(), "parked, not left in the way"
    assert "something a person" in memory._repo.run("stash", "show", "-p", "--include-untracked")


async def test_recovery_is_a_no_op_on_a_clean_tree(memory: MemoryStore) -> None:
    assert memory.recover() is False


# -- racing another machine --------------------------------------------------


async def test_a_push_that_lost_a_race_is_rebased_never_forced(
    memory: MemoryStore, remote: Path, tmp_path: Path
) -> None:
    """The other half of "serialized, not merged badly": another clone got there first."""
    other = GitRepo.clone(str(remote), tmp_path / "other", branch="main")
    theirs = other.path / "memory/facts/theirs.md"
    theirs.parent.mkdir(parents=True, exist_ok=True)
    doc = MemoryDoc.new(type="fact", title="Theirs")
    theirs.write_text(doc.render())
    other.commit("memory: from another machine")
    other.push("main")

    path, content = a_memory()
    result = await memory.apply([Write(path, content)], META)

    assert result.pushed
    landed = run_git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    assert "memory/facts/theirs.md" in landed, "the other machine's commit survived"
    assert path in landed, "and so did ours"


async def test_the_remote_is_never_rewritten(memory: MemoryStore, remote: Path) -> None:
    """Every commit that ever landed must still be reachable. That is the undo buffer."""
    shas = []
    for i in range(3):
        result = await memory.apply([Write(*a_memory(title=f"Person {i}"))], META)
        shas.append(result.sha)

    reachable = run_git("rev-list", "main", cwd=remote).stdout.split()
    assert all(sha in reachable for sha in shas if sha)


async def test_writing_works_without_a_remote(tmp_path: Path, store: Store) -> None:
    """Memory is usable offline; the next write pushes."""
    path = tmp_path / "local"
    repo = GitRepo.init(path, branch="main")
    bootstrap(path)
    repo.commit("memory: bootstrap")

    memory = await MemoryStore.open(config_for(path, tmp_path), store)
    result = await memory.apply([Write(*a_memory())], META)

    assert result.sha
    assert result.pushed is False


async def test_an_unreachable_remote_still_commits_locally(
    memory: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    memory._repo.set_remote("https://example.invalid/nope.git")

    with caplog.at_level("WARNING", logger="kasa.memory.ltm"):
        result = await memory.apply([Write(*a_memory())], META)

    assert result.sha, "the commit is local; the push is what failed"
    assert result.pushed is False


# -- opening -----------------------------------------------------------------


async def test_opening_a_missing_clone_says_to_run_init(tmp_path: Path, store: Store) -> None:
    with pytest.raises(MemoryStoreError, match="kasa init"):
        await MemoryStore.open(config_for(tmp_path / "nothing", tmp_path), store)


async def test_opening_an_unbootstrapped_clone_says_to_run_init(
    tmp_path: Path, store: Store
) -> None:
    GitRepo.init(tmp_path / "bare-clone", branch="main")
    with pytest.raises(MemoryStoreError, match="skeleton"):
        await MemoryStore.open(config_for(tmp_path / "bare-clone", tmp_path), store)


async def test_reading_a_missing_memory_is_an_error(memory: MemoryStore) -> None:
    with pytest.raises(MemoryStoreError, match="does not exist"):
        memory.read("memory/people/nobody.md")


def test_commit_meta_omits_absent_trailers() -> None:
    message = CommitMeta(summary="s", job="promote").message()
    assert "Kasa-Job: promote" in message
    assert "Kasa-Job-Id" not in message
    assert "Kasa-Memory-Ids" not in message


def test_no_git_command_is_ever_forced() -> None:
    """Guarded by a test because it is the one flag that would undo the design.

    Matches the quoted string, not the prose: these modules discuss force-pushing
    at length in order to explain why they never do it.
    """
    from kasa.memory import gitcmd, ltm

    for module in (gitcmd, ltm):
        source = Path(module.__file__ or "").read_text()  # type: ignore[arg-type]
        assert '"--force"' not in source
        assert '"--force-with-lease"' not in source
        assert '"-f"' not in source


async def test_a_manifest_problem_does_not_block_the_write(
    memory: MemoryStore, caplog: pytest.LogCaptureFixture
) -> None:
    """One file somebody broke by hand must not wedge every later write."""
    broken = memory.path / "memory/facts/broken.md"
    broken.write_text("no frontmatter here\n")
    memory._repo.commit("add a broken file", paths=["memory/facts/broken.md"])

    with caplog.at_level("WARNING", logger="kasa.memory.ltm"):
        result = await memory.apply([Write(*a_memory())], META)

    assert result.sha
    assert "broken.md" in caplog.text
    assert Manifest.load(memory.path).memories, "the good memories are still indexed"


def test_gitrepo_clone_is_usable_after_reopen(clone: Path) -> None:
    assert GitRepo.at(clone).current_branch() == "main"


async def test_a_push_rejected_mid_write_is_rebased_and_retried(
    memory: MemoryStore, remote: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race the pre-write rebase cannot catch: they pushed while we were writing.

    Without this the retry path is dead code, because `_sync_with_remote` picks
    up anything that landed *before* the write started.
    """
    other = GitRepo.clone(str(remote), tmp_path / "other", branch="main")
    original = MemoryStore._rewrite_manifest
    raced = False

    def race(self: MemoryStore) -> str:
        # Runs after our files are written and before our commit is pushed.
        nonlocal raced
        if not raced:
            raced = True
            theirs = other.path / "memory/facts/theirs.md"
            theirs.parent.mkdir(parents=True, exist_ok=True)
            theirs.write_text(MemoryDoc.new(type="fact", title="Theirs").render())
            other.commit("memory: from another machine")
            other.push("main")
        return original(self)

    monkeypatch.setattr(MemoryStore, "_rewrite_manifest", race)

    path, content = a_memory()
    result = await memory.apply([Write(path, content)], META)

    assert raced, "the race did not actually happen"
    assert result.pushed, "the rejected push should have been rebased and retried"

    landed = run_git("ls-tree", "-r", "--name-only", "main", cwd=remote).stdout
    assert "memory/facts/theirs.md" in landed, "their commit survived; nothing was forced over it"
    assert path in landed


# -- the manifest, versus a working copy edited by hand -----------------------


async def test_a_hand_written_memory_is_resolvable_without_a_patch_ever_running(
    memory: MemoryStore, clone: Path
) -> None:
    """#43: `memory_search` returned ids that `memory_read` denied existed.

    Applying a patch was the only thing that ever rebuilt the manifest, so a
    repo whose memories were written by hand — the workflow the README
    recommends — had a manifest describing none of them. Retrieval reads
    SQLite and saw every file; id resolution read the manifest and saw nothing.
    """
    path, content = a_memory(title="Jane Kowalski")
    target = clone / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    doc = MemoryDoc.parse(content)

    assert len(Manifest.load(clone)) == 0, "nothing has rebuilt it, and nothing will"
    entry = memory.manifest().resolve(doc.id)
    assert entry is not None and entry.path == path


async def test_reconciling_the_manifest_on_read_does_not_write_to_the_repo(
    memory: MemoryStore, clone: Path
) -> None:
    """A read path that commits to somebody's repo is a read path with a bug."""
    path, content = a_memory(title="Bob Nkemelu")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)
    before = (clone / MANIFEST_PATH).read_bytes()

    memory.manifest()

    assert (clone / MANIFEST_PATH).read_bytes() == before


async def test_refresh_manifest_rebuilds_and_commits_it(memory: MemoryStore, clone: Path) -> None:
    path, content = a_memory(title="Ada Achebe")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)

    result = await memory.refresh_manifest()

    assert result.rebuilt and result.memories == 1
    assert (result.added, result.removed, result.updated) == (1, 0, 0)
    assert result.summary() == "manifest rebuilt: 1 memories (1 added)"
    assert len(Manifest.load(clone)) == 1
    assert result.sha, "the repair is committed, not left dirty for the next write to stash"


async def test_an_edited_memory_is_reported_as_an_update_not_a_count(
    memory: MemoryStore, clone: Path
) -> None:
    """#58. Drift compares whole entries, so an edit changes no count at all."""
    path, content = a_memory(title="Ada Achebe")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)
    await memory.refresh_manifest()

    (clone / path).write_text(content + "\nShe also runs the Q4 release.\n")
    result = await memory.refresh_manifest()

    assert result.rebuilt
    assert (result.added, result.removed, result.updated) == (0, 0, 1)
    assert result.summary() == "manifest rebuilt: 1 memories (1 updated)"


async def test_a_memory_gone_from_disk_is_reported_as_removed(
    memory: MemoryStore, clone: Path
) -> None:
    path, content = a_memory(title="Temporary note")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)
    await memory.refresh_manifest()

    (clone / path).unlink()
    result = await memory.refresh_manifest()

    assert (result.added, result.removed, result.updated) == (0, 1, 0)
    assert result.summary() == "manifest rebuilt: 0 memories (1 removed)"


async def test_refresh_manifest_is_a_no_op_when_it_already_describes_the_repo(
    memory: MemoryStore, clone: Path
) -> None:
    path, content = a_memory(title="Ada Achebe")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)
    await memory.refresh_manifest()

    again = await memory.refresh_manifest()

    assert not again.rebuilt and again.memories == 1
    assert again.summary() == "manifest already describes all 1 memories"
    assert again.sha is None, "no empty commit"


async def test_a_memory_deleted_by_hand_leaves_the_manifest(
    memory: MemoryStore, clone: Path
) -> None:
    """Drift in either direction, not just files the manifest has never seen."""
    path, content = a_memory(title="Temporary note")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)
    await memory.refresh_manifest()
    doc = MemoryDoc.parse(content)
    (clone / path).unlink()

    assert memory.manifest().resolve(doc.id) is None


async def test_refreshing_the_manifest_never_stashes_the_memories_it_describes(
    memory: MemoryStore, clone: Path
) -> None:
    """The files being recorded are, by definition, the uncommitted ones.

    `apply` parks a dirty working copy before it writes, which is right for a
    patch and exactly wrong here — it would stash the hand-written memory and
    then record a manifest describing a file that is no longer there.
    """
    path, content = a_memory(title="Uncommitted by design")
    (clone / path).parent.mkdir(parents=True, exist_ok=True)
    (clone / path).write_text(content)

    await memory.refresh_manifest()

    assert (clone / path).exists()
    assert (clone / path).read_text() == content
    assert len(Manifest.load(clone)) == 1

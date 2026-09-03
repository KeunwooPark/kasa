from __future__ import annotations

import os
from pathlib import Path

import pytest

from kasa.errors import GitError
from kasa.memory.gitcmd import GitRepo, run_git, token_auth


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    return GitRepo.init(tmp_path / "work", branch="main")


def test_commit_returns_none_when_nothing_changed(repo: GitRepo) -> None:
    (repo.path / "a.txt").write_text("a")
    first = repo.commit("add a")
    assert first
    assert repo.commit("again") is None


def test_commit_stages_only_the_paths_it_was_given(repo: GitRepo) -> None:
    (repo.path / "mine.txt").write_text("kasa wrote this")
    (repo.path / "yours.txt").write_text("a person is mid-edit here")
    repo.commit("memory: only mine", paths=["mine.txt"])

    tracked = repo.run("ls-tree", "-r", "--name-only", "HEAD")
    assert tracked == "mine.txt"
    assert "yours.txt" in repo.run("status", "--porcelain")


@pytest.fixture
def no_git_identity(repo: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no `user.name` — a fresh server, a container, a CI runner."""
    missing = str(repo.path / "nonexistent-gitconfig")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", missing)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", missing)
    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_commit_does_not_need_a_configured_git_identity(
    repo: GitRepo, no_git_identity: None
) -> None:
    """Kasa borrows the user's machine; it must not depend on their git config."""
    (repo.path / "a.txt").write_text("a")

    assert repo.commit("add a")
    assert "Kasa" in repo.run("log", "-1", "--format=%an")


def test_rebase_does_not_need_a_configured_git_identity(
    repo: GitRepo, no_git_identity: None, tmp_path: Path
) -> None:
    """Rebase writes commits too, and CI found this the hard way.

    Without an identity `git rebase` aborts with "empty ident name", which meant
    every push that lost a race failed on any machine without a global
    `user.name` — including every container Kasa is likely to run in.
    """
    bare = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(bare))
    (repo.path / "base.txt").write_text("base")
    repo.commit("base")
    repo.set_remote(str(bare))
    repo.push("main", set_upstream=True)

    # Another clone gets a commit in first.
    other = GitRepo.clone(str(bare), tmp_path / "other", branch="main")
    (other.path / "theirs.txt").write_text("theirs")
    other.commit("theirs")
    other.push("main")

    # Ours lands on top rather than failing.
    (repo.path / "ours.txt").write_text("ours")
    repo.commit("ours")
    repo.fetch()

    assert repo.rebase_onto("main") is True
    log = repo.run("log", "--format=%s")
    assert log.splitlines() == ["ours", "theirs", "base"]


def test_stash_does_not_need_a_configured_git_identity(
    repo: GitRepo, no_git_identity: None
) -> None:
    """Stash writes commits too, and it is the crash-recovery path."""
    (repo.path / "a.txt").write_text("a")
    repo.commit("a")
    (repo.path / "b.txt").write_text("left behind by a crashed run")

    assert repo.stash("kasa: recovered") is True
    assert not repo.is_dirty()


def test_failures_carry_gits_own_diagnostics(repo: GitRepo) -> None:
    with pytest.raises(GitError) as caught:
        repo.run("checkout", "no-such-branch")
    assert "no-such-branch" in str(caught.value)


def test_queries_on_a_fresh_repo(repo: GitRepo) -> None:
    assert repo.exists
    assert not repo.has_commits()
    assert repo.head() is None
    assert repo.remote_url() is None
    assert repo.current_branch() == "main"


def test_set_remote_adds_then_updates(repo: GitRepo) -> None:
    repo.set_remote("https://example.test/a.git")
    assert repo.remote_url() == "https://example.test/a.git"
    repo.set_remote("https://example.test/b.git")
    assert repo.remote_url() == "https://example.test/b.git"


def test_clone_and_push_round_trip(tmp_path: Path) -> None:
    bare = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(bare))

    source = GitRepo.init(tmp_path / "source", branch="main")
    (source.path / "a.txt").write_text("a")
    source.commit("add a")
    source.set_remote(str(bare))
    source.push("main", set_upstream=True)

    clone = GitRepo.clone(str(bare), tmp_path / "clone", branch="main")
    assert (clone.path / "a.txt").read_text() == "a"
    assert clone.current_branch() == "main"


def test_clone_of_an_empty_repo_lands_on_the_right_branch(tmp_path: Path) -> None:
    """An empty remote has no branch to check out; `--branch` on one is fatal."""
    bare = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "trunk", str(bare))

    clone = GitRepo.clone(str(bare), tmp_path / "clone", branch="trunk")
    assert clone.current_branch() == "trunk"


def test_token_auth_keeps_the_token_out_of_argv_and_config() -> None:
    with token_auth("ghp_secret") as env:
        assert env["KASA_GIT_TOKEN"] == "ghp_secret"
        script = Path(env["GIT_ASKPASS"])
        assert script.exists()
        assert "ghp_secret" not in script.read_text(), "the script reads the env, it does not embed"
        assert script.stat().st_mode & 0o077 == 0
    assert not script.exists(), "the askpass script is removed after use"


def test_token_auth_is_a_no_op_without_a_token() -> None:
    with token_auth(None) as env:
        assert env == {}


def test_git_never_blocks_on_a_credential_prompt(tmp_path: Path) -> None:
    repo = GitRepo.init(tmp_path / "work", branch="main")
    repo.set_remote("https://example.invalid/private.git")
    (repo.path / "a.txt").write_text("a")
    repo.commit("add a")

    # Without GIT_TERMINAL_PROMPT=0 this would sit waiting for a username, which
    # in a daemon means hanging forever rather than failing.
    with pytest.raises(GitError):
        repo.push("main")
    assert os.environ.get("GIT_TERMINAL_PROMPT") is None, "the setting stays scoped to the child"


def test_ahead_of_upstream_counts_only_what_was_not_pushed(tmp_path: Path) -> None:
    remote = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(remote))
    repo = GitRepo.init(tmp_path / "clone", branch="main")
    (tmp_path / "clone" / "a.md").write_text("a")
    repo.commit("a")
    repo.set_remote(str(remote))
    repo.push("main", set_upstream=True)
    assert repo.ahead_of_upstream() == 0

    for name in ("b", "c"):
        (tmp_path / "clone" / f"{name}.md").write_text(name)
        repo.commit(name)

    assert repo.ahead_of_upstream() == 2
    repo.push("main")
    assert repo.ahead_of_upstream() == 0


def test_ahead_of_upstream_is_none_without_an_upstream(tmp_path: Path) -> None:
    """Never pushed, or deliberately local — a configuration, not a failure."""
    repo = GitRepo.init(tmp_path / "clone", branch="main")
    (tmp_path / "clone" / "a.md").write_text("a")
    repo.commit("a")

    assert repo.ahead_of_upstream() is None

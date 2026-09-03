"""A thin, synchronous wrapper around the `git` binary.

Shelling out rather than binding libgit2: the operations Kasa needs (clone,
fetch, rebase, commit, push) are exactly the ones `git` does well, and porcelain
we do not have to reimplement is porcelain that cannot disagree with the git the
user runs in the same working copy by hand.

Synchronous on purpose. Git is fast and the callers are either a CLI command or
a background job holding the single-writer lease, so there is no concurrency to
win back; async callers wrap these in `asyncio.to_thread`.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from kasa.errors import GitError

#: Git subprocesses never get to block on a credential prompt: a daemon that
#: stops to ask for a password on stdin is a daemon that hangs forever.
_NO_PROMPT = {"GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}

_ASKPASS = "#!/bin/sh\nprintf '%s' \"$KASA_GIT_TOKEN\"\n"


def git_available() -> bool:
    return shutil.which("git") is not None


def run_git(
    *args: str,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one git command and return it. Raises `GitError` when `check`."""
    command = ["git", *args]
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, **_NO_PROMPT, **(env or {})},
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (exit {result.returncode})",
            command=" ".join(command),
            stderr=result.stderr,
        )
    return result


@contextmanager
def token_auth(token: str | None) -> Iterator[dict[str, str]]:
    """Yield an environment that authenticates HTTPS git operations.

    The token goes in via `GIT_ASKPASS` rather than in the remote URL or on the
    command line, so it never lands in `.git/config`, in a reflog, or in the
    process table where every other user on the box can read it.
    """
    if not token:
        yield {}
        return
    directory = tempfile.mkdtemp(prefix="kasa-askpass-")
    script = Path(directory) / "askpass.sh"
    try:
        script.write_text(_ASKPASS)
        script.chmod(stat.S_IRWXU)
        yield {"GIT_ASKPASS": str(script), "KASA_GIT_TOKEN": token}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


@dataclass(frozen=True)
class GitRepo:
    """A local working copy."""

    path: Path
    token: str | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def at(cls, path: Path, *, token: str | None = None) -> GitRepo:
        return cls(path.expanduser(), token)

    @classmethod
    def clone(
        cls, url: str, dest: Path, *, branch: str = "main", token: str | None = None
    ) -> GitRepo:
        dest = dest.expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        repo = cls(dest, token)
        with token_auth(token) as env:
            # `--branch` is a hard error against an empty remote, which is
            # exactly what a repo Kasa just created is, so the branch is
            # selected afterwards instead.
            run_git("clone", "--origin", "origin", url, str(dest), env=env)
        repo.checkout(branch)
        return repo

    @classmethod
    def init(cls, dest: Path, *, branch: str = "main", token: str | None = None) -> GitRepo:
        dest = dest.expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        run_git("init", "--initial-branch", branch, str(dest))
        return cls(dest, token)

    # -- queries -------------------------------------------------------------

    def run(self, *args: str, check: bool = True) -> str:
        return run_git(*args, cwd=self.path, check=check).stdout.strip()

    def authed(self, *args: str, check: bool = True) -> str:
        with token_auth(self.token) as env:
            return run_git(*args, cwd=self.path, env=env, check=check).stdout.strip()

    @property
    def exists(self) -> bool:
        return (self.path / ".git").exists()

    def is_dirty(self) -> bool:
        return bool(self.run("status", "--porcelain"))

    def current_branch(self) -> str:
        # Not `rev-parse --abbrev-ref HEAD`, which fails outright on the unborn
        # HEAD of a repo that has no commits yet.
        return self.run("branch", "--show-current")

    def has_branch(self, name: str) -> bool:
        return (
            run_git(
                "rev-parse", "--verify", "--quiet", f"refs/heads/{name}", cwd=self.path, check=False
            ).returncode
            == 0
        )

    def has_commits(self) -> bool:
        return run_git("rev-parse", "--verify", "HEAD", cwd=self.path, check=False).returncode == 0

    def remote_url(self, name: str = "origin") -> str | None:
        result = run_git("remote", "get-url", name, cwd=self.path, check=False)
        return result.stdout.strip() or None

    def head(self) -> str | None:
        return self.run("rev-parse", "HEAD") if self.has_commits() else None

    # -- mutation ------------------------------------------------------------

    def checkout(self, branch: str) -> None:
        """Move onto `branch`, creating it if it does not exist yet."""
        if self.has_branch(branch):
            self.run("checkout", "--quiet", branch)
        elif self.has_commits():
            self.run("checkout", "--quiet", "-b", branch)
        else:
            # No commits means HEAD is unborn, and `checkout -b` cannot move an
            # unborn HEAD. Repointing the symbolic ref can.
            self.run("symbolic-ref", "HEAD", f"refs/heads/{branch}")

    def set_remote(self, url: str, name: str = "origin") -> None:
        verb = "set-url" if self.remote_url(name) else "add"
        self.run("remote", verb, name, url)

    def commit(self, message: str, *, paths: Sequence[str] | None = None) -> str | None:
        """Stage `paths` and commit them. Returns the SHA, or None if nothing changed.

        Only the index is committed, so an unrelated edit a person left in the
        working copy is neither staged nor swept into Kasa's commit.
        """
        self.run("add", "--", *(paths or ["."]))
        if not self.run("diff", "--cached", "--name-only"):
            return None
        # `-c` rather than `git config`: identity belongs to this commit, not to
        # a user's working copy that Kasa happens to be borrowing.
        self.run(
            "-c",
            "user.name=Kasa",
            "-c",
            "user.email=kasa@localhost",
            "commit",
            "--quiet",
            "--message",
            message,
        )
        return self.head()

    def push(self, branch: str, *, set_upstream: bool = False) -> None:
        args = ["push"]
        if set_upstream:
            args.append("--set-upstream")
        self.authed(*args, "origin", branch)

    def fetch(self) -> None:
        self.authed("fetch", "--prune", "origin")

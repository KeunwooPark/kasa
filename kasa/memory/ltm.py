"""The write path to long-term memory.

This is the only component in Kasa that can destroy something a person cannot
get back, so it is built around three rules that are not negotiable:

- **Never force-push.** History is the undo buffer. Everything else in this
  design — "delete is reversible", "you can revert a bad belief" — is a
  consequence of that one line, and a single `--force` erases all of it.
- **Delete is `git rm`.** The blob stays reachable forever.
- **Never destroy uncommitted work.** A working copy left dirty by a crashed run
  gets stashed, not reset. A stash is recoverable; a person's hand edits to a
  memory file are not.

Everything runs under the single-writer lease (see `lease.py`), because two
daemons pushing concurrently is the one way to lose data here.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, Self

from kasa.config import Config
from kasa.errors import GitError, KasaError
from kasa.github import GitHubClient, PullRequestInfo, is_full_name
from kasa.memory.bootstrap import is_bootstrapped
from kasa.memory.document import Problem
from kasa.memory.gitcmd import GitRepo
from kasa.memory.layout import MANIFEST_PATH, is_index_page, is_memory_path
from kasa.memory.lease import LOCK_FILENAME, Lease, stale_lease
from kasa.memory.manifest import Manifest
from kasa.store import Store

log = logging.getLogger(__name__)

PUSH_ATTEMPTS = 4
PUSH_BASE_DELAY = 0.5

#: git's own words when a push lost a race. The answer is to rebase and retry,
#: never to force.
_REJECTED = ("non-fast-forward", "fetch first", "rejected", "behind its remote")


class MemoryStoreError(KasaError):
    """The memory repo could not be written."""


class PullRequestOpener(Protocol):
    async def create_pull_request(
        self, full_name: str, *, head: str, base: str, title: str, body: str
    ) -> PullRequestInfo: ...


@dataclass(frozen=True, slots=True)
class Write:
    """Create or replace one file."""

    path: str
    content: str


@dataclass(frozen=True, slots=True)
class Remove:
    """`git rm` one file. The blob stays in history."""

    path: str


Change = Write | Remove


@dataclass(frozen=True, slots=True)
class CommitMeta:
    """What the commit message says, and what its trailers record.

    Machine-parseable on purpose: `git log` is the audit trail for everything the
    agent decided to believe, and an audit trail you have to grep by prose is one
    nobody reads.
    """

    summary: str
    job: str
    job_id: str | None = None
    session: str | None = None
    memory_ids: Sequence[str] = ()

    def message(self) -> str:
        trailers = [f"Kasa-Job: {self.job}"]
        if self.job_id:
            trailers.append(f"Kasa-Job-Id: {self.job_id}")
        if self.session:
            trailers.append(f"Kasa-Session: {self.session}")
        if self.memory_ids:
            trailers.append(f"Kasa-Memory-Ids: {', '.join(self.memory_ids)}")
        return f"memory: {self.summary}\n\n" + "\n".join(trailers) + "\n"


@dataclass(slots=True)
class ManifestRefresh:
    """What `refresh_manifest` found, and what it did about it."""

    memories: int = 0
    #: What the rebuild changed. Drift is detected by comparing whole entries,
    #: so a hand-edit to one file is drift with no change in count at all — a
    #: before/after count cannot describe that, and used to report it as a
    #: discrepancy between two equal numbers.
    added: int = 0
    removed: int = 0
    updated: int = 0
    rebuilt: bool = False
    sha: str | None = None
    problems: list[Problem] = field(default_factory=list)

    def summary(self) -> str:
        if not self.rebuilt:
            return f"manifest already describes all {self.memories} memories"
        changes = ", ".join(
            f"{count} {label}"
            for count, label in (
                (self.added, "added"),
                (self.removed, "removed"),
                (self.updated, "updated"),
            )
            if count
        )
        return f"manifest rebuilt: {self.memories} memories ({changes})"


@dataclass(slots=True)
class ApplyResult:
    sha: str | None = None
    changed: list[str] = field(default_factory=list)
    pushed: bool = False
    #: Set when a crashed run's leftovers were parked before this one ran.
    stashed: bool = False
    branch: str | None = None
    pull_request_url: str | None = None


class MemoryStore:
    """The local clone, and the only sanctioned way to write to it."""

    def __init__(
        self,
        repo: GitRepo,
        store: Store,
        *,
        branch: str = "main",
        push: bool = True,
        supervised: Sequence[str] = (),
        repo_name: str | None = None,
        github_token: str | None = None,
        github: PullRequestOpener | None = None,
    ) -> None:
        self._repo = repo
        self._store = store
        self._branch = branch
        self._push = push
        self._supervised = frozenset(supervised)
        self._repo_name = repo_name
        self._github_token = github_token
        self._github = github

    @classmethod
    async def open(cls, cfg: Config, store: Store) -> Self:
        """Open the configured clone, reporting anything a crash left behind."""
        path = cfg.ltm.resolved_clone_path()
        repo = GitRepo.at(path, token=cfg.ltm.token())
        if not repo.exists:
            raise MemoryStoreError(f"no memory repo at {path}; run `kasa init`")
        if not is_bootstrapped(path):
            raise MemoryStoreError(f"{path} has no memory skeleton; run `kasa init`")

        opened = cls(
            repo,
            store,
            branch=cfg.ltm.branch,
            push=bool(repo.remote_url()),
            supervised=cfg.ltm.supervised,
            repo_name=cfg.ltm.repo if cfg.ltm.repo and is_full_name(cfg.ltm.repo) else None,
            github_token=cfg.ltm.token(),
        )
        if (leftover := await stale_lease(store, opened.lock_path)) is not None:
            log.warning(
                "the previous run held the memory write lease (%s, job %s) and did not "
                "release it; it stopped mid-write",
                leftover["holder"],
                leftover["job"] or "unknown",
            )
        return opened

    @property
    def path(self) -> Path:
        return self._repo.path

    @property
    def lock_path(self) -> Path:
        return self._repo.path / ".git" / LOCK_FILENAME

    def lease(self) -> Lease:
        return Lease(self._store, self.lock_path)

    # -- reading -------------------------------------------------------------

    def read(self, relative_path: str) -> str:
        target = self._resolve(relative_path)
        if not target.exists():
            raise MemoryStoreError(f"{relative_path} does not exist in the memory repo")
        return target.read_text()

    def manifest(self) -> Manifest:
        """The id → path index, reconciled against the working copy.

        The manifest is derived data that happens to be committed, and until
        #43 the only thing that rebuilt it was applying a patch. A repo whose
        memories were written by hand — the workflow the README recommends —
        therefore had a manifest describing none of them, and `memory_read`
        denied the existence of ids `memory_search` had just returned.

        Rebuilding unconditionally would parse the whole corpus on every tool
        call, so the manifest is trusted whenever it accounts for the files on
        disk and rebuilt when it does not. In memory: a read does not write to
        anyone's repo. `refresh_manifest` is how the repair becomes durable.
        """
        manifest = Manifest.load(self._repo.path)
        if manifest.accounts_for(self._repo.path):
            return manifest

        rebuilt, problems = self._rebuild_manifest()
        log.info(
            "the manifest did not describe the working copy at %s; rebuilt it "
            "from %d memory file(s). `kasa reindex` makes this stick.",
            self._repo.path,
            len(rebuilt),
        )
        del problems  # already logged
        return rebuilt

    async def refresh_manifest(self) -> ManifestRefresh:
        """Regenerate the committed manifest from the working copy.

        `kasa reindex` calls this. The SQLite index and the manifest are both
        derived from the repo, and rebuilding only one of them is what left
        them able to disagree.
        """
        before = Manifest.load(self._repo.path)
        after, problems = self._rebuild_manifest()
        if before.memories == after.memories:
            return ManifestRefresh(memories=len(after), problems=problems)

        lease = self.lease()
        await lease.acquire(job="reindex")
        try:
            sha = await asyncio.to_thread(self._commit_manifest, after)
        finally:
            await lease.release()
        return ManifestRefresh(
            memories=len(after),
            added=len(after.memories.keys() - before.memories.keys()),
            removed=len(before.memories.keys() - after.memories.keys()),
            updated=sum(
                1
                for memory_id, entry in after.memories.items()
                if memory_id in before.memories and before.memories[memory_id] != entry
            ),
            rebuilt=True,
            sha=sha,
            problems=problems,
        )

    # -- writing -------------------------------------------------------------

    async def apply(self, changes: Sequence[Change], meta: CommitMeta) -> ApplyResult:
        """Apply `changes` as one commit, under the write lease.

        All or nothing: a failure part-way through leaves the working copy
        exactly as it was found, because a half-applied plan is a corpus nobody
        can reason about.
        """
        if not changes:
            return ApplyResult()
        for change in changes:
            self._resolve(change.path)  # rejects traversal before the lease is taken

        lease = self.lease()
        await lease.acquire(job=meta.job)
        try:
            if meta.job in self._supervised:
                return await self._apply_supervised(changes, meta)
            # Git is fast, but it is blocking, and the caller may be a chat turn.
            return await asyncio.to_thread(self._apply, changes, meta)
        finally:
            await lease.release()

    def recover(self) -> bool:
        """Park anything uncommitted, so the next write starts from a clean tree.

        Returns whether there was something to park. Stash rather than reset:
        the leftovers may be a crashed job's half-write, or they may be an edit
        somebody made by hand, and from here the two are indistinguishable.

        Which is why the stash says only what is true of both. It used to be
        labelled "recovered from an interrupted write", asserting the case that
        is safe to throw away — so somebody clearing `git stash list` would
        read a hand edit as Kasa's own debris and drop it, losing the one thing
        stashing rather than resetting exists to protect (#78). The timestamp
        is there because these accumulate, and two identically named entries
        cannot be told apart.
        """
        if not self._repo.is_dirty():
            return False
        log.warning(
            "the memory repo at %s has uncommitted changes; stashing them before writing",
            self._repo.path,
        )
        when = datetime.now(UTC).isoformat(timespec="seconds")
        return self._repo.stash(f"kasa: uncommitted changes parked before a memory write ({when})")

    # -- internals -----------------------------------------------------------

    async def _apply_supervised(self, changes: Sequence[Change], meta: CommitMeta) -> ApplyResult:
        if not self._push or not self._repo_name:
            raise MemoryStoreError("supervised jobs require a GitHub repository remote")
        if self._github is None and not self._github_token:
            raise MemoryStoreError("supervised jobs require the configured GitHub token")

        branch = f"kasa/{meta.job}-{datetime.now(UTC).date().isoformat()}"
        body = self._review_body(changes, meta)
        result = await asyncio.to_thread(self._commit_review_branch, changes, meta, branch)
        try:
            title = f"memory: {meta.summary}"
            if self._github is not None:
                pr = await self._github.create_pull_request(
                    self._repo_name, head=branch, base=self._branch, title=title, body=body
                )
            else:
                async with GitHubClient(self._github_token or "") as github:
                    pr = await github.create_pull_request(
                        self._repo_name, head=branch, base=self._branch, title=title, body=body
                    )
        except Exception:
            # The proposal remains safely pushed on its review branch. The
            # default branch and local working tree still contain no mutation.
            raise
        result.branch = branch
        result.pull_request_url = pr.html_url
        return result

    def _commit_review_branch(
        self, changes: Sequence[Change], meta: CommitMeta, branch: str
    ) -> ApplyResult:
        stashed = self.recover()
        self._sync_with_remote()
        self._repo.checkout(branch)
        try:
            result = self._apply(changes, meta, push=False)
            result.stashed = stashed
            if result.sha:
                self._repo.push(branch, set_upstream=True)
                result.pushed = True
            return result
        finally:
            self._repo.checkout(self._branch)
            self._repo.reset_hard(f"origin/{self._branch}")

    def _review_body(self, changes: Sequence[Change], meta: CommitMeta) -> str:
        archived = {
            change.path
            for change in changes
            if isinstance(change, Write) and "/archive/" in change.path
        }
        lines = [
            f"Kasa proposes this **{meta.job}** job for review.",
            "",
            f"Why: {meta.summary}",
            "",
        ]
        for change in changes:
            if isinstance(change, Remove):
                action = "Archive" if archived and "/archive/" not in change.path else "Delete"
            elif "/archive/" in change.path:
                action = "Archive as"
            elif self._exists(change.path):
                action = "Update"
            else:
                action = "Create"
            lines.append(f"- **{action}:** `{change.path}`")
        lines.extend(["", "Merging this pull request makes the changes effective."])
        return "\n".join(lines)

    def _apply(
        self, changes: Sequence[Change], meta: CommitMeta, *, push: bool | None = None
    ) -> ApplyResult:
        result = ApplyResult(stashed=self.recover())
        self._sync_with_remote()

        created = [c.path for c in changes if isinstance(c, Write) and not self._exists(c.path)]
        try:
            touched = [self._write_one(change) for change in changes]
            touched.append(self._rewrite_manifest())
        except Exception:
            self._roll_back(created)
            raise

        result.changed = sorted(set(touched))
        result.sha = self._repo.commit(meta.message(), paths=result.changed)
        if result.sha and (self._push if push is None else push):
            result.pushed = self._push_with_rebase()
        return result

    def _write_one(self, change: Change) -> str:
        target = self._resolve(change.path)
        if isinstance(change, Write):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change.content)
            return change.path

        if not target.exists():
            raise MemoryStoreError(f"cannot remove {change.path}: it does not exist")
        if self._repo.tracked(change.path):
            self._repo.rm(change.path)
        else:
            target.unlink()
        return change.path

    def _rewrite_manifest(self) -> str:
        manifest, _ = self._rebuild_manifest()
        manifest.save(self._repo.path)
        return MANIFEST_PATH

    def _rebuild_manifest(self) -> tuple[Manifest, list[Problem]]:
        manifest, problems = Manifest.rebuild(self._repo.path)
        for problem in problems:
            # Not fatal: one file somebody broke by hand must not block every
            # later write to the repo. It is loud, and `kasa doctor` sees it too.
            log.warning("manifest: %s — %s", problem.path, problem.reason)
        return manifest, problems

    def _commit_manifest(self, manifest: Manifest) -> str | None:
        # Deliberately no `recover()`: the memories this manifest is being
        # rebuilt to describe are usually the uncommitted ones, and stashing
        # them is how a repair becomes a disappearance. Only the manifest path
        # is staged, so whatever else is in flight stays in flight — the same
        # rule `init` follows when it commits only what it bootstrapped.
        manifest.save(self._repo.path)
        sha = self._repo.commit(
            "memory: rebuild the manifest from the working copy\n\nKasa-Job: reindex",
            paths=[MANIFEST_PATH],
        )
        if sha and self._push:
            self._push_with_rebase()
        return sha

    def _sync_with_remote(self) -> None:
        if not self._push:
            return
        try:
            self._repo.fetch()
            self._repo.rebase_onto(self._branch)
        except GitError as exc:
            # Offline is survivable: commit locally and push on the next write.
            log.warning("could not sync with the remote before writing: %s", exc)

    def _push_with_rebase(self) -> bool:
        for attempt in range(PUSH_ATTEMPTS):
            try:
                self._repo.push(self._branch, set_upstream=True)
                return True
            except GitError as exc:
                if not _is_rejection(exc) or attempt == PUSH_ATTEMPTS - 1:
                    log.warning("could not push memory: %s", exc)
                    return False
                # Someone else got there first. Rebase on top of them and try
                # again — never force, which would delete their commit.
                delay = PUSH_BASE_DELAY * (2**attempt) * (1 + random.random() * 0.25)
                log.info("push rejected, rebasing and retrying in %.1fs", delay)
                time.sleep(delay)
                try:
                    self._repo.fetch()
                    self._repo.rebase_onto(self._branch)
                except GitError as rebase_error:
                    log.warning("could not rebase onto the remote: %s", rebase_error)
                    return False
        return False

    def sync_default(self) -> bool:
        """Fast-forward the clean default branch; report whether HEAD changed."""
        if not self._push:
            return False
        if self._repo.is_dirty():
            raise MemoryStoreError("cannot sync the memory repo while its working tree is dirty")
        before = self._repo.head()
        self._repo.checkout(self._branch)
        self._repo.fetch()
        self._repo.rebase_onto(self._branch)
        return self._repo.head() != before

    def _roll_back(self, created: Sequence[str]) -> None:
        """Undo a partially applied plan. The tree was clean when we started."""
        for path in created:
            (self._repo.path / path).unlink(missing_ok=True)
        if self._repo.has_commits():
            self._repo.reset_hard()

    def _exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).exists()

    def _resolve(self, relative_path: str) -> Path:
        """Map a repo-relative path to disk, refusing anything outside it.

        The last line of defence rather than the first — #13's validator rejects
        these long before they get here — but this is the function that actually
        touches the filesystem, so it checks too.
        """
        # The two generated exceptions. Both are written by deterministic code
        # only: the manifest by this class, the listings by `reorganize`. A
        # patch plan cannot reach either — `is_machinery` refuses them in the
        # validator, which is the check that matters, and this is the last one.
        if (
            not is_memory_path(relative_path)
            and relative_path != MANIFEST_PATH
            and not is_index_page(relative_path)
        ):
            raise MemoryStoreError(f"{relative_path!r} is not a writable memory path")
        target = (self._repo.path / relative_path).resolve()
        root = self._repo.path.resolve()
        if not target.is_relative_to(root):
            raise MemoryStoreError(f"{relative_path!r} resolves outside the memory repo")
        return target


def _is_rejection(exc: GitError) -> bool:
    text = f"{exc}".lower()
    return any(marker in text for marker in _REJECTED)

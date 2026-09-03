"""`kasa doctor` — check the things that are expensive to discover later.

The failures this catches share a shape: they are all silent. A token that lost
a scope, a memory repo that quietly became public, a config that points at a
model nobody configured a key for. None of them announce themselves; they show
up as a job that stopped working three weeks ago, or as a private conversation
in a public repository.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kasa.config import Config, config_path, default_key_env
from kasa.errors import ConfigError, GitHubError, KasaError
from kasa.github import GitHubClient, RepoInfo, is_full_name
from kasa.memory.bootstrap import is_bootstrapped
from kasa.memory.document import MemoryError_
from kasa.memory.gitcmd import GitRepo, git_available
from kasa.memory.index import MemoryIndex
from kasa.memory.lease import LEASE_NAME, LOCK_FILENAME, stale_lease
from kasa.memory.manifest import Manifest
from kasa.store import Store

log = logging.getLogger(__name__)

REPO_WENT_PUBLIC = (
    "{name} is public.\n"
    "It was configured as a private long-term memory repo, and it holds whatever "
    "the agent has been told. Kasa will not start against it. Make it private "
    "again, and audit who had access while it was not."
)


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


@dataclass(frozen=True, slots=True)
class Report:
    checks: tuple[Check, ...]

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status is Status.FAIL)

    @property
    def ok(self) -> bool:
        return not self.failed


async def diagnose(
    cfg: Config, *, path: Path | None = None, github: GitHubClient | None = None
) -> Report:
    checks = [_config_file(path or config_path()), _git_binary()]
    checks += _models(cfg)
    checks += await _memory_repo(cfg, github=github)
    checks += await _store_checks(cfg)
    checks.append(_manifest(cfg))
    checks += _not_yet()
    return Report(tuple(checks))


async def verify_repo_visibility(cfg: Config, *, github: GitHubClient | None = None) -> None:
    """Refuse to start when the memory repo is now public.

    Only a definitive answer stops the daemon. Being unable to reach GitHub, or
    having no token to ask with, is a warning's worth of information — refusing
    to start whenever the network is down would make the check the outage.
    """
    if not cfg.ltm.configured:
        return
    try:
        info = await _lookup(cfg, github=github)
    except GitHubError as exc:
        log.warning("could not verify that %s is still private: %s", cfg.ltm.repo, exc)
        return
    if info is not None and not info.private:
        raise ConfigError(REPO_WENT_PUBLIC.format(name=info.full_name))


# -- individual checks -------------------------------------------------------


def _config_file(path: Path) -> Check:
    if not path.exists():
        return Check(
            "config",
            Status.WARN,
            f"{path} does not exist; running from the environment. Try `kasa init`.",
        )
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        return Check("config", Status.FAIL, f"{path}: {exc}")
    if mode & 0o077:
        return Check("config", Status.WARN, f"{path} is readable by other users (chmod 600 it)")
    return Check("config", Status.OK, str(path))


def _git_binary() -> Check:
    if not git_available():
        return Check("git", Status.FAIL, "git is not on PATH; long-term memory needs it")
    return Check("git", Status.OK, "available")


def _models(cfg: Config) -> list[Check]:
    if not cfg.llm:
        return [Check("models", Status.FAIL, "no model configured for the 'chat' role")]

    checks = []
    for role, provider in sorted(cfg.llm.items()):
        env = provider.key_env or default_key_env(provider.kind)
        try:
            provider.api_key()
        except ConfigError:
            checks.append(
                Check(f"model:{role}", Status.FAIL, f"{provider.model} — {env} is not set")
            )
            continue
        checks.append(Check(f"model:{role}", Status.OK, f"{provider.model} via {env}"))

    if "chat" not in cfg.llm:
        checks.append(Check("models", Status.FAIL, "no model configured for the 'chat' role"))
    return checks


async def _memory_repo(cfg: Config, *, github: GitHubClient | None) -> list[Check]:
    ltm = cfg.ltm
    if not ltm.configured:
        return [
            Check("memory repo", Status.WARN, "not configured; run `kasa init`"),
            Check("repo privacy", Status.SKIP, "no repo to check"),
            Check("clone", Status.SKIP, "no repo to check"),
        ]

    checks = [Check("memory repo", Status.OK, str(ltm.repo))]
    checks.append(_token(cfg))
    checks.append(await _privacy(cfg, github=github))
    checks.append(_clone(cfg))
    return checks


def _token(cfg: Config) -> Check:
    if cfg.ltm.token():
        return Check("github token", Status.OK, f"from {cfg.ltm.token_env}")
    return Check(
        "github token",
        Status.WARN,
        f"{cfg.ltm.token_env} is not set; Kasa cannot push memory or verify privacy",
    )


async def _privacy(cfg: Config, *, github: GitHubClient | None) -> Check:
    try:
        info = await _lookup(cfg, github=github)
    except GitHubError as exc:
        return Check("repo privacy", Status.WARN, f"could not check: {exc}")
    if info is None:
        return Check(
            "repo privacy", Status.WARN, "unverifiable — not a GitHub owner/name, or no token"
        )
    if not info.private:
        return Check("repo privacy", Status.FAIL, f"{info.full_name} is PUBLIC")
    access = "write" if info.can_push else "read only — memory cannot be pushed"
    status = Status.OK if info.can_push else Status.WARN
    return Check("repo privacy", status, f"private, {access}")


def _clone(cfg: Config) -> Check:
    path = cfg.ltm.resolved_clone_path()
    repo = GitRepo.at(path)
    if not repo.exists:
        return Check("clone", Status.WARN, f"{path} does not exist; run `kasa init`")
    if not is_bootstrapped(path):
        return Check("clone", Status.WARN, f"{path} has no memory skeleton; run `kasa init`")

    notes = [f"{path} on {repo.current_branch()}"]
    if repo.is_dirty():
        # A dirty working copy is how a crashed write announces itself, and the
        # next `apply` has to reset it before it can do anything.
        notes.append("uncommitted changes present")
        return Check("clone", Status.WARN, ", ".join(notes))
    return Check("clone", Status.OK, ", ".join(notes))


async def _store_checks(cfg: Config) -> list[Check]:
    """Database health and lease state, on one connection."""
    path = cfg.store.resolved()
    try:
        async with await Store.open(path) as store:
            rows = await store.raw("SELECT name FROM schema_version ORDER BY name")
            lease = await _lease(cfg, store)
            index = await _index(cfg, store)
    except KasaError as exc:
        return [Check("database", Status.FAIL, f"{path}: {exc}")]
    return [
        Check("database", Status.OK, f"{path}, {len(rows)} migration(s) applied"),
        lease,
        index,
    ]


async def _index(cfg: Config, store: Store) -> Check:
    if not cfg.ltm.configured:
        return Check("index freshness", Status.SKIP, "no memory repo to index")
    root = cfg.ltm.resolved_clone_path()
    if not root.exists():
        return Check("index freshness", Status.SKIP, "no clone to index")

    index = MemoryIndex(store, root)
    stats = await index.stats()
    if await index.is_stale():
        return Check(
            "index freshness",
            Status.WARN,
            f"{stats['chunks']} chunk(s) indexed, but the repo has moved on — run `kasa reindex`",
        )
    return Check(
        "index freshness",
        Status.OK,
        f"{stats['chunks']} chunk(s) across {stats['memories']} memories",
    )


def _manifest(cfg: Config) -> Check:
    """Whether `.kasa/manifest.json` still describes the repo.

    Index freshness alone reported a healthy system while the manifest was
    empty, which is how #43 stayed invisible: retrieval reads SQLite and sees
    every file, `memory_read` resolves through the manifest and sees only what
    it lists.
    """
    if not cfg.ltm.configured:
        return Check("manifest", Status.SKIP, "no memory repo")
    root = cfg.ltm.resolved_clone_path()
    if not root.exists():
        return Check("manifest", Status.SKIP, "no clone")

    try:
        manifest = Manifest.load(root)
    except MemoryError_ as exc:
        return Check("manifest", Status.FAIL, str(exc))

    rebuilt, problems = Manifest.rebuild(root)
    if problems:
        listed = ", ".join(f"{p.path} ({p.reason})" for p in problems[:3])
        return Check("manifest", Status.WARN, f"{len(problems)} unreadable file(s): {listed}")
    if not manifest.accounts_for(root):
        missing = len(rebuilt) - len(manifest)
        drift = f"{abs(missing)} memories {'missing from' if missing > 0 else 'stale in'} it"
        return Check(
            "manifest",
            Status.WARN,
            f"does not describe the repo — {drift}; run `kasa reindex`",
        )
    return Check("manifest", Status.OK, f"{len(manifest)} memories resolvable by id")


async def _lease(cfg: Config, store: Store) -> Check:
    if not cfg.ltm.configured:
        return Check("write lease", Status.SKIP, "no memory repo to write to")

    lock = cfg.ltm.resolved_clone_path() / ".git" / LOCK_FILENAME
    row = await store.get_lease(LEASE_NAME)
    if row is None:
        return Check("write lease", Status.OK, "free")
    if await stale_lease(store, lock) is not None:
        # A row whose holder is gone. Harmless — the next write takes it over —
        # but it means the previous run stopped in the middle of writing.
        return Check(
            "write lease",
            Status.WARN,
            f"left behind by {row['holder']} (job {row['job'] or 'unknown'}, "
            f"since {row['acquired_at']}); the next write will take it over",
        )
    return Check(
        "write lease", Status.OK, f"held by {row['holder']} for job {row['job'] or 'unknown'}"
    )


def _not_yet() -> list[Check]:
    """Checks whose subjects do not exist yet, listed so they are not forgotten."""
    return [Check("embeddings", Status.SKIP, "arrives with hybrid retrieval (#31)")]


async def _lookup(cfg: Config, *, github: GitHubClient | None) -> RepoInfo | None:
    """The repo as GitHub sees it, or None when it cannot be asked."""
    spec = cfg.ltm.repo or ""
    if not is_full_name(spec):
        return None
    token = cfg.ltm.token()
    if not token and github is None:
        return None

    client = github or GitHubClient(token or "")
    try:
        return await client.get_repo(spec)
    finally:
        if github is None:
            with contextlib.suppress(Exception):
                await client.aclose()

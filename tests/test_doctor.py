from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from kasa.config import Config, SlackSettings, write_config
from kasa.doctor import Check, Report, Status, _slack, diagnose, verify_repo_visibility
from kasa.errors import ConfigError
from kasa.github import GitHubClient
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo, run_git
from kasa.memory.index import MemoryIndex
from kasa.memory.manifest import Manifest
from kasa.store import Store
from kasa.vault import VAULT_ENV, Vault, clear_cache
from tests.conftest import mock_client

REPO = {
    "full_name": "someone/kasa-memory",
    "private": True,
    "default_branch": "main",
    "clone_url": "https://github.com/someone/kasa-memory.git",
    "ssh_url": "git@github.com:someone/kasa-memory.git",
    "html_url": "https://github.com/someone/kasa-memory",
    "permissions": {"push": True},
    "size": 12,
}


def github(**overrides: Any) -> GitHubClient:
    payload = {**REPO, **overrides}
    return GitHubClient(
        "t",
        client=mock_client(
            lambda r: httpx.Response(200, json=payload), base_url="https://api.github.test"
        ),
    )


def unreachable_github() -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network is down")

    return GitHubClient("t", client=mock_client(handler, base_url="https://api.github.test"))


def config_for(tmp_path: Path, **ltm: Any) -> Config:
    return Config.model_validate(
        {
            "ltm": {"repo": "someone/kasa-memory", "clone_path": str(tmp_path / "ltm"), **ltm},
            "llm": {"chat": {"kind": "anthropic", "model": "m", "key_env": "TEST_KEY"}},
            "store": {"path": str(tmp_path / "kasa.db")},
        }
    )


def status_of(report: Report, name: str) -> Status:
    return next(c.status for c in report.checks if c.name == name)


def detail_of(report: Report, name: str) -> str:
    return next(c.detail for c in report.checks if c.name == name)


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-test")
    monkeypatch.setenv("KASA_GITHUB_TOKEN", "ghp_token")


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    path = tmp_path / "ltm"
    GitRepo.init(path, branch="main")
    bootstrap(path)
    GitRepo.at(path).commit("bootstrap")
    return path


# -- the report --------------------------------------------------------------


async def test_a_healthy_setup_reports_no_failures(tmp_path: Path, clone: Path) -> None:
    report = await diagnose(config_for(tmp_path), github=github())

    assert report.ok
    assert status_of(report, "repo privacy") is Status.OK
    assert status_of(report, "clone") is Status.OK
    assert status_of(report, "model:chat") is Status.OK
    assert status_of(report, "database") is Status.OK


async def test_a_public_repo_fails_the_report(tmp_path: Path, clone: Path) -> None:
    report = await diagnose(config_for(tmp_path), github=github(private=False))

    assert not report.ok
    assert status_of(report, "repo privacy") is Status.FAIL
    assert "PUBLIC" in detail_of(report, "repo privacy")


async def test_read_only_access_warns_rather_than_fails(tmp_path: Path, clone: Path) -> None:
    """Read-only memory still answers questions; it just cannot learn."""
    report = await diagnose(config_for(tmp_path), github=github(permissions={"push": False}))

    assert report.ok
    assert status_of(report, "repo privacy") is Status.WARN
    assert "read only" in detail_of(report, "repo privacy")


async def test_a_missing_key_fails_the_role(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TEST_KEY")
    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "model:chat") is Status.FAIL
    assert "TEST_KEY is not set" in detail_of(report, "model:chat")


async def test_no_chat_model_fails(tmp_path: Path) -> None:
    report = await diagnose(Config(), github=github())
    assert not report.ok
    assert status_of(report, "models") is Status.FAIL


async def test_an_unconfigured_repo_warns_and_skips_the_rest(tmp_path: Path) -> None:
    cfg = Config.model_validate(
        {
            "llm": {"chat": {"kind": "anthropic", "model": "m", "key_env": "TEST_KEY"}},
            "store": {"path": str(tmp_path / "kasa.db")},
        }
    )
    report = await diagnose(cfg)

    assert report.ok, "not having set up memory yet is not a failure"
    assert status_of(report, "memory repo") is Status.WARN
    assert status_of(report, "repo privacy") is Status.SKIP


async def test_a_missing_clone_warns(tmp_path: Path) -> None:
    report = await diagnose(config_for(tmp_path), github=github())
    assert status_of(report, "clone") is Status.WARN
    assert "does not exist" in detail_of(report, "clone")


async def test_a_dirty_clone_warns(tmp_path: Path, clone: Path) -> None:
    """A dirty working copy is how a crashed write announces itself."""
    (clone / "memory/facts/leftover.md").write_text("half a write")

    report = await diagnose(config_for(tmp_path), github=github())
    assert status_of(report, "clone") is Status.WARN
    assert "uncommitted" in detail_of(report, "clone")


async def test_an_unreachable_github_warns_rather_than_failing(tmp_path: Path, clone: Path) -> None:
    report = await diagnose(config_for(tmp_path), github=unreachable_github())

    assert report.ok, "being offline is not a misconfiguration"
    assert status_of(report, "repo privacy") is Status.WARN


async def test_a_missing_token_is_reported_as_unverifiable(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KASA_GITHUB_TOKEN")
    report = await diagnose(config_for(tmp_path))

    assert status_of(report, "github token") is Status.WARN
    assert status_of(report, "repo privacy") is Status.WARN
    assert "unverifiable" in detail_of(report, "repo privacy")


async def test_a_world_readable_config_warns(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_config(config_for(tmp_path), path)
    path.chmod(0o644)

    report = await diagnose(config_for(tmp_path), path=path, github=github())
    assert status_of(report, "config") is Status.WARN
    assert "readable by other users" in detail_of(report, "config")


async def test_checks_that_do_not_exist_yet_are_listed_as_skipped(tmp_path: Path) -> None:
    """Named rather than absent, so they are not quietly forgotten."""
    report = await diagnose(config_for(tmp_path), github=github())
    assert status_of(report, "index freshness") is Status.SKIP


async def test_a_free_lease_reports_ok(tmp_path: Path, clone: Path) -> None:
    report = await diagnose(config_for(tmp_path), github=github())
    assert status_of(report, "write lease") is Status.OK
    assert detail_of(report, "write lease") == "free"


async def test_a_lease_left_by_a_crashed_run_warns(tmp_path: Path, clone: Path) -> None:
    cfg = config_for(tmp_path)
    async with await Store.open(cfg.store.resolved()) as store:
        await store.take_lease("ltm", holder="somehost:999", job="promote", ttl_seconds=900)

    report = await diagnose(cfg, github=github())
    assert status_of(report, "write lease") is Status.WARN
    assert "somehost:999" in detail_of(report, "write lease")


# -- the startup re-check ----------------------------------------------------


async def test_startup_refuses_a_repo_that_went_public(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="is public"):
        await verify_repo_visibility(config_for(tmp_path), github=github(private=False))


async def test_startup_allows_a_private_repo(tmp_path: Path) -> None:
    await verify_repo_visibility(config_for(tmp_path), github=github())


async def test_startup_does_not_block_on_an_unreachable_github(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Refusing to start whenever the network is down would make the check the outage."""
    with caplog.at_level("WARNING", logger="kasa.doctor"):
        await verify_repo_visibility(config_for(tmp_path), github=unreachable_github())

    assert "could not verify" in caplog.text, "silently skipping it would be worse"


async def test_startup_skips_when_no_repo_is_configured() -> None:
    await verify_repo_visibility(Config())


async def test_startup_skips_a_plain_git_url(tmp_path: Path) -> None:
    """Nothing to ask GitHub about; init already warned when it was configured."""
    await verify_repo_visibility(config_for(tmp_path, repo="git@example.test:a/b.git"))


# -- the manifest ------------------------------------------------------------


async def test_a_manifest_that_does_not_describe_the_repo_is_reported(
    tmp_path: Path, clone: Path
) -> None:
    """Index freshness alone called this system healthy, which is how #43 hid.

    Retrieval reads SQLite and sees every file on disk; id resolution reads the
    manifest and sees only what it lists. Nothing compared the two.
    """
    doc = MemoryDoc.new(type="fact", title="Written by hand", body="Not by a patch.")
    target = clone / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "manifest") is Status.WARN
    assert "reindex" in detail_of(report, "manifest")


async def test_a_manifest_in_step_with_the_repo_passes(tmp_path: Path, clone: Path) -> None:
    doc = MemoryDoc.new(type="fact", title="Recorded", body="By a patch, notionally.")
    target = clone / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    rebuilt, _ = Manifest.rebuild(clone)
    rebuilt.save(clone)

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "manifest") is Status.OK
    assert "1 memories resolvable" in detail_of(report, "manifest")


# -- index freshness ---------------------------------------------------------


async def test_a_stale_index_says_to_reindex(tmp_path: Path, clone: Path) -> None:
    doc = MemoryDoc.new(type="fact", title="Never indexed", body="Written by hand.")
    target = clone / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "index freshness") is Status.WARN
    assert "run `kasa reindex`" in detail_of(report, "index freshness")


async def test_an_unreadable_file_is_not_reported_as_staleness(tmp_path: Path, clone: Path) -> None:
    """#69. `kasa reindex` had already run three times and could not clear it,
    because a file the parser refuses never reaches `index_state`."""
    async with await Store.open(tmp_path / "kasa.db") as store:
        await MemoryIndex(store, clone).reindex()
    (clone / "memory" / "facts" / "broken.md").write_text("no frontmatter here at all")

    report = await diagnose(config_for(tmp_path), github=github())
    detail = detail_of(report, "index freshness")

    assert status_of(report, "index freshness") is Status.WARN
    assert "the repo has moved on" not in detail
    assert "run `kasa reindex`" not in detail, "it has, and it cannot fix this"
    assert "memory/facts/broken.md" in detail
    assert "cannot be indexed" in detail


async def test_a_clean_index_says_what_it_holds(tmp_path: Path, clone: Path) -> None:
    async with await Store.open(tmp_path / "kasa.db") as store:
        await MemoryIndex(store, clone).reindex()

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "index freshness") is Status.OK


async def test_a_parked_edit_is_reported_where_somebody_would_look(
    tmp_path: Path, clone: Path
) -> None:
    """#78. A write over uncommitted work stashes it and says so only in a log
    record nothing prints. Recoverable, with nobody told there is anything to
    recover."""
    (clone / "memory" / "facts").mkdir(parents=True, exist_ok=True)
    (clone / "memory" / "facts" / "by-hand.md").write_text("mid-edit\n")
    GitRepo.at(clone).stash("kasa: uncommitted changes parked before a memory write (t)")

    report = await diagnose(config_for(tmp_path), github=github())
    detail = detail_of(report, "clone")

    assert status_of(report, "clone") is Status.WARN
    assert "1 stashed change(s)" in detail
    assert "stash list" in detail, "and how to look at them"


async def test_a_clone_with_nothing_parked_is_ok(tmp_path: Path, clone: Path) -> None:
    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "clone") is Status.OK
    assert "stashed" not in detail_of(report, "clone")


async def test_memory_that_could_not_be_pushed_is_reported(tmp_path: Path, clone: Path) -> None:
    """#91. A failed push keeps its local commit, which is right — and nothing
    said so. `ApplyResult.pushed` was read by nobody and the log record goes
    nowhere without a handler."""
    remote = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(remote))
    repo = GitRepo.at(clone)
    repo.set_remote(str(remote))
    repo.push("main", set_upstream=True)

    doc = MemoryDoc.new(type="fact", title="Written offline", body="Never left this disk.")
    target = clone / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    repo.commit("memory: written offline")

    report = await diagnose(config_for(tmp_path), github=github())
    detail = detail_of(report, "clone")

    assert status_of(report, "clone") is Status.WARN
    assert "1 commit(s) not pushed" in detail
    assert "only on this disk" in detail


async def test_a_clone_in_step_with_its_remote_is_ok(tmp_path: Path, clone: Path) -> None:
    remote = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(remote))
    repo = GitRepo.at(clone)
    repo.set_remote(str(remote))
    repo.push("main", set_upstream=True)

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "clone") is Status.OK
    assert "not pushed" not in detail_of(report, "clone")


async def test_a_clone_with_no_upstream_is_not_called_unpushed(tmp_path: Path, clone: Path) -> None:
    """Never pushed, or deliberately local. A configuration, not a failure."""
    assert GitRepo.at(clone).ahead_of_upstream() is None

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "clone") is Status.OK
    assert "not pushed" not in detail_of(report, "clone")


def slack_config(**settings: object) -> Config:
    return Config(slack=SlackSettings(**settings))  # type: ignore[arg-type]


async def test_slack_is_skipped_when_it_is_not_configured() -> None:
    assert _slack(slack_config()) == [
        Check("slack", Status.SKIP, "not configured; `kasa run` serves the terminal")
    ]


async def test_a_missing_slack_token_fails_before_the_daemon_needs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Socket Mode daemon with one of the two tokens missing fails on
    connect, in a library, minutes into a deploy."""
    monkeypatch.setenv("KASA_SLACK_BOT", "xoxb-1")
    monkeypatch.delenv("KASA_SLACK_APP", raising=False)
    cfg = slack_config(bot_token_env="KASA_SLACK_BOT", app_token_env="KASA_SLACK_APP")

    check = _slack(cfg)[0]

    assert check.status is Status.FAIL
    assert check.detail == "KASA_SLACK_APP not set"


async def test_a_configured_slack_says_where_it_will_listen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KASA_SLACK_BOT", "xoxb-1")
    monkeypatch.setenv("KASA_SLACK_APP", "xapp-1")
    cfg = slack_config(
        bot_token_env="KASA_SLACK_BOT",
        app_token_env="KASA_SLACK_APP",
        allowed_channels=["C0DEPLOY", "C0GENERAL"],
    )

    assert _slack(cfg)[0] == Check("slack", Status.OK, "socket mode; C0DEPLOY, C0GENERAL")


# -- the vault ---------------------------------------------------------------


def stored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **secrets: str) -> Path:
    vault = Vault(tmp_path / "vault.json")
    for name, value in secrets.items():
        vault.set(name, value)
    vault.save()
    monkeypatch.setenv(VAULT_ENV, str(vault.path))
    clear_cache()
    return vault.path


async def test_no_vault_is_skipped_rather_than_flagged(tmp_path: Path, clone: Path) -> None:
    """Most installations have no vault. That is not a finding."""
    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "vault") is Status.SKIP


async def test_a_healthy_vault_reports_how_much_it_holds(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = stored(tmp_path, monkeypatch, SOME_TOKEN="a-stored-credential")
    report = await diagnose(config_for(tmp_path), github=github())

    assert report.ok
    assert status_of(report, "vault") is Status.OK
    assert "1 secret(s)" in detail_of(report, "vault")
    assert "a-stored-credential" not in detail_of(report, "vault")
    assert str(path) in detail_of(report, "vault")


async def test_a_readable_vault_fails_the_report(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And fails it as a check, rather than taking the whole report down."""
    path = stored(tmp_path, monkeypatch, SOME_TOKEN="a-stored-credential")
    path.chmod(0o644)

    report = await diagnose(config_for(tmp_path), github=github())

    assert not report.ok
    assert status_of(report, "vault") is Status.FAIL
    assert status_of(report, "repo privacy") is Status.OK  # the rest still ran


async def test_a_vault_inside_the_memory_repo_fails(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worst case: a credential store in a directory jobs push on a schedule."""
    vault = Vault(clone / "vault.json")
    vault.set("SOME_TOKEN", "a-stored-credential")
    vault.save()
    monkeypatch.setenv(VAULT_ENV, str(vault.path))
    clear_cache()

    report = await diagnose(config_for(tmp_path), github=github())

    assert not report.ok
    assert status_of(report, "vault") is Status.FAIL
    assert "inside the long-term memory repo" in detail_of(report, "vault")


async def test_a_vault_in_a_git_work_tree_warns(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dotfiles case: a 0600 file somebody's home-directory repo would push."""
    home = tmp_path / "home"
    GitRepo.init(home, branch="main")
    vault = Vault(home / "share" / "vault.json")
    vault.set("SOME_TOKEN", "a-stored-credential")
    vault.save()
    monkeypatch.setenv(VAULT_ENV, str(vault.path))
    clear_cache()

    report = await diagnose(config_for(tmp_path), github=github())

    assert report.ok  # a warning, not a refusal
    assert status_of(report, "vault location") is Status.WARN


async def test_an_exported_variable_shadowing_the_vault_warns(
    tmp_path: Path, clone: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise a rotated key looks stored and quietly has no effect."""
    stored(tmp_path, monkeypatch, TEST_KEY="the-stored-one")
    monkeypatch.setenv("TEST_KEY", "a-different-exported-one")

    report = await diagnose(config_for(tmp_path), github=github())

    assert status_of(report, "vault shadowed") is Status.WARN
    assert "TEST_KEY" in detail_of(report, "vault shadowed")
    assert "the-stored-one" not in detail_of(report, "vault shadowed")

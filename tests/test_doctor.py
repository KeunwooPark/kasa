from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from kasa.config import Config, write_config
from kasa.doctor import Report, Status, diagnose, verify_repo_visibility
from kasa.errors import ConfigError
from kasa.github import GitHubClient
from kasa.memory.bootstrap import bootstrap
from kasa.memory.gitcmd import GitRepo
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
    assert status_of(report, "write lease") is Status.SKIP
    assert status_of(report, "index freshness") is Status.SKIP


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

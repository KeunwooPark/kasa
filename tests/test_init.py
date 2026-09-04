"""`kasa init` end to end, against a real git remote on disk.

The remote is a bare repo in `tmp_path` and GitHub is a mock transport, so the
whole setup path — create, clone, bootstrap, commit, push — runs for real
without a network or an account.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from kasa.config import ProviderKind, load_config
from kasa.errors import ConfigError
from kasa.github import GitHubClient
from kasa.init import FAUCET_BASE_URL, run_init
from kasa.memory.gitcmd import GitRepo, run_git
from kasa.memory.layout import MANIFEST_PATH, SCHEMA_PATH
from tests.conftest import mock_client

TOKEN_ENV = "KASA_GITHUB_TOKEN"


class ScriptedPrompter:
    """Answers questions by substring match, falling back to the default.

    Matching on a fragment rather than replaying a fixed sequence means a test
    states only the answers it cares about, and adding a prompt to the flow does
    not invalidate every test that came before it.
    """

    def __init__(
        self,
        answers: dict[str, str] | None = None,
        confirms: dict[str, bool] | None = None,
        *,
        default_confirm: bool = True,
    ) -> None:
        self.answers = answers or {}
        self.confirms = confirms or {}
        self.default_confirm = default_confirm
        self.said: list[str] = []
        self.warned: list[str] = []
        self.asked: list[str] = []
        self.chosen: list[str] = []
        self.selected: list[str] = []

    def _match(self, question: str, table: dict[str, Any]) -> Any | None:
        for fragment, value in table.items():
            if fragment.lower() in question.lower():
                return value
        return None

    def ask(self, question: str, *, default: str | None = None) -> str:
        self.asked.append(question)
        answer = self._match(question, self.answers)
        return answer if answer is not None else (default or "")

    def choose(self, question: str, choices: tuple[str, ...], *, default: str) -> str:
        self.chosen.append(question)
        answer = self._match(question, self.answers)
        return answer if answer is not None else default

    def select(self, question: str, options: tuple[str, ...], *, default: str) -> str:
        self.selected.append(question)
        answer = self._match(question, self.answers)
        return answer if answer is not None else default

    def confirm(self, question: str, *, default: bool = False) -> bool:
        answer = self._match(question, self.confirms)
        return answer if answer is not None else self.default_confirm

    def say(self, text: str) -> None:
        self.said.append(text)

    def warn(self, text: str) -> None:
        self.warned.append(text)


def repo_payload(
    full_name: str = "someone/kasa-memory",
    *,
    private: bool = True,
    clone_url: str = "",
    empty: bool = True,
) -> dict[str, Any]:
    return {
        "full_name": full_name,
        "private": private,
        "default_branch": "main",
        "clone_url": clone_url,
        "ssh_url": f"git@github.com:{full_name}.git",
        "html_url": f"https://github.com/{full_name}",
        "permissions": {"push": True},
        "size": 0 if empty else 12,
    }


def fake_github(
    *,
    exists: bool = False,
    private: bool = True,
    clone_url: str = "",
    empty: bool = True,
    calls: list[str] | None = None,
) -> GitHubClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(f"{request.method} {request.url.path}")
        payload = repo_payload(private=private, clone_url=clone_url, empty=empty)
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "someone"})
        if request.method == "GET" and not exists:
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(200 if request.method == "GET" else 201, json=payload)

    return GitHubClient("t", client=mock_client(handler, base_url="https://api.github.test"))


@pytest.fixture
def remote(tmp_path: Path) -> str:
    """A bare repo standing in for the one on GitHub."""
    path = tmp_path / "remote.git"
    run_git("init", "--bare", "--initial-branch", "main", str(path))
    return str(path)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (TOKEN_ENV, "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "FAUCET_API_KEY"):
        monkeypatch.delenv(name, raising=False)


async def do_init(
    tmp_path: Path,
    remote: str,
    *,
    prompter: ScriptedPrompter | None = None,
    **github: Any,
) -> tuple[ScriptedPrompter, Path, Path]:
    config = tmp_path / "config.toml"
    clone = tmp_path / "ltm"
    prompter = prompter or ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(clone)}
    )
    client = fake_github(clone_url=remote, **github)
    try:
        await run_init(prompter, path=config, github=client)
    finally:
        await client.aclose()
    return prompter, config, clone


# -- the happy path ----------------------------------------------------------


async def test_init_creates_clones_bootstraps_and_pushes(tmp_path: Path, remote: str) -> None:
    calls: list[str] = []
    _, config, clone = await do_init(tmp_path, remote, calls=calls)

    assert "POST /user/repos" in calls, "a missing repo should be created"
    assert (clone / SCHEMA_PATH).exists()
    assert (clone / MANIFEST_PATH).exists()
    assert (clone / "memory/people/.gitkeep").exists()
    assert (clone / "README.md").exists()

    cfg = load_config(config)
    assert cfg.ltm.repo == "someone/kasa-memory"
    assert cfg.ltm.branch == "main"
    assert cfg.llm["chat"].model

    # The bootstrap commit reached the remote, not just the working copy.
    listing = run_git("ls-tree", "-r", "--name-only", "main", cwd=Path(remote)).stdout
    assert SCHEMA_PATH in listing


async def test_manifest_starts_empty_and_valid(tmp_path: Path, remote: str) -> None:
    _, _, clone = await do_init(tmp_path, remote)
    manifest = json.loads((clone / MANIFEST_PATH).read_text())
    assert manifest["memories"] == {}
    assert manifest["version"] >= 1


async def test_existing_repo_is_cloned_rather_than_created(tmp_path: Path, remote: str) -> None:
    # Seed the remote so it is non-empty, the way a repo from a previous machine is.
    seed = tmp_path / "seed"
    GitRepo.init(seed, branch="main")
    (seed / "README.md").write_text("hello")
    GitRepo.at(seed).commit("seed")
    GitRepo.at(seed).set_remote(remote)
    GitRepo.at(seed).push("main", set_upstream=True)

    calls: list[str] = []
    _, _, clone = await do_init(tmp_path, remote, exists=True, empty=False, calls=calls)

    assert "POST /user/repos" not in calls
    assert (clone / SCHEMA_PATH).exists()
    assert "hello" in (clone / "README.md").read_text(), "the existing README survives"


# -- acceptance: running it twice --------------------------------------------


async def test_second_run_does_not_clobber_the_corpus(tmp_path: Path, remote: str) -> None:
    _, config, clone = await do_init(tmp_path, remote)

    # A memory arrives between runs, plus a hand edit to a bootstrapped file.
    memory = clone / "memory/people/jane.md"
    memory.write_text("---\nid: mem_01\n---\n\nJane runs the deploy pipeline.\n")
    (clone / "README.md").write_text("hand-edited\n")
    GitRepo.at(clone).commit("memory: add jane", paths=["memory/people/jane.md", "README.md"])
    before = GitRepo.at(clone).head()

    prompter = ScriptedPrompter({"memory repo": "someone/kasa-memory", "clone path": str(clone)})
    await do_init(tmp_path, remote, prompter=prompter, exists=True, empty=False)

    assert memory.read_text().startswith("---\nid: mem_01")
    assert (clone / "README.md").read_text() == "hand-edited\n"
    assert GitRepo.at(clone).head() == before, "a no-op run should not commit"
    assert load_config(config).ltm.repo == "someone/kasa-memory"


async def test_second_run_leaves_uncommitted_work_alone(tmp_path: Path, remote: str) -> None:
    _, _, clone = await do_init(tmp_path, remote)
    draft = clone / "memory/facts/draft.md"
    draft.write_text("a half-written thought\n")

    prompter = ScriptedPrompter({"memory repo": "someone/kasa-memory", "clone path": str(clone)})
    await do_init(tmp_path, remote, prompter=prompter, exists=True, empty=False)

    assert draft.exists()
    assert "draft.md" in GitRepo.at(clone).run("status", "--porcelain"), "still uncommitted"


async def test_changing_the_repo_requires_confirmation(tmp_path: Path, remote: str) -> None:
    _, config, clone = await do_init(tmp_path, remote)

    declining = ScriptedPrompter(
        {"memory repo": "someone/other-repo", "clone path": str(clone)},
        confirms={"instead": False},
    )
    await do_init(tmp_path, remote, prompter=declining, exists=True, empty=False)

    assert load_config(config).ltm.repo == "someone/kasa-memory"
    assert any("replaces the configured repo" in w for w in declining.warned)


# -- acceptance: no secrets in the file --------------------------------------


async def test_config_file_holds_env_var_names_not_secrets(
    tmp_path: Path, remote: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "ghp_supersecret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret")

    _, config, _ = await do_init(tmp_path, remote)
    written = config.read_text()

    assert "supersecret" not in written
    assert "ANTHROPIC_API_KEY" in written
    assert TOKEN_ENV in written
    assert config.stat().st_mode & 0o077 == 0, "config should not be group- or world-readable"


async def test_the_token_never_lands_in_git_config(
    tmp_path: Path, remote: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "ghp_supersecret")
    _, _, clone = await do_init(tmp_path, remote)
    assert "supersecret" not in (clone / ".git" / "config").read_text()


# -- refusals ----------------------------------------------------------------


async def test_public_repo_is_refused(tmp_path: Path, remote: str) -> None:
    with pytest.raises(ConfigError, match="public repository"):
        await do_init(tmp_path, remote, exists=True, private=False)


async def test_public_repo_is_refused_before_anything_is_written(
    tmp_path: Path, remote: str
) -> None:
    with pytest.raises(ConfigError):
        await do_init(tmp_path, remote, exists=True, private=False)
    assert not (tmp_path / "config.toml").exists()
    assert not (tmp_path / "ltm").exists()


async def test_owner_name_without_a_token_is_a_clear_error(tmp_path: Path) -> None:
    prompter = ScriptedPrompter({"memory repo": "someone/kasa-memory"})
    with pytest.raises(ConfigError, match=TOKEN_ENV):
        await run_init(prompter, path=tmp_path / "config.toml")


async def test_declining_to_create_the_repo_aborts(tmp_path: Path, remote: str) -> None:
    prompter = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(tmp_path / "ltm")},
        confirms={"Create": False},
    )
    with pytest.raises(ConfigError, match="creating it was declined"):
        await do_init(tmp_path, remote, prompter=prompter)


async def test_a_plain_git_url_skips_github_but_warns(tmp_path: Path, remote: str) -> None:
    """A URL Kasa cannot ask GitHub about is allowed, loudly."""
    seed = tmp_path / "seed"
    GitRepo.init(seed, branch="main")
    (seed / "x").write_text("x")
    GitRepo.at(seed).commit("seed")
    GitRepo.at(seed).set_remote(remote)
    GitRepo.at(seed).push("main", set_upstream=True)

    prompter = ScriptedPrompter({"memory repo": remote, "clone path": str(tmp_path / "ltm")})
    await run_init(prompter, path=tmp_path / "config.toml")

    assert any("cannot check that this repo is private" in w for w in prompter.warned)
    assert (tmp_path / "ltm" / SCHEMA_PATH).exists()


# -- what got configured -----------------------------------------------------


async def test_optional_roles_can_be_declined(tmp_path: Path, remote: str) -> None:
    prompter = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(tmp_path / "ltm")},
        confirms={"different models": False, "Slack": False},
    )
    _, config, _ = await do_init(tmp_path, remote, prompter=prompter)

    cfg = load_config(config)
    assert set(cfg.llm) == {"chat"}
    assert not cfg.slack.configured
    assert not any("utility" in question for question in prompter.chosen)
    assert not any("embedding" in question for question in prompter.chosen)


async def test_existing_extra_roles_make_the_advanced_gate_default_to_yes(
    tmp_path: Path, remote: str
) -> None:
    config = tmp_path / "config.toml"
    clone = tmp_path / "ltm"
    config.write_text(
        f'[ltm]\nrepo = "someone/kasa-memory"\nclone_path = "{clone}"\n\n'
        '[llm.chat]\nkind = "anthropic"\nmodel = "chat"\n\n'
        '[llm.utility]\nkind = "openai"\nmodel = "utility"\n\n'
        '[llm.embedding]\nkind = "openai"\nmodel = "embedding"\n'
    )
    prompter = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(clone)},
        confirms={"Slack": False},
    )
    client = fake_github(clone_url=remote)
    try:
        await run_init(prompter, path=config, github=client)
    finally:
        await client.aclose()

    assert set(load_config(config).llm) == {"chat", "utility", "embedding"}


async def test_slack_tokens_are_recorded_by_name(tmp_path: Path, remote: str) -> None:
    prompter = ScriptedPrompter(
        {
            "memory repo": "someone/kasa-memory",
            "clone path": str(tmp_path / "ltm"),
            "channel ids": "C0123,C0456",
        },
        confirms={"Slack": True},
    )
    _, config, _ = await do_init(tmp_path, remote, prompter=prompter)

    slack = load_config(config).slack
    assert slack.app_token_env == "SLACK_APP_TOKEN"
    assert slack.allowed_channels == ["C0123", "C0456"]


async def test_unset_key_env_is_warned_about_not_fatal(tmp_path: Path, remote: str) -> None:
    prompter, config, _ = await do_init(tmp_path, remote)
    assert any("ANTHROPIC_API_KEY is not set" in w for w in prompter.warned)
    assert config.exists()


async def test_faucet_preset_discovers_and_writes_plain_openai_config(
    tmp_path: Path, remote: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAUCET_API_KEY", "faucet-secret")
    prompter = ScriptedPrompter(
        {
            "memory repo": "someone/kasa-memory",
            "clone path": str(tmp_path / "ltm"),
            "preset for chat": "faucet",
            "model for chat": "anthropic/claude-opus-5",
        },
        confirms={"utility model": False, "embedding model": False, "Slack": False},
    )
    discovered: list[tuple[str, str | None, str]] = []

    async def models(kind: ProviderKind, base_url: str | None, key_env: str) -> list[str]:
        discovered.append((kind, base_url, key_env))
        return ["openai/gpt-5.2", "anthropic/claude-opus-5"]

    client = fake_github(clone_url=remote)
    try:
        await run_init(
            prompter,
            path=tmp_path / "config.toml",
            github=client,
            model_discovery=models,
        )
    finally:
        await client.aclose()

    chat = load_config(tmp_path / "config.toml").llm["chat"]
    assert (chat.kind, chat.base_url, chat.key_env, chat.model) == (
        "openai",
        FAUCET_BASE_URL,
        "FAUCET_API_KEY",
        "anthropic/claude-opus-5",
    )
    assert discovered[0] == ("openai", FAUCET_BASE_URL, "FAUCET_API_KEY")
    assert not any("Base URL for chat" in question for question in prompter.asked)
    assert not any("chat API key" in question for question in prompter.asked)


async def test_model_discovery_is_optional_and_typed_names_remain_open(
    tmp_path: Path, remote: str
) -> None:
    typed = "vendor/a-model-not-in-the-list"
    prompter = ScriptedPrompter(
        {
            "memory repo": "someone/kasa-memory",
            "clone path": str(tmp_path / "ltm"),
            "preset for chat": "faucet",
            "model for chat": typed,
        },
        confirms={"utility model": False, "embedding model": False, "Slack": False},
    )

    async def listed(kind: ProviderKind, base_url: str | None, key_env: str) -> list[str]:
        return ["openai/gpt-5.2"]

    client = fake_github(clone_url=remote)
    try:
        await run_init(
            prompter,
            path=tmp_path / "config.toml",
            github=client,
            model_discovery=listed,
        )
    finally:
        await client.aclose()

    assert load_config(tmp_path / "config.toml").llm["chat"].model == typed


async def test_a_relative_clone_answer_is_stored_absolutely(
    tmp_path: Path, remote: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#88. `init` accepted a relative answer verbatim, created the clone next
    to wherever it was run, and wrote a path that later resolved somewhere
    else. Both halves have to agree, and the config file is the anchor."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    prompter = ScriptedPrompter({"memory repo": "someone/kasa-memory", "clone path": "ltm-here"})
    _, config, _ = await do_init(tmp_path, remote, prompter=prompter)

    cfg = load_config(config)
    assert Path(cfg.ltm.clone_path or "").is_absolute()
    assert cfg.ltm.resolved_clone_path() == tmp_path / "ltm-here"
    assert (tmp_path / "ltm-here" / SCHEMA_PATH).exists(), "and that is where it cloned"
    assert not (elsewhere / "ltm-here").exists(), "not next to the shell it was run from"


# -- web search --------------------------------------------------------------


async def test_web_search_is_off_unless_it_is_asked_for(tmp_path: Path, remote: str) -> None:
    """Reading the open web is a decision about what may enter Kasa's prompts,
    so it is made on purpose rather than arrived at by pressing enter."""
    prompter = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(tmp_path / "ltm")},
        confirms={"web search": False},
    )
    _, config, _ = await do_init(tmp_path, remote, prompter=prompter)

    assert not load_config(config).search.configured


async def test_web_search_records_the_key_by_name(tmp_path: Path, remote: str) -> None:
    prompter = ScriptedPrompter(
        {
            "memory repo": "someone/kasa-memory",
            "clone path": str(tmp_path / "ltm"),
            "Brave Search API key": "KASA_BRAVE",
        },
        confirms={"web search": True},
    )
    _, config, _ = await do_init(tmp_path, remote, prompter=prompter)

    search = load_config(config).search
    assert search.kind == "brave"
    assert search.key_env == "KASA_BRAVE"


async def test_the_search_step_says_what_it_lets_in(tmp_path: Path, remote: str) -> None:
    prompter = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(tmp_path / "ltm")},
        confirms={"web search": False},
    )
    await do_init(tmp_path, remote, prompter=prompter)

    assert any("open web" in said for said in prompter.said)


async def test_turning_search_off_again_clears_the_section(tmp_path: Path, remote: str) -> None:
    """Declining on a re-run must remove it, not leave a stale key behind."""
    enabling = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(tmp_path / "ltm")},
        confirms={"web search": True},
    )
    _, config, _ = await do_init(tmp_path, remote, prompter=enabling)
    assert load_config(config).search.configured

    disabling = ScriptedPrompter(
        {"memory repo": "someone/kasa-memory", "clone path": str(tmp_path / "ltm")},
        confirms={"web search": False},
    )
    await do_init(tmp_path, remote, prompter=disabling)

    assert not load_config(config).search.configured

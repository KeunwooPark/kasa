from __future__ import annotations

from pathlib import Path

import pytest

from kasa.config import Config, config_from_env, load_config, write_config
from kasa.errors import ConfigError
from kasa.llm.registry import ModelRole


def test_env_only_config_needs_just_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """First run should be `export ANTHROPIC_API_KEY=... && kasa run`."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("KASA_CHAT_MODEL", raising=False)

    cfg = config_from_env()
    assert cfg.llm["chat"].kind == "anthropic"


def test_no_key_still_yields_a_usable_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """`kasa db migrate` and `kasa cost` must work before any model is set up."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = config_from_env()
    assert cfg.llm == {}
    assert cfg.store.resolved()  # resolvable without a provider

    # The complaint arrives only when something actually needs a model.
    with pytest.raises(ConfigError, match="chat"):
        cfg.chains()


def test_missing_key_env_is_reported_with_the_variable_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MY_KEY", raising=False)
    cfg = Config.model_validate(
        {"llm": {"chat": {"kind": "openai", "model": "m", "key_env": "MY_KEY"}}}
    )
    with pytest.raises(ConfigError, match="MY_KEY"):
        cfg.llm["chat"].api_key()


def test_utility_falls_back_to_the_chat_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """One key should be enough to get started; #28 makes utility cheap later."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    cfg = Config.model_validate({"llm": {"chat": {"kind": "anthropic", "model": "m"}}})

    chains = cfg.chains()
    assert chains[ModelRole.UTILITY] == chains[ModelRole.CHAT]


def test_chat_role_is_required() -> None:
    with pytest.raises(ConfigError, match="chat"):
        Config().chains()


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic's own error
        Config.model_validate({"llm": {}, "nonsense": 1})


def test_toml_is_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm.chat]
kind = "anthropic"
model = "test-model"

[agent]
max_tool_iterations = 3

[pricing."test-model"]
input = 1.0
output = 2.0
"""
    )
    cfg = load_config(path)

    assert cfg.llm["chat"].model == "test-model"
    assert cfg.agent_config().max_tool_iterations == 3
    assert cfg.price_book().lookup("test-model") is not None


def test_malformed_toml_reports_the_path(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("this is not = = toml")
    with pytest.raises(ConfigError, match="valid TOML"):
        load_config(path)


def test_config_dump_contains_no_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """`kasa config` output should be safe to paste into an issue."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret")
    cfg = config_from_env()
    assert "super-secret" not in str(cfg.redacted())


# -- writing -----------------------------------------------------------------


def _full_config() -> Config:
    return Config.model_validate(
        {
            "ltm": {
                "repo": "someone/kasa-memory",
                "clone_path": "~/.kasa/ltm",
                "branch": "trunk",
                "token_env": "GH_TOKEN",
                "supervised": ["forget", "reorganize"],
            },
            "slack": {
                "app_token_env": "SLACK_APP_TOKEN",
                "bot_token_env": "SLACK_BOT_TOKEN",
                "allowed_channels": ["C0123", "C0456"],
            },
            "llm": {
                "chat": {
                    "kind": "anthropic",
                    "model": "claude-opus-5",
                    "key_env": "ANTHROPIC_API_KEY",
                    "fallbacks": [{"kind": "openai", "model": "gpt-4o-mini"}],
                },
                "embedding": {
                    "kind": "openai",
                    "model": "text-embedding-3-small",
                    "base_url": "https://api.openai.com/v1",
                    "embedding_dimensions": 1536,
                },
            },
            "agent": {"max_tool_iterations": 3},
            "pricing": {"claude-opus-5": {"input": 3.0, "output": 15.0}},
        }
    )


def test_written_config_round_trips(tmp_path: Path) -> None:
    original = _full_config()
    path = tmp_path / "config.toml"
    write_config(original, path)

    assert load_config(path) == original


def test_written_config_omits_untouched_defaults(tmp_path: Path) -> None:
    """The file should read as the decisions someone made, not a settings dump."""
    path = tmp_path / "config.toml"
    write_config(_full_config(), path)
    written = path.read_text()

    assert "max_tool_iterations = 3" in written
    assert "history_limit" not in written, "left at its default"
    assert "[context]" not in written


def test_written_config_is_owner_readable_only(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_config(Config(), path)
    assert path.stat().st_mode & 0o077 == 0


def test_special_characters_survive_the_round_trip(tmp_path: Path) -> None:
    cfg = Config.model_validate(
        {"llm": {"chat": {"kind": "openai", "model": 'a"quoted\\model', "base_url": "x\ty"}}}
    )
    path = tmp_path / "config.toml"
    write_config(cfg, path)
    assert load_config(path) == cfg


def test_an_empty_config_still_writes_something_loadable(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_config(Config(), path)
    assert load_config(path) == Config()

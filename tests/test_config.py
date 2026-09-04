from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from kasa.config import Config, SearchSettings, config_from_env, load_config, write_config
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


def test_daily_budget_ceiling_is_optional_and_non_negative() -> None:
    assert Config().budget.daily_usd_ceiling is None
    assert (
        Config.model_validate({"budget": {"daily_usd_ceiling": 2.5}}).budget.daily_usd_ceiling
        == 2.5
    )
    with pytest.raises(Exception):  # noqa: B017 - pydantic's own error
        Config.model_validate({"budget": {"daily_usd_ceiling": -1}})


def test_daily_budget_ceiling_round_trips_through_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_config(Config.model_validate({"budget": {"daily_usd_ceiling": 2.5}}), path)

    assert "[budget]" in path.read_text()
    assert load_config(path).budget.daily_usd_ceiling == 2.5


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
            "search": {
                "kind": "brave",
                "key_env": "BRAVE_SEARCH_API_KEY",
                "max_results": 8,
                "cost_per_call_usd": 0.005,
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


# -- the context budget is checked at load time (#76) -------------------------


@pytest.mark.parametrize(
    ("stanza", "expected"),
    [
        ("[context]\nsystem = 0.9\nrecent = 0.9\n", "must sum to 1.0"),
        ("[context]\nretrieved = 0.0\n", "must sum to 1.0"),
        ("[context]\ntotal = 0\n", "must be positive"),
    ],
)
def test_an_unusable_context_budget_is_rejected_when_the_config_is_read(
    tmp_path: Path, stanza: str, expected: str
) -> None:
    """#76. `ContextBudget` validated in `__post_init__`, and nothing built one
    until a command built a packer — so `kasa config` and `kasa doctor` were
    green on a config `kasa run` would not start with."""
    path = tmp_path / "config.toml"
    path.write_text(stanza)

    with pytest.raises(ConfigError) as caught:
        load_config(path)

    assert expected in str(caught.value)
    assert str(path) in str(caught.value), "and it names the file"


def test_the_default_budget_is_still_valid(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[store]\npath = "x.db"\n')
    assert load_config(path).context.to_budget().total == 128_000


# -- a relative path means "next to the config file" (#88) --------------------


def write_relative(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(
        '[ltm]\nrepo = "someone/mem"\nclone_path = "ltm-here"\n\n[store]\npath = "kasa-here.db"\n'
    )
    return path


def test_a_relative_path_is_read_against_the_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#88. Against the working directory instead, the same config file meant a
    different memory repo and a different database depending on where Kasa was
    started — and the second one bootstrapped an empty world and called it
    healthy."""
    config = write_relative(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    monkeypatch.chdir(elsewhere)
    cfg = load_config(config)

    assert cfg.store.resolved() == tmp_path / "kasa-here.db"
    assert cfg.ltm.resolved_clone_path() == tmp_path / "ltm-here"


def test_it_does_not_matter_where_kasa_was_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = write_relative(tmp_path)
    seen = set()
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.chdir(directory)
        seen.add(load_config(config).store.resolved())

    assert len(seen) == 1
    # A relative path is also "the same" from everywhere, and means something
    # different at each one. Absolute is the property that makes it true.
    assert seen.pop() == tmp_path / "kasa-here.db"


def test_a_relative_config_argument_still_anchors_absolutely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--config config.toml` from the directory holding it. The anchor has to
    be absolute, or joining onto it leaves a relative path behind."""
    write_relative(tmp_path)
    monkeypatch.chdir(tmp_path)

    resolved = load_config(Path("config.toml")).store.resolved()

    assert resolved.is_absolute()
    assert resolved == tmp_path / "kasa-here.db"


@pytest.mark.parametrize("value", ["~/.kasa/ltm", "/var/lib/kasa/ltm"])
def test_a_path_that_is_already_absolute_is_left_exactly_as_written(
    tmp_path: Path, value: str
) -> None:
    """`~` included: it is unambiguous as it stands, and rewriting it would mean
    `kasa init` could not round-trip the file it just wrote."""
    path = tmp_path / "config.toml"
    path.write_text(f'[ltm]\nrepo = "someone/mem"\nclone_path = "{value}"\n')

    assert load_config(path).ltm.clone_path == value


# -- web search --------------------------------------------------------------


def test_search_is_off_until_a_kind_is_set() -> None:
    """Absent configuration must mean *no tool*, not a tool that fails on use:
    a model told it can search spends a turn discovering that it cannot."""
    assert not Config().search.configured
    assert Config(search=SearchSettings(kind="brave")).search.configured


def test_an_unconfigured_search_writes_no_section(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    write_config(Config(), path)

    assert "[search]" not in path.read_text()


def test_a_configured_search_survives_the_round_trip(tmp_path: Path) -> None:
    cfg = Config(search=SearchSettings(kind="brave", key_env="KASA_BRAVE", max_results=3))
    path = tmp_path / "config.toml"
    write_config(cfg, path)

    assert load_config(path) == cfg


def test_a_search_key_is_never_written_into_the_config(tmp_path: Path) -> None:
    """Only the name of the variable, as with every other credential."""
    path = tmp_path / "config.toml"
    write_config(Config(search=SearchSettings(kind="brave", key_env="KASA_BRAVE")), path)

    written = path.read_text()
    assert "KASA_BRAVE" in written
    assert "cost_per_call_usd" in written, "written in full, so the price is visible to edit"


def test_a_search_asking_for_more_results_than_the_tool_allows_is_refused() -> None:
    with pytest.raises(ValidationError):
        SearchSettings(kind="brave", max_results=50)

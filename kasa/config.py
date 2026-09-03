"""Configuration loading and object construction.

Secrets are never stored in the config file — only the *name* of the environment
variable holding them. `kasa init` writes this file; with no file present a
usable config is synthesized from the environment, so a first run needs nothing
but an API key exported.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kasa.core.agent import AgentConfig
from kasa.core.context import ContextBudget
from kasa.errors import ConfigError
from kasa.llm.anthropic_compat import AnthropicCompatProvider
from kasa.llm.base import LLMProvider
from kasa.llm.cost import CostMeter, Price, PriceBook
from kasa.llm.openai_compat import OpenAICompatProvider
from kasa.llm.registry import ModelRole, ProviderRegistry, RetryPolicy
from kasa.store import Store

ProviderKind = Literal["openai", "anthropic"]

NO_CHAT_PROVIDER = (
    "no model configured for the 'chat' role.\n"
    "Export ANTHROPIC_API_KEY or OPENAI_API_KEY, or write a config file "
    "(see `kasa config` for where it is looked for)."
)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_CLONE_PATH = "~/.kasa/ltm"


def config_path() -> Path:
    if override := os.environ.get("KASA_CONFIG"):
        return Path(override).expanduser()
    return Path(user_config_dir("kasa")) / "config.toml"


def default_db_path() -> Path:
    if override := os.environ.get("KASA_DB"):
        return Path(override).expanduser()
    return Path(user_data_dir("kasa")) / "kasa.db"


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: ProviderKind
    model: str
    base_url: str | None = None
    key_env: str | None = None
    embedding_dimensions: int | None = None
    fallbacks: list[ProviderConfig] = Field(default_factory=list)

    def api_key(self) -> str:
        env = self.key_env or default_key_env(self.kind)
        key = os.environ.get(env)
        if not key:
            raise ConfigError(
                f"{env} is not set (needed for the {self.kind} provider {self.model!r})"
            )
        return key

    def build(self) -> LLMProvider:
        if self.kind == "anthropic":
            return AnthropicCompatProvider(
                model=self.model,
                api_key=self.api_key(),
                base_url=self.base_url or "https://api.anthropic.com/v1",
            )
        return OpenAICompatProvider(
            model=self.model,
            api_key=self.api_key(),
            base_url=self.base_url or "https://api.openai.com/v1",
            embedding_dimensions=self.embedding_dimensions,
        )

    def chain(self) -> list[LLMProvider]:
        return [self.build(), *(f.build() for f in self.fallbacks)]


class LTMSettings(BaseModel):
    """The private GitHub repository that holds long-term memory."""

    model_config = ConfigDict(extra="forbid")

    #: `owner/name`, or any URL git can clone. None until `kasa init` has run.
    repo: str | None = None
    clone_path: str | None = None
    branch: str = "main"
    token_env: str = "KASA_GITHUB_TOKEN"
    #: Jobs that open a pull request instead of pushing to the branch. Defaults
    #: to the destructive one: `forget` is the job you want to read before it
    #: lands, and `promote` is the one you would get tired of approving.
    supervised: list[str] = Field(default_factory=lambda: ["forget"])

    @property
    def configured(self) -> bool:
        return bool(self.repo)

    def resolved_clone_path(self) -> Path:
        return Path(self.clone_path or DEFAULT_CLONE_PATH).expanduser()

    def token(self) -> str | None:
        return os.environ.get(self.token_env) or None


class SlackSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app_token_env: str | None = None  # xapp-, Socket Mode
    bot_token_env: str | None = None  # xoxb-
    allowed_channels: list[str] = Field(default_factory=list)

    @property
    def configured(self) -> bool:
        return bool(self.app_token_env and self.bot_token_env)


class MemorySettings(BaseModel):
    """Limits on what a consolidation job is allowed to do in one commit.

    These are the numbers the patch validator enforces. They exist because the
    plan on the other side of them was written by a model reading text that
    someone else typed.
    """

    model_config = ConfigDict(extra="forbid")

    #: A memory is a paragraph or two. Anything near this is a transcript that
    #: escaped, or a prompt-injection payload.
    max_file_bytes: int = 65_536
    #: One job touching more than this is a runaway, not a consolidation.
    max_files_per_commit: int = 25
    #: How long an archived memory must sit before it can be deleted at all.
    retention_floor_days: int = 30


class AgentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system_prompt: str | None = None
    max_tool_iterations: int = 8
    max_tokens: int = 4096
    temperature: float | None = None
    history_limit: int = 200


class ContextSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int = 128_000
    system: float = 0.05
    pinned: float = 0.10
    retrieved: float = 0.30
    episode: float = 0.10
    recent: float = 0.35
    headroom: float = 0.10

    @model_validator(mode="after")
    def _budget_is_constructible(self) -> ContextSettings:
        """Fail at load time, not at the first turn.

        `ContextBudget` validates the shares in `__post_init__`, and nothing
        constructed one until a command built a packer — so a config whose
        shares did not sum to 1.0 passed `kasa config` and `kasa doctor` and
        then stopped `kasa run` from starting (#76). A health check that is
        green on a config the program refuses is worse than no health check.
        """
        self.to_budget()
        return self

    def to_budget(self) -> ContextBudget:
        return ContextBudget(**self.model_dump())

    def tokens_for_retrieval(self) -> int:
        """The share retrieval may fill. Retrieval truncates to this itself, so
        that what `kasa why` reports is what the packer would have received."""
        return self.to_budget().tokens_for(self.retrieved)


class StoreSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str | None = None

    def resolved(self) -> Path:
        return Path(self.path).expanduser() if self.path else default_db_path()


class PriceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class RetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 20.0
    jitter: float = 0.25

    def to_policy(self) -> RetryPolicy:
        return RetryPolicy(**self.model_dump())


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ltm: LTMSettings = Field(default_factory=LTMSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    llm: dict[str, ProviderConfig] = Field(default_factory=dict)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    context: ContextSettings = Field(default_factory=ContextSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    #: USD per million tokens, keyed by model-name prefix. Empty by default:
    #: a stale built-in price table is worse than an absent one.
    pricing: dict[str, PriceSettings] = Field(default_factory=dict)

    def agent_config(self) -> AgentConfig:
        settings = self.agent
        base = AgentConfig()
        return AgentConfig(
            system_prompt=settings.system_prompt or base.system_prompt,
            max_tool_iterations=settings.max_tool_iterations,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            history_limit=settings.history_limit,
        )

    def price_book(self) -> PriceBook:
        return PriceBook({name: Price(**p.model_dump()) for name, p in self.pricing.items()})

    def chains(self) -> dict[ModelRole, list[LLMProvider]]:
        chains: dict[ModelRole, list[LLMProvider]] = {}
        for role in ModelRole:
            if entry := self.llm.get(role.value):
                chains[role] = entry.chain()
        if ModelRole.CHAT not in chains:
            raise ConfigError(NO_CHAT_PROVIDER)
        # Utility falls back to the chat model rather than failing: an expensive
        # summary beats no summary, and it keeps first-run setup to one key.
        if ModelRole.UTILITY not in chains:
            chains[ModelRole.UTILITY] = chains[ModelRole.CHAT]
        return chains

    def build_registry(self, *, store: Store | None = None) -> ProviderRegistry:
        meter = CostMeter(
            self.price_book(),
            sink=store.record_call if store is not None else _null_sink,
        )
        return ProviderRegistry(self.chains(), meter=meter, retry=self.retry.to_policy())

    def redacted(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


async def _null_sink(record: Any) -> None:
    return None


def default_key_env(kind: ProviderKind) -> str:
    return "ANTHROPIC_API_KEY" if kind == "anthropic" else "OPENAI_API_KEY"


def anchored(value: str, *, base: Path) -> Path:
    """Resolve a configured path, reading a relative one as relative to `base`.

    `base` is the directory holding config.toml. Resolved against the process's
    working directory instead — which is what `expanduser()` alone leaves —
    a relative path in a config file means a different memory repo and a
    different database depending on where Kasa was started, and neither
    Kasa nor the operator is told: the second directory gets a freshly
    bootstrapped, empty world reported as healthy (#88).

    Relative to the config file is the reading that gives the value a stable
    meaning, and it is what makes a config directory portable.
    """
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: Path | None = None) -> Config:
    target = path or config_path()
    if not target.exists():
        return config_from_env()
    try:
        raw = tomllib.loads(target.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{target} is not valid TOML: {exc}") from exc
    try:
        cfg = Config.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{target} is not a valid Kasa config: {exc}") from exc
    return anchor_paths(cfg, base=target.expanduser().resolve().parent)


def anchor_paths(cfg: Config, *, base: Path) -> Config:
    """Rewrite relative configured paths to be absolute. See `anchored`.

    Done once here rather than inside each `resolved()`, so that everything
    downstream — including `kasa config`, which prints the config back — sees
    the path that will actually be used.

    A value that is already absolute is left exactly as written, `~` included:
    it is unambiguous as it stands, and rewriting it would mean `kasa init`
    could not round-trip the file it just wrote.
    """
    cfg.ltm.clone_path = _anchor(cfg.ltm.clone_path, base)
    cfg.store.path = _anchor(cfg.store.path, base)
    return cfg


def _anchor(value: str | None, base: Path) -> str | None:
    if not value:
        return value
    path = Path(value).expanduser()
    return value if path.is_absolute() else str(base / path)


def config_from_env() -> Config:
    """Synthesize a config from whatever API key is exported.

    Keeps first run to `export ANTHROPIC_API_KEY=... && kasa run`. `kasa init`
    replaces this with a real config file.

    Returns a provider-less config rather than raising when no key is present:
    `kasa db migrate`, `kasa cost` and `kasa config` are all useful before any
    model is configured, and only `chains()` actually needs a provider.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        chat = ProviderConfig(
            kind="anthropic",
            model=os.environ.get("KASA_CHAT_MODEL") or DEFAULT_ANTHROPIC_MODEL,
        )
    elif os.environ.get("OPENAI_API_KEY"):
        chat = ProviderConfig(
            kind="openai",
            model=os.environ.get("KASA_CHAT_MODEL") or DEFAULT_OPENAI_MODEL,
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
    else:
        return Config()
    return Config(llm={"chat": chat})


# -- writing -----------------------------------------------------------------

_HEADER = """# Kasa configuration — written by `kasa init`.
#
# This file holds no secrets. Credentials are referenced by the *name* of the
# environment variable that carries them, so this file is safe to read, diff,
# and back up. See docs/DESIGN.md Appendix A.
"""

_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    text = "".join(_ESCAPES.get(ch, ch) for ch in str(value))
    return f'"{text}"'


def _table(
    name: str,
    model: BaseModel,
    *,
    comment: str = "",
    full: bool = False,
    exclude: set[str] | None = None,
) -> list[str]:
    """Render one table, omitting fields left at their default unless `full`."""
    fields = model.model_dump(
        mode="json", exclude_defaults=not full, exclude_none=True, exclude=exclude
    )
    if not fields:
        return []
    lines = [f"[{name}]"]
    if comment:
        lines.insert(0, comment)
    width = max(len(key) for key in fields)
    lines += [f"{key.ljust(width)} = {_toml_value(value)}" for key, value in fields.items()]
    return [*lines, ""]


def render_toml(cfg: Config) -> str:
    """Serialize a config back to TOML.

    Hand-rolled rather than via a TOML writer so the generated file can carry
    the comments that make it editable by hand — which is the only reason to
    write a config file instead of a pickle.
    """
    lines = [_HEADER]

    if cfg.ltm.configured:
        lines += _table(
            "ltm",
            cfg.ltm,
            comment="# The private GitHub repo holding long-term memory.",
            full=True,
        )

    for role, provider in cfg.llm.items():
        # `fallbacks` is a TOML array of tables, so it is emitted as its own
        # `[[...]]` sections rather than inline in the parent.
        lines += _table(f"llm.{role}", provider, full=True, exclude={"fallbacks"})
        for fallback in provider.fallbacks:
            lines += _table(f"[llm.{role}.fallbacks]", fallback, full=True, exclude={"fallbacks"})

    if cfg.slack.configured:
        lines += _table("slack", cfg.slack, comment="# Socket Mode; see #21.")

    for name, section in (
        ("memory", cfg.memory),
        ("agent", cfg.agent),
        ("context", cfg.context),
        ("store", cfg.store),
        ("retry", cfg.retry),
    ):
        lines += _table(name, section)
    for model, price in cfg.pricing.items():
        lines += _table(f'pricing."{model}"', price, full=True)

    return "\n".join(lines).rstrip() + "\n"


def write_config(cfg: Config, path: Path) -> None:
    """Write the config file with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_toml(cfg))
    path.chmod(0o600)

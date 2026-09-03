"""Configuration loading and object construction.

Secrets are never stored in the config file — only the *name* of the environment
variable holding them. `kasa init` (#10) will write this file; until then a
usable config is synthesized from the environment so v0 runs with nothing but an
API key exported.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from platformdirs import user_config_dir, user_data_dir
from pydantic import BaseModel, ConfigDict, Field

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
        env = self.key_env or _default_key_env(self.kind)
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

    def to_budget(self) -> ContextBudget:
        return ContextBudget(**self.model_dump())


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


def _default_key_env(kind: ProviderKind) -> str:
    return "ANTHROPIC_API_KEY" if kind == "anthropic" else "OPENAI_API_KEY"


def load_config(path: Path | None = None) -> Config:
    target = path or config_path()
    if not target.exists():
        return config_from_env()
    try:
        raw = tomllib.loads(target.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{target} is not valid TOML: {exc}") from exc
    try:
        return Config.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{target} is not a valid Kasa config: {exc}") from exc


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

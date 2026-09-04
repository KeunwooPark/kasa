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
from kasa.memory.salience import Decay
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


class EpisodeSettings(BaseModel):
    """When a stretch of conversation is over, and how much of it to read.

    The boundaries are what `docs/DESIGN.md` §6 draws: a thread that has gone
    quiet, or one that has run long enough to be worth consolidating before it
    goes quiet at all.
    """

    model_config = ConfigDict(extra="forbid")

    #: Quiet for this long and the segment is over. Twenty minutes is the
    #: design's number: long enough that a slow reply does not split a
    #: conversation in half, short enough that today's thread is consolidated
    #: today.
    idle_minutes: int = 20
    #: A long thread is closed on length rather than left to grow. An episode
    #: that never ends is one whose transcript eventually stops fitting in a
    #: context window, and consolidating the first half of a conversation is
    #: better than consolidating none of it.
    max_messages: int = 40
    #: Episodes consolidated per sweep. A bound rather than a target: it is
    #: what stops a backlog — a daemon down for a day — from spending its whole
    #: token budget in one tick.
    max_per_run: int = 20
    #: How much of an episode the extractor reads. Above `max_messages` on
    #: purpose: a session end closes an episode of any length.
    transcript_messages: int = 200
    #: More candidate facts than this out of one conversation is a model
    #: narrating the transcript rather than distilling it.
    max_observations: int = 12
    #: The cost gate (`docs/DESIGN.md` §6.1). An episode scored below this
    #: closes with its summary and is never extracted from, so it never
    #: reaches `promote`. Low, because the thing being traded is tokens
    #: against knowledge, and only one of those can be got back later. `0.0`
    #: turns the gate off — every score clears it.
    signal_threshold: float = 0.3


class PromoteSettings(BaseModel):
    """How much of the pending queue one `promote` run works through.

    Every number here bounds a cost. `promote` calls the *chat* model once per
    subject and writes one commit per run, so an unbounded run is both the
    largest bill in the system and the largest diff a person has to review.
    """

    model_config = ConfigDict(extra="forbid")

    #: Pending observations read per run.
    max_observations: int = 100
    #: Subjects reconciled per run — one chat-model call each.
    max_subjects: int = 10
    #: Existing memories offered per subject as competition. This is what makes
    #: a restated fact an update instead of a duplicate, so it is the number to
    #: raise if duplicates start appearing.
    competing_memories: int = 5
    #: A memory larger than this is not shown as competition. Not truncated:
    #: an `Update` replaces the body wholesale, so a model shown half a memory
    #: would rewrite it as half a memory. One this size is a file `reorganize`
    #: should be splitting.
    max_memory_chars: int = 8_000
    #: Rejected plans a subject may produce before its observations are
    #: discarded. Without it, one poison group costs a chat call every hour
    #: forever and is never promoted anyway.
    max_attempts: int = 3


class ReflectSettings(BaseModel):
    """The nightly pass: the journal, the salience curve, and the digest."""

    model_config = ConfigDict(extra="forbid")

    #: The salience curve. See `kasa/memory/salience.py` — salience is
    #: recomputed from age and recall rather than adjusted, which is what lets
    #: a bounded nightly pass converge instead of double-counting.
    base_salience: float = 0.5
    half_life_days: float = 30.0
    hit_boost: float = 0.08
    max_hit_boost: float = 0.3
    salience_floor: float = 0.05
    #: How far back recalls count. The same order as the half-life: a memory
    #: that was busy last season and quiet since should be fading.
    hit_window_days: int = 30
    #: Below this, a memory's salience is left alone. The commit is a diff a
    #: person reads, and a file rewritten for a change in the third decimal
    #: place is noise in it.
    min_salience_move: float = 0.02
    #: Salience rewrites per run, under the per-commit file cap. On a corpus
    #: larger than one commit, the memories furthest from their true score go
    #: first and the rest converge over the following nights.
    max_salience_updates: int = 20
    #: How many recently-touched memories are read together when looking for
    #: contradictions. One prompt, so this is a token budget as much as a
    #: coverage decision.
    max_conflict_candidates: int = 12
    #: Contradictions surfaced per night. More than a handful is a corpus with
    #: a structural problem, and a list nobody reads to the end.
    max_conflicts: int = 5
    journal_tokens: int = 800
    #: Slack channel id for the nightly digest. None posts nothing.
    digest_channel: str | None = None

    def decay(self) -> Decay:
        return Decay(
            base=self.base_salience,
            half_life_days=self.half_life_days,
            per_hit=self.hit_boost,
            max_boost=self.max_hit_boost,
            floor=self.salience_floor,
        )


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
    episodes: EpisodeSettings = Field(default_factory=EpisodeSettings)
    promote: PromoteSettings = Field(default_factory=PromoteSettings)
    reflect: ReflectSettings = Field(default_factory=ReflectSettings)
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
        lines += _table(
            "slack",
            cfg.slack,
            comment="# Socket Mode: the connection is outbound, so there is no ingress.",
        )

    for name, section in (
        ("memory", cfg.memory),
        ("episodes", cfg.episodes),
        ("promote", cfg.promote),
        ("reflect", cfg.reflect),
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

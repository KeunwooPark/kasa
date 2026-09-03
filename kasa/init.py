"""`kasa init` — interactive first-run setup.

Three things have to be true before Kasa can run: there is a private git repo to
put memories in, there is a local clone of it, and there is a model to talk to.
This walks through all three and writes the answers to `config.toml`.

Every step is re-entrant. `init` is run again after a token rotates, on a second
machine, or by someone who forgot they had run it, so nothing here overwrites an
answer without being told to, and nothing touches the contents of a memory repo
that already has memories in it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kasa.config import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_CLONE_PATH,
    DEFAULT_OPENAI_MODEL,
    Config,
    LTMSettings,
    ProviderConfig,
    ProviderKind,
    SlackSettings,
    config_path,
    default_key_env,
    load_config,
    write_config,
)
from kasa.errors import ConfigError, GitError
from kasa.github import GitHubClient, RepoInfo, is_full_name
from kasa.memory.bootstrap import bootstrap, is_bootstrapped, refresh_schema
from kasa.memory.gitcmd import GitRepo, git_available
from kasa.memory.layout import SCHEMA_PATH

_OPTIONAL_ROLES = {
    "utility": "summaries and consolidation; falls back to the chat model",
    "embedding": "semantic retrieval; lexical search works without it",
}

PUBLIC_REPO_REFUSED = (
    "{name} is a public repository.\n"
    "Long-term memory accumulates whatever the agent is told, including the "
    "contents of DMs and private channels, so Kasa will not write it to a repo "
    "the world can read. Make it private, or choose another repo."
)

NO_TOKEN = (
    "{env} is not set, so {name} cannot be looked up or created on GitHub.\n"
    "Export a fine-grained token with `contents: write` on that repo and run "
    "`kasa init` again, or enter a git URL instead of an owner/name."
)


class Prompter(Protocol):
    """Where the answers come from. Scripted in tests, a terminal in life."""

    def ask(self, question: str, *, default: str | None = None) -> str: ...
    def choose(self, question: str, choices: tuple[str, ...], *, default: str) -> str: ...
    def confirm(self, question: str, *, default: bool = False) -> bool: ...
    def say(self, text: str) -> None: ...
    def warn(self, text: str) -> None: ...


@dataclass
class InitResult:
    config_path: Path
    repo_path: Path
    repo_url: str
    created_repo: bool = False
    cloned: bool = False
    bootstrapped: list[str] = field(default_factory=list)
    pushed: bool = False


async def run_init(
    prompter: Prompter,
    *,
    path: Path | None = None,
    github: GitHubClient | None = None,
) -> InitResult:
    """Walk the setup and write the config file. Returns what it did."""
    if not git_available():
        raise ConfigError("git is not on PATH, and Kasa stores long-term memory in a git repo")

    target = path or config_path()
    cfg = load_config(target) if target.exists() else Config()
    if target.exists():
        prompter.say(f"Updating the existing config at {target}.")
    else:
        prompter.say(f"Setting up Kasa. This writes {target}, and no secrets go in it.")

    ltm, info = await _configure_repo(prompter, cfg.ltm, github=github)
    repo, result = _prepare_clone(prompter, ltm, info, config_file=target)

    # Bound to names first: as keyword arguments these would be evaluated in
    # source order, which asks about the optional Slack surface before the
    # required chat model.
    llm = _configure_providers(prompter, cfg.llm)
    slack = _configure_slack(prompter, cfg.slack)

    cfg = Config(
        ltm=ltm,
        slack=slack,
        llm=llm,
        agent=cfg.agent,
        context=cfg.context,
        store=cfg.store,
        retry=cfg.retry,
        pricing=cfg.pricing,
    )
    write_config(cfg, target)
    prompter.say(f"Wrote {target}.")

    result.pushed = _publish(prompter, repo, ltm)
    return result


# -- the repository ----------------------------------------------------------


async def _configure_repo(
    prompter: Prompter, current: LTMSettings, *, github: GitHubClient | None
) -> tuple[LTMSettings, RepoInfo | None]:
    spec = prompter.ask(
        "Long-term memory repo (owner/name, or a git URL)", default=current.repo
    ).strip()
    if not spec:
        raise ConfigError("a long-term memory repository is required")
    if current.repo and spec != current.repo:
        prompter.warn(f"This replaces the configured repo {current.repo}.")
        if not prompter.confirm(f"Point Kasa at {spec} instead?"):
            spec = current.repo

    token_env = prompter.ask("Env var holding the GitHub token", default=current.token_env).strip()
    settings = LTMSettings(
        repo=spec,
        clone_path=current.clone_path,
        branch=current.branch,
        token_env=token_env,
        supervised=current.supervised,
    )

    info = await _resolve_on_github(prompter, spec, settings, github=github)
    if info is not None:
        settings = settings.model_copy(update={"branch": info.default_branch})
        if not info.can_push:
            prompter.warn(f"The token cannot push to {info.full_name}; memory will stay local.")

    clone_path = prompter.ask(
        "Local clone path", default=str(settings.clone_path or DEFAULT_CLONE_PATH)
    ).strip()
    return settings.model_copy(update={"clone_path": clone_path or DEFAULT_CLONE_PATH}), info


async def _resolve_on_github(
    prompter: Prompter, spec: str, settings: LTMSettings, *, github: GitHubClient | None
) -> RepoInfo | None:
    """Look the repo up, offering to create it. None for a plain git URL."""
    if not is_full_name(spec):
        prompter.warn(
            "Kasa cannot check that this repo is private, because it is not an "
            "owner/name on GitHub. Make sure it is."
        )
        return None

    token = os.environ.get(settings.token_env)
    if not token and github is None:
        raise ConfigError(NO_TOKEN.format(env=settings.token_env, name=spec))

    client = github or GitHubClient(token or "")
    try:
        info = await client.get_repo(spec)
        if info is None:
            prompter.say(f"{spec} does not exist, or the token cannot see it.")
            if not prompter.confirm(f"Create {spec} as a private repository?", default=True):
                raise ConfigError(f"{spec} is not available, and creating it was declined")
            info = await client.create_repo(spec, private=True, description="Kasa long-term memory")
            prompter.say(f"Created {info.html_url} (private).")
    finally:
        if github is None:
            await client.aclose()

    # #18 hardens this into a startup check as well; the refusal starts here,
    # because this is the last moment before secrets begin accumulating.
    if not info.private:
        raise ConfigError(PUBLIC_REPO_REFUSED.format(name=info.full_name))
    return info


def _prepare_clone(
    prompter: Prompter, ltm: LTMSettings, info: RepoInfo | None, *, config_file: Path
) -> tuple[GitRepo, InitResult]:
    url = info.clone_url if info else str(ltm.repo)
    destination = ltm.resolved_clone_path()
    result = InitResult(
        config_path=config_file,
        repo_path=destination,
        repo_url=url,
        created_repo=bool(info and info.empty),
    )

    repo = GitRepo.at(destination, token=ltm.token())
    if repo.exists:
        prompter.say(f"Reusing the clone at {destination}.")
        if (existing := repo.remote_url()) and existing != url:
            prompter.warn(f"Its origin is {existing}, not {url}.")
            if prompter.confirm("Repoint origin?", default=False):
                repo.set_remote(url)
    elif info is None or not info.empty:
        repo = GitRepo.clone(url, destination, branch=ltm.branch, token=ltm.token())
        result.cloned = True
        prompter.say(f"Cloned into {destination}.")
    else:
        # Cloning an empty repo works but leaves a working copy with no branch;
        # initializing one locally and pushing is the cleaner shape.
        repo = GitRepo.init(destination, branch=ltm.branch, token=ltm.token())
        repo.set_remote(url)
        prompter.say(f"Initialized a new working copy at {destination}.")

    if is_bootstrapped(destination):
        prompter.say("The repo already has a memory skeleton; leaving its contents alone.")
        result.bootstrapped = [SCHEMA_PATH] if refresh_schema(destination) else []
        if result.bootstrapped:
            prompter.say("Updated .kasa/schema.md to this version's contract.")
    else:
        result.bootstrapped = bootstrap(destination)
        prompter.say(f"Bootstrapped {len(result.bootstrapped)} files.")

    # Only the paths this run wrote get staged. A user may have edits in flight
    # in their own memory repo, and `init` is not entitled to commit them.
    if result.bootstrapped:
        repo.commit(
            "memory: bootstrap long-term memory store\n\nKasa-Job: init",
            paths=result.bootstrapped,
        )
    return repo, result


def _publish(prompter: Prompter, repo: GitRepo, ltm: LTMSettings) -> bool:
    if not repo.remote_url() or not repo.has_commits():
        return False
    if not prompter.confirm(f"Push to {ltm.repo}?", default=True):
        return False
    try:
        repo.push(ltm.branch, set_upstream=True)
    except GitError as exc:
        # Not fatal: the memory repo is usable offline, and the next job that
        # writes to it will push then.
        prompter.warn(f"Could not push yet — {exc}")
        return False
    prompter.say(f"Pushed {ltm.branch}.")
    return True


# -- models and surfaces -----------------------------------------------------


def _configure_providers(
    prompter: Prompter, current: dict[str, ProviderConfig]
) -> dict[str, ProviderConfig]:
    providers = {"chat": _configure_role(prompter, "chat", current.get("chat"))}
    for role, hint in _OPTIONAL_ROLES.items():
        prompter.say(f"The {role} model is optional — {hint}.")
        if prompter.confirm(f"Configure the {role} model?", default=role in current):
            providers[role] = _configure_role(prompter, role, current.get(role))
    return providers


def _configure_role(
    prompter: Prompter, role: str, current: ProviderConfig | None
) -> ProviderConfig:
    kind = prompter.choose(
        f"Provider kind for {role}",
        ("anthropic", "openai"),
        default=current.kind if current else "anthropic",
    )
    provider_kind: ProviderKind = "anthropic" if kind == "anthropic" else "openai"
    default_model = (
        current.model
        if current
        else (DEFAULT_ANTHROPIC_MODEL if provider_kind == "anthropic" else DEFAULT_OPENAI_MODEL)
    )
    model = prompter.ask(f"Model for {role}", default=default_model).strip()
    base_url = prompter.ask(
        f"Base URL for {role} (blank for the provider default)",
        default=current.base_url if current else "",
    ).strip()
    default_env = current.key_env if current and current.key_env else default_key_env(provider_kind)
    key_env = prompter.ask(f"Env var holding the {role} API key", default=default_env).strip()
    if not os.environ.get(key_env):
        prompter.warn(f"{key_env} is not set in this shell. Export it before `kasa run`.")

    return ProviderConfig(
        kind=provider_kind,
        model=model,
        base_url=base_url or None,
        key_env=key_env,
        embedding_dimensions=current.embedding_dimensions if current else None,
    )


def _configure_slack(prompter: Prompter, current: SlackSettings) -> SlackSettings:
    prompter.say("Slack arrives in v2; the tokens can be filled in now or later.")
    if not prompter.confirm("Configure Slack?", default=current.configured):
        return current

    app_token_env = prompter.ask(
        "Env var holding the Slack app token (xapp-)",
        default=current.app_token_env or "SLACK_APP_TOKEN",
    ).strip()
    bot_token_env = prompter.ask(
        "Env var holding the Slack bot token (xoxb-)",
        default=current.bot_token_env or "SLACK_BOT_TOKEN",
    ).strip()
    channels = prompter.ask(
        "Allowed channel ids, comma-separated (blank for all)",
        default=",".join(current.allowed_channels),
    )
    return SlackSettings(
        app_token_env=app_token_env or None,
        bot_token_env=bot_token_env or None,
        allowed_channels=[c.strip() for c in channels.split(",") if c.strip()],
    )

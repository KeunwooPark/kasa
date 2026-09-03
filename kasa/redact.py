"""Keep secret material out of logs and out of model prompts.

Two sources feed this. The first is exact: Kasa knows the names of the
environment variables holding its own credentials, so it knows their values and
can match them literally. The second is shape-based, for tokens Kasa was never
told about — a key pasted into a chat message, or one echoed back inside a tool
result — matched by the prefixes the major providers issue.

Neither is a guarantee, and this is not a substitute for not putting secrets
somewhere. It is the net under the times somebody does.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kasa.config import Config

#: Below this length a "secret" matches too much ordinary prose to replace. A
#: test key of "k" would otherwise redact every letter k in the transcript.
MIN_SECRET_LENGTH = 12

REDACTED = "[redacted]"

#: Token shapes worth catching even when Kasa has never been told the value.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai key", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_-]{20,}")),
    ("github token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}")),
    ("github token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}")),
    ("slack token", re.compile(r"\bxox[aborps]-[A-Za-z0-9-]{10,}")),
    ("slack app token", re.compile(r"\bxapp-[A-Za-z0-9-]{10,}")),
    # A credential smuggled into a URL, which is how tokens usually end up in a
    # git remote or an error message.
    ("url credentials", re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)")),
)


class Redactor:
    """Replaces known and probable secrets in text."""

    def __init__(self, secrets: Mapping[str, str] | None = None) -> None:
        self._known = {
            value: f"[redacted:{name}]"
            for name, value in (secrets or {}).items()
            if len(value) >= MIN_SECRET_LENGTH
        }

    @classmethod
    def from_config(cls, cfg: Config) -> Redactor:
        """Read the values of every env var the config references."""
        return cls(_environ_values(_referenced_env_names(cfg)))

    def scrub(self, text: str) -> str:
        if not text:
            return text
        # Longest first, so a secret that contains another is not left with a
        # readable tail after the shorter one is replaced inside it.
        for value in sorted(self._known, key=len, reverse=True):
            text = text.replace(value, self._known[value])
        for _, pattern in _PATTERNS:
            text = pattern.sub(REDACTED, text)
        return text

    def install(self, logger: logging.Logger | None = None) -> logging.Filter:
        """Scrub every record written by `logger`'s handlers (the root by default).

        Attached to the *handlers*, not to the logger. A filter on a logger is
        consulted only for records logged on that logger itself — records from
        `kasa.core.tools` propagate to the handlers of `kasa` without ever
        consulting its filters — so installing on the logger would silently miss
        almost everything Kasa logs.
        """
        target = logger if logger is not None else logging.getLogger()
        log_filter = _RedactingFilter(self)
        for handler in target.handlers:
            handler.addFilter(log_filter)
        if not target.handlers:
            # Nothing to attach to yet. Better a filter that catches only direct
            # records than none at all.
            target.addFilter(log_filter)
        return log_filter


class _RedactingFilter(logging.Filter):
    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        # Formatting here rather than scrubbing `msg` and `args` separately: a
        # secret can be split across the format string and its arguments, and
        # only the joined result is guaranteed to contain it intact.
        record.msg = self._redactor.scrub(record.getMessage())
        record.args = ()
        return True


def _referenced_env_names(cfg: Config) -> set[str]:
    from kasa.config import default_key_env

    names = {cfg.ltm.token_env}
    names |= {n for n in (cfg.slack.app_token_env, cfg.slack.bot_token_env) if n}
    for provider in cfg.llm.values():
        for entry in (provider, *provider.fallbacks):
            names.add(entry.key_env or default_key_env(entry.kind))
    return names


def _environ_values(names: Iterable[str]) -> dict[str, str]:
    return {name: value for name in names if (value := os.environ.get(name))}

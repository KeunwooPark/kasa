"""Slack, over Socket Mode.

Imports `slack_bolt`, which is the `slack` extra: `uv sync --extra slack`.
"""

from kasa.adapters.slack.app import SlackAdapter
from kasa.adapters.slack.events import (
    Accepted,
    Decision,
    Ignored,
    SlackContext,
    message_id,
    normalize,
    scope_for,
    session_id,
)

__all__ = [
    "Accepted",
    "Decision",
    "Ignored",
    "SlackAdapter",
    "SlackContext",
    "message_id",
    "normalize",
    "scope_for",
    "session_id",
]

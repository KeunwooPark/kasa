"""Posting into a channel from something that is not a conversation.

`reflect` has a digest to publish and no Slack connection: it is a background
job, not a turn, and the daemon's Socket Mode adapter is a receiver rather than
something a job can borrow. One authenticated POST is the whole requirement.

Plain `httpx` rather than `slack_sdk`, deliberately. `slack_sdk` arrives with
the `slack` extra, and a nightly digest must not be the thing that makes that
extra mandatory for an install talking to Kasa over a terminal. It is also why
this module imports nothing from `app.py`.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from kasa.errors import KasaError

log = logging.getLogger(__name__)

POST_MESSAGE = "https://slack.com/api/chat.postMessage"

TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0)


class SlackPostError(KasaError):
    """Slack refused the message."""


async def post_message(
    token: str, channel: str, text: str, *, client: httpx.AsyncClient | None = None
) -> str:
    """Post `text` to `channel`, returning the message timestamp.

    Slack answers a rejected call with HTTP 200 and `{"ok": false}`, so the
    status code is not the check — a caller that only looked at it would report
    a digest as delivered every night while nothing was ever posted.
    """
    owned = client is None
    http = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        response = await http.post(
            POST_MESSAGE,
            headers={"Authorization": f"Bearer {token}"},
            json={"channel": channel, "text": text},
        )
        body: dict[str, Any] = response.json()
    except httpx.HTTPError as exc:
        raise SlackPostError(f"could not reach Slack: {exc}") from exc
    except ValueError as exc:
        raise SlackPostError("Slack did not answer with JSON") from exc
    finally:
        if owned:
            await http.aclose()

    if not body.get("ok"):
        raise SlackPostError(f"Slack refused the message: {body.get('error', 'unknown')}")
    return str(body.get("ts", ""))

"""Posting the nightly digest into a channel."""

from __future__ import annotations

import httpx
import pytest

from kasa.adapters.slack.notify import SlackPostError, post_message
from tests.conftest import mock_client


async def test_a_message_is_posted_as_the_bot() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read().decode()
        return httpx.Response(200, json={"ok": True, "ts": "1756890000.123"})

    ts = await post_message(
        "xoxb-token", "C0123", "a digest", client=mock_client(handler, "https://slack.com")
    )

    assert ts == "1756890000.123"
    assert seen["auth"] == "Bearer xoxb-token"
    assert "a digest" in str(seen["body"])


async def test_a_refusal_is_an_error_however_slack_dressed_it() -> None:
    """Slack answers a rejected call with HTTP 200 and `ok: false`. A caller
    that only looked at the status would report the digest as delivered every
    night while nothing was ever posted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "error": "channel_not_found"})

    with pytest.raises(SlackPostError, match="channel_not_found"):
        await post_message(
            "xoxb-token", "C0123", "a digest", client=mock_client(handler, "https://slack.com")
        )


async def test_slack_being_unreachable_is_an_error_not_a_traceback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with pytest.raises(SlackPostError, match="could not reach Slack"):
        await post_message(
            "xoxb-token", "C0123", "a digest", client=mock_client(handler, "https://slack.com")
        )

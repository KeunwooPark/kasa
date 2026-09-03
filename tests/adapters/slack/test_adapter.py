"""The Slack adapter: ack fast, dedupe hard, answer in the thread."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("slack_bolt", reason="the `slack` extra")

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from kasa.adapters.slack.app import NO_HTTP_VERIFICATION, SlackAdapter
from kasa.adapters.slack.events import SlackContext
from kasa.core.agent import Agent
from kasa.core.context import ContextPacker
from kasa.core.tools import ToolRegistry
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.tokens import Tokenizer
from kasa.llm.types import ChatRequest, Delta
from kasa.store import Store
from tests.conftest import until
from tests.core.test_agent import ScriptedProvider, says

BOT = "U0KASA"
TEAM = "T0TEAM"
HUMAN = "U0HUMAN"


class RecordingClient(AsyncWebClient):
    """A Slack client that keeps what it was asked to post instead of posting."""

    def __init__(self) -> None:
        super().__init__(token="xoxb-test")
        self.posted: list[dict[str, Any]] = []

    async def chat_postMessage(self, **kwargs: Any) -> Any:
        self.posted.append(kwargs)
        return {"ok": True}


class SlowProvider(ScriptedProvider):
    """A model that takes its time, which is the case ingress has to survive."""

    def __init__(self, script: list[Any], *, delay: float) -> None:
        super().__init__(script)
        self._delay = delay

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        await asyncio.sleep(self._delay)
        async for delta in super().stream(req):
            yield delta


def make_agent(store: Store, tokenizer: Tokenizer, provider: ScriptedProvider) -> Agent:
    return Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=ToolRegistry([]),
        packer=ContextPacker(tokenizer=tokenizer),
    )


def make_adapter(
    store: Store,
    tokenizer: Tokenizer,
    *,
    provider: ScriptedProvider | None = None,
    concurrency: int = 8,
) -> tuple[SlackAdapter, RecordingClient]:
    client = RecordingClient()
    app = AsyncApp(
        client=client,
        signing_secret=NO_HTTP_VERIFICATION,
        request_verification_enabled=False,
    )
    adapter = SlackAdapter(
        make_agent(store, tokenizer, provider or ScriptedProvider([says("noted")] * 200)),
        app=app,
        context=SlackContext(bot_user_id=BOT, team_id=TEAM),
        app_token="xapp-test",
        concurrency=concurrency,
    )
    return adapter, client


def mention(
    ts: str = "1700000000.000100", text: str = f"<@{BOT}> what did we decide?"
) -> dict[str, Any]:
    return {
        "type": "message",
        "channel_type": "channel",
        "channel": "C0DEPLOY",
        "user": HUMAN,
        "text": text,
        "ts": ts,
    }


# -- ingress ------------------------------------------------------------------


async def test_the_ack_path_never_waits_for_a_turn(store: Store, tokenizer: Tokenizer) -> None:
    """The acceptance criterion. Slack gives a listener three seconds and bolt
    acks once it returns, so the measurement has to be taken with every turn
    slot occupied — which is the state a busy channel is normally in."""
    provider = SlowProvider([says("noted")] * 200, delay=1.0)
    adapter, _ = make_adapter(store, tokenizer, provider=provider, concurrency=8)
    running = asyncio.create_task(adapter.runtime.run())

    slowest = 0.0
    try:
        for n in range(8):
            await adapter.on_event(mention(ts=f"1700000000.{n:06d}"))
        await until(lambda: adapter.runtime.dispatcher.in_flight == 8)

        for n in range(50):
            started = time.perf_counter()
            await adapter.on_event(mention(ts=f"1700000001.{n:06d}"))
            slowest = max(slowest, time.perf_counter() - started)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=30.0)

    assert len(provider.requests) == 8, "the slots really were full"
    assert slowest < 0.5, f"slowest ack was {slowest:.3f}s against a 3s budget"


async def test_an_ignored_event_is_not_queued(store: Store, tokenizer: Tokenizer) -> None:
    adapter, _ = make_adapter(store, tokenizer)

    await adapter.on_event(mention(text="nothing to do with the bot"))

    assert await adapter.runtime.inbox.counts() == {}


async def test_a_forced_retry_produces_exactly_one_reply(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The other acceptance criterion. Slack re-sends aggressively, and the
    same message also arrives as both `message` and `app_mention`."""
    adapter, client = make_adapter(store, tokenizer)
    body = mention()
    as_mention = {k: v for k, v in body.items() if k != "channel_type"} | {"type": "app_mention"}

    running = asyncio.create_task(adapter.runtime.run())
    try:
        for delivery in (body, body, as_mention, body):
            await adapter.on_event(delivery)
        await until(lambda: len(client.posted) >= 1)
        await asyncio.sleep(0.2)  # a second answer would land in this window
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert len(client.posted) == 1, client.posted
    assert await adapter.runtime.inbox.counts() == {"done": 1}


# -- egress -------------------------------------------------------------------


async def test_the_answer_goes_back_into_the_thread(store: Store, tokenizer: Tokenizer) -> None:
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: len(client.posted) == 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.posted[0] == {
        "channel": "C0DEPLOY",
        "thread_ts": "1700000000.000100",
        "text": "noted",
    }


async def test_a_turn_with_nothing_to_say_says_why(store: Store, tokenizer: Tokenizer) -> None:
    """A silent turn on Slack is no more debuggable than a silent one on a tty."""
    adapter, client = make_adapter(store, tokenizer, provider=ScriptedProvider([says("")] * 4))
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: len(client.posted) == 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert client.posted[0]["text"] == "_the model returned nothing._"


async def test_a_turn_is_scoped_to_the_channel_it_came_from(
    store: Store, tokenizer: Tokenizer
) -> None:
    """Retrieval filters on the session's scope before it ranks, so a scope
    written wrong here is where a private conversation starts leaking."""
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention())
        await until(lambda: len(client.posted) == 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    session = await store.get_session(f"slack:{TEAM}:C0DEPLOY:1700000000.000100")
    assert session is not None
    assert session["scope"] == "channel:C0DEPLOY"


async def test_a_dm_is_scoped_to_the_person_in_it(store: Store, tokenizer: Tokenizer) -> None:
    adapter, client = make_adapter(store, tokenizer)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(
            {
                "type": "message",
                "channel_type": "im",
                "channel": "D0PRIVATE",
                "user": HUMAN,
                "text": "remember that I hate standups",
                "ts": "1700000000.000100",
            }
        )
        await until(lambda: len(client.posted) == 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    session = await store.get_session(f"slack:{TEAM}:D0PRIVATE:1700000000.000100")
    assert session is not None
    assert session["scope"] == f"private:{HUMAN}"

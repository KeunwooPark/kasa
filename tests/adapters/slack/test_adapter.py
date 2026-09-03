"""The Slack adapter: ack fast, dedupe hard, answer in the thread."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest

pytest.importorskip("slack_bolt", reason="the `slack` extra")

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from kasa.adapters import slack as package
from kasa.adapters.slack.app import NO_HTTP_VERIFICATION, SlackAdapter
from kasa.adapters.slack.events import SlackContext, normalize
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


async def test_a_file_entry_that_is_not_an_object_still_reaches_the_inbox(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The repro from #123, measured where it hurts: the row, not the decision.

    `_with_attachments` read the payload's shape and trusted it, so an entry
    that was not an object raised out of `normalize` — before the INSERT, and
    therefore before anything recorded that the message had arrived at all.
    """
    adapter, client = make_adapter(store, tokenizer)
    body = mention(text=f"<@{BOT}> what's in this?") | {
        "subtype": "file_share",
        "files": ["F0123456"],
    }

    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(body)
        await until(lambda: len(client.posted) == 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

    assert await adapter.runtime.inbox.counts() == {"done": 1}


async def test_an_event_that_cannot_be_read_is_ignored_rather_than_raised(
    store: Store, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """Ingress must reach a decision for every payload Slack sends.

    An exception out of `normalize` lands before the inbox row is written,
    which is the one thing this path exists to guarantee: Slack retries three
    times, each raises, and the message is gone with no record that it ever
    arrived. `normalize` is patched rather than fed a payload that breaks it
    today, because the guard is for the shape nobody has thought of yet.
    """

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("a shape nobody anticipated")

    monkeypatch.setattr("kasa.adapters.slack.app.normalize", boom)
    adapter, _ = make_adapter(store, tokenizer)

    with caplog.at_level(logging.ERROR, logger="kasa.adapters.slack.app"):
        await adapter.on_event(mention())

    assert "could not read a slack event" in caplog.text
    assert "a shape nobody anticipated" in caplog.text, "the traceback is the whole point"


async def test_ingress_still_works_after_an_event_it_could_not_read(
    store: Store, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surviving one bad event is only worth having if the socket survives too."""
    adapter, client = make_adapter(store, tokenizer)
    real = normalize
    calls = 0

    async def once(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("a shape nobody anticipated")
        return await real(*args, **kwargs)

    monkeypatch.setattr("kasa.adapters.slack.app.normalize", once)
    running = asyncio.create_task(adapter.runtime.run())
    try:
        await adapter.on_event(mention(ts="1700000000.000100"))
        await adapter.on_event(mention(ts="1700000000.000200"))
        await until(lambda: len(client.posted) == 1)
    finally:
        adapter.runtime.stop()
        await asyncio.wait_for(running, timeout=10.0)

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


# -- the package --------------------------------------------------------------


def test_the_adapter_resolves_through_the_lazy_import() -> None:
    """#119 put `SlackAdapter` behind a module `__getattr__` so that
    `kasa.adapters.slack.events` imports on an install that never asked for the
    `slack` extra. The two tests that cover *that* half run in subprocesses,
    because making `slack_bolt` unimportable inside an environment that has it
    is the only way to reproduce a missing extra — and a subprocess is
    something coverage cannot see, so the lazy path read as dead code.

    This is the half that runs here: with the extra installed, the name
    resolves to the class. It is also the assertion that fails if the name is
    dropped from `__all__` or misspelled inside `__getattr__`, neither of which
    the subprocess pair would notice — both assert on the failure path.
    """
    assert "SlackAdapter" in package.__all__
    assert package.SlackAdapter is SlackAdapter


def test_every_name_the_package_exports_can_be_reached() -> None:
    """`__all__` is what `from ... import *` and a reader both go by, so a name
    on it that resolves to nothing is a promise the package does not keep."""
    for name in package.__all__:
        assert getattr(package, name) is not None, name


def test_the_package_says_no_to_a_name_it_does_not_have() -> None:
    """A `__getattr__` that falls off the end returns `None` instead of
    raising, which makes `hasattr` true for everything and turns a typo into a
    silent `None` at the call site rather than an import error."""
    missing = "Nonexistent"

    with pytest.raises(AttributeError, match="has no attribute 'Nonexistent'"):
        getattr(package, missing)

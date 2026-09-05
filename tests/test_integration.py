"""The whole v0 stack, wired the way `kasa run` wires it.

Everything below the CLI is real: a real provider adapter parsing real wire
format off a mock transport, the real registry, packer, tool dispatch and store.
Only the socket is fake.
"""

from __future__ import annotations

import json

import httpx

from kasa.core.agent import Agent, AgentConfig
from kasa.core.context import ContextPacker
from kasa.core.tools import ToolRegistry, builtin_tools
from kasa.llm.anthropic_compat import AnthropicCompatProvider
from kasa.llm.cost import CostMeter, Price, PriceBook
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.tokens import Tokenizer
from kasa.store import Store
from tests.conftest import mock_client, sse

ASKS_FOR_TOOL = [
    ("message_start", {"message": {"model": "stub-1", "usage": {"input_tokens": 30}}}),
    (
        "content_block_start",
        {"index": 0, "content_block": {"type": "tool_use", "id": "tu_1", "name": "current_time"}},
    ),
    (
        "content_block_delta",
        {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
    ),
    ("content_block_stop", {"index": 0}),
    ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}}),
    ("message_stop", {}),
]

ANSWERS = [
    ("message_start", {"message": {"model": "stub-1", "usage": {"input_tokens": 42}}}),
    ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "It is "}}),
    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "noon UTC."}}),
    ("content_block_stop", {"index": 0}),
    ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}}),
    ("message_stop", {}),
]


async def test_tool_using_turn_end_to_end(store: Store, tokenizer: Tokenizer) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        # First call reaches for the tool; the follow-up answers with it.
        events = ASKS_FOR_TOOL if len(seen) == 1 else ANSWERS
        return httpx.Response(200, content=sse(events))

    provider = AnthropicCompatProvider(model="stub-1", api_key="fake", client=mock_client(handler))
    meter = CostMeter(PriceBook({"stub": Price(input=3.0, output=15.0)}), sink=store.record_call)
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}, meter=meter),
        store=store,
        tools=ToolRegistry(builtin_tools()),
        packer=ContextPacker(tokenizer=tokenizer),
        config=AgentConfig(),
    )

    result = await agent.respond("cli:1", "what time is it?")

    assert result.text == "It is noon UTC."
    assert result.tool_calls == 1
    assert result.iterations == 2

    # The wire carried what each side expects.
    assert seen[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert [t["name"] for t in seen[0]["tools"]] == ["current_time"]
    assert any(
        block.get("type") == "tool_result"
        for message in seen[1]["messages"]
        for block in message["content"]
    )
    # Prompt caching depends on the cached block holding for the life of the
    # session. The per-turn block behind it is where anything that changes goes,
    # and since #201 the tool-budget countdown is one of those things.
    assert seen[0]["system"][0] == seen[1]["system"][0]
    assert "40 tool rounds are left" in seen[0]["system"][1]["text"]
    assert "39 tool rounds are left" in seen[1]["system"][1]["text"]
    assert "cache_control" not in seen[1]["system"][1]

    # The transcript replays.
    stored = await store.recent_messages("cli:1", limit=100)
    assert [m.role for m in stored] == ["user", "assistant", "user", "assistant"]
    assert {b.id for m in stored for b in m.tool_uses} == {
        b.tool_use_id for m in stored for b in m.tool_results_in
    }

    # Both calls were metered, with usage and cost.
    calls = await store.raw("SELECT model, input_tokens, output_tokens, cost_usd FROM llm_calls")
    assert len(calls) == 2
    assert sum(c["input_tokens"] for c in calls) == 72
    assert sum(c["output_tokens"] for c in calls) == 11
    assert all(c["cost_usd"] > 0 for c in calls)

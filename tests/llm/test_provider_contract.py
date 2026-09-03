"""One suite, run against both provider families.

This is the acceptance criterion for the two adapters (#4, #5): whatever the
wire format, the canonical types coming out the other side must be identical.
Anything asserted here is a promise the rest of Kasa relies on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from kasa.errors import (
    AuthError,
    ContextOverflowError,
    ProviderProtocolError,
    RateLimitError,
    TransientError,
)
from kasa.llm.anthropic_compat import AnthropicCompatProvider
from kasa.llm.base import LLMProvider, collect
from kasa.llm.openai_compat import OpenAICompatProvider
from kasa.llm.types import (
    ChatRequest,
    Message,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)
from tests.conftest import mock_client, sse

WEATHER = ToolDef(
    name="get_weather",
    description="Look up the weather.",
    input_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


@dataclass
class Case:
    """A provider plus canned responses in its own wire format."""

    name: str
    path: str
    text_body: dict[str, Any]
    tool_body: dict[str, Any]
    stream_events: list[tuple[str | None, dict[str, Any]]]
    stream_done: bool
    requests: list[httpx.Request] = field(default_factory=list)

    def build(self, response: httpx.Response | bytes) -> LLMProvider:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if isinstance(response, bytes):
                return httpx.Response(200, content=response)
            return response

        client = mock_client(handler)
        if self.name == "openai":
            return OpenAICompatProvider(model="test-model", api_key="k", client=client)
        return AnthropicCompatProvider(model="test-model", api_key="k", client=client)

    @property
    def sent(self) -> dict[str, Any]:
        import json

        return json.loads(self.requests[-1].content)


def openai_case() -> Case:
    return Case(
        name="openai",
        path="/chat/completions",
        text_body={
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 7},
            },
        },
        tool_body={
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "checking",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Seoul"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 9},
        },
        stream_events=[
            (
                None,
                {"model": "test-model", "choices": [{"index": 0, "delta": {"content": "check"}}]},
            ),
            (None, {"choices": [{"index": 0, "delta": {"content": "ing"}}]}),
            (
                None,
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "get_weather", "arguments": '{"ci'},
                                    }
                                ]
                            },
                        }
                    ]
                },
            ),
            (
                None,
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": 'ty": "Seoul"}'}}
                                ]
                            },
                        }
                    ]
                },
            ),
            (None, {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            (None, {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9}}),
        ],
        stream_done=True,
    )


def anthropic_case() -> Case:
    return Case(
        name="anthropic",
        path="/messages",
        text_body={
            "model": "test-model",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 3, "cache_read_input_tokens": 7},
        },
        tool_body={
            "model": "test-model",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Seoul"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 9},
        },
        stream_events=[
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {"model": "test-model", "usage": {"input_tokens": 20}},
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "check"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ing"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"ci'},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": 'ty": "Seoul"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 9},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ],
        stream_done=False,
    )


@pytest.fixture(params=["openai", "anthropic"])
def case(request: pytest.FixtureRequest) -> Case:
    return openai_case() if request.param == "openai" else anthropic_case()


CONVERSATION = ChatRequest(
    system="You are Kasa.",
    messages=(Message.user("weather in Seoul?"),),
    tools=(WEATHER,),
    max_tokens=512,
)


# -- responses ---------------------------------------------------------------


async def test_text_turn(case: Case) -> None:
    provider = case.build(httpx.Response(200, json=case.text_body))
    resp = await provider.complete(CONVERSATION)

    assert resp.text == "hi there"
    assert resp.stop_reason == "end_turn"
    assert resp.message.role == "assistant"
    assert resp.usage.input_tokens == 11
    assert resp.usage.output_tokens == 3
    assert resp.usage.cache_read_tokens == 7


async def test_tool_turn(case: Case) -> None:
    provider = case.build(httpx.Response(200, json=case.tool_body))
    resp = await provider.complete(CONVERSATION)

    assert resp.stop_reason == "tool_use"
    assert resp.text == "checking"
    assert len(resp.tool_uses) == 1
    call = resp.tool_uses[0]
    assert call.name == "get_weather"
    assert call.input == {"city": "Seoul"}


async def test_streaming_matches_non_streaming(case: Case) -> None:
    """The same turn, streamed, must fold back into the same response."""
    provider = case.build(sse(case.stream_events, done=case.stream_done))
    streamed = await collect(provider.stream(CONVERSATION), model="test-model", provider=case.name)

    assert streamed.text == "checking"
    assert streamed.stop_reason == "tool_use"
    # Arguments arrived split mid-key across two frames.
    assert streamed.tool_uses[0].input == {"city": "Seoul"}
    assert streamed.usage.output_tokens == 9
    assert streamed.usage.input_tokens == 20


# -- requests ----------------------------------------------------------------


async def test_system_prompt_placement(case: Case) -> None:
    provider = case.build(httpx.Response(200, json=case.text_body))
    await provider.complete(CONVERSATION)
    sent = case.sent

    if case.name == "anthropic":
        # Top-level parameter, in block form so it can carry the cache marker.
        assert sent["system"][0]["text"] == "You are Kasa."
        assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert all(m["role"] != "system" for m in sent["messages"])
    else:
        assert sent["messages"][0] == {"role": "system", "content": "You are Kasa."}


async def test_per_turn_context_stays_out_of_the_cached_prefix(case: Case) -> None:
    provider = case.build(httpx.Response(200, json=case.text_body))
    await provider.complete(CONVERSATION.model_copy(update={"context": "recalled: it is cold"}))
    sent = case.sent

    if case.name == "anthropic":
        blocks = sent["system"]
        assert len(blocks) == 2
        assert "cache_control" in blocks[0]
        # The half that changes every turn must not be marked cacheable.
        assert "cache_control" not in blocks[1]
        assert blocks[1]["text"] == "recalled: it is cold"
    else:
        content = sent["messages"][0]["content"]
        assert content.startswith("You are Kasa.")
        assert "recalled: it is cold" in content


async def test_tool_schema_shape(case: Case) -> None:
    provider = case.build(httpx.Response(200, json=case.text_body))
    await provider.complete(CONVERSATION)
    tool = case.sent["tools"][0]

    if case.name == "anthropic":
        assert tool["name"] == "get_weather"
        assert tool["input_schema"] == WEATHER.input_schema
    else:
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"] == WEATHER.input_schema


async def test_tool_results_round_trip(case: Case) -> None:
    """A full tool exchange must serialize into whatever shape each API wants."""
    provider = case.build(httpx.Response(200, json=case.text_body))
    conversation = ChatRequest(
        system="You are Kasa.",
        messages=(
            Message.user("weather in Seoul?"),
            Message(
                role="assistant",
                content=(
                    TextBlock(text="checking"),
                    ToolUseBlock(id="t1", name="get_weather", input={"city": "Seoul"}),
                ),
            ),
            Message.tool_results([ToolResultBlock(tool_use_id="t1", content="4C")]),
        ),
        tools=(WEATHER,),
    )
    await provider.complete(conversation)
    messages = case.sent["messages"]
    # The OpenAI family prepends a system message, so index by role, not position.
    assistant = next(m for m in messages if m["role"] == "assistant")

    if case.name == "anthropic":
        # No `tool` role here: the result rides inside a user turn.
        assert messages[-1]["role"] == "user"
        block = messages[-1]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "t1"
        assert assistant["content"][1]["type"] == "tool_use"
    else:
        assert messages[-1] == {"role": "tool", "tool_call_id": "t1", "content": "4C"}
        assert assistant["tool_calls"][0]["id"] == "t1"


async def test_max_tokens_always_sent(case: Case) -> None:
    # Required by the Anthropic family, optional for the other; sending it
    # unconditionally keeps one code path.
    provider = case.build(httpx.Response(200, json=case.text_body))
    await provider.complete(CONVERSATION)
    assert case.sent["max_tokens"] == 512


# -- errors ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, '{"error": "bad key"}', AuthError),
        (429, '{"error": "slow down"}', RateLimitError),
        (503, '{"error": "upstream down"}', TransientError),
        (400, '{"error": "maximum context length exceeded"}', ContextOverflowError),
    ],
)
async def test_error_mapping(case: Case, status: int, body: str, expected: type[Exception]) -> None:
    provider = case.build(httpx.Response(status, text=body))
    with pytest.raises(expected):
        await provider.complete(CONVERSATION)


async def test_rate_limit_carries_retry_after(case: Case) -> None:
    provider = case.build(httpx.Response(429, text="slow down", headers={"retry-after": "2.5"}))
    with pytest.raises(RateLimitError) as exc:
        await provider.complete(CONVERSATION)
    assert exc.value.retry_after == 2.5


async def test_retryability_is_classified(case: Case) -> None:
    """The registry's policy depends on this flag, not on status codes."""
    provider = case.build(httpx.Response(500, text="boom"))
    with pytest.raises(TransientError) as exc:
        await provider.complete(CONVERSATION)
    assert exc.value.retryable is True

    provider = case.build(httpx.Response(401, text="nope"))
    with pytest.raises(AuthError) as auth:
        await provider.complete(CONVERSATION)
    assert auth.value.retryable is False


async def test_unparseable_tool_arguments_are_a_protocol_error(case: Case) -> None:
    if case.name == "openai":
        body = {
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": "{not json"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    else:
        # This API sends parsed input, so the equivalent failure is a stream
        # whose JSON fragments never form a valid object.
        body = None

    if body is None:
        events = [
            (
                "content_block_start",
                {
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "get_weather"},
                },
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{not json"}},
            ),
            ("message_stop", {}),
        ]
        provider = case.build(sse(events))
        with pytest.raises(ProviderProtocolError):
            await collect(provider.stream(CONVERSATION), model="m", provider=case.name)
    else:
        provider = case.build(httpx.Response(200, json=body))
        with pytest.raises(ProviderProtocolError):
            await provider.complete(CONVERSATION)

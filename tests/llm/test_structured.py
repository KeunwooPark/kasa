"""Asking a model for a typed answer, and what happens when it does not give one."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel, ConfigDict

from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.structured import (
    StructuredOutputError,
    complete_json,
    parse_json_object,
)
from kasa.llm.types import ChatRequest, ChatResponse, Delta, Message, Usage


class Shape(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    count: int


class Scripted:
    """Returns a fixed list of replies, one per call."""

    name = "scripted"
    model = "m"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        return ChatResponse(
            message=Message.assistant(self.replies.pop(0)),
            stop_reason="end_turn",
            usage=Usage(),
            model="m",
        )

    def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


def registry_of(provider: Scripted) -> ProviderRegistry:
    return ProviderRegistry({ModelRole.UTILITY: [provider]})


# -- parsing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"name": "a", "count": 1}',
        '```json\n{"name": "a", "count": 1}\n```',
        'Here is the JSON:\n{"name": "a", "count": 1}',
        '{"name": "a", "count": 1}\n\nLet me know if you need anything else.',
    ],
)
def test_the_json_is_found_however_the_model_wrapped_it(reply: str) -> None:
    """Leniency about the wrapping, never about the contents. Every one of
    these is a real shape a model returns when told to reply with JSON only."""
    assert parse_json_object(reply) == {"name": "a", "count": 1}


@pytest.mark.parametrize("reply", ["", "I would rather not.", "{not json at all}"])
def test_a_reply_with_no_object_in_it_is_an_error_not_a_guess(reply: str) -> None:
    with pytest.raises(StructuredOutputError):
        parse_json_object(reply)


# -- the call ----------------------------------------------------------------


async def test_a_valid_reply_costs_one_call() -> None:
    provider = Scripted('{"name": "a", "count": 1}')

    result = await complete_json(
        registry_of(provider),
        ModelRole.UTILITY,
        Shape,
        system="be a shape",
        prompt="the material",
        tag="test",
    )

    assert result == Shape(name="a", count=1)
    assert len(provider.requests) == 1


async def test_the_schema_travels_in_the_system_prompt() -> None:
    """The material changes every call and the schema does not, so the schema
    belongs in the half a provider is asked to cache."""
    provider = Scripted('{"name": "a", "count": 1}')

    await complete_json(
        registry_of(provider),
        ModelRole.UTILITY,
        Shape,
        system="be a shape",
        prompt="the material",
        tag="test",
    )

    request = provider.requests[0]
    assert "be a shape" in (request.system or "")
    assert '"count"' in (request.system or "")
    assert request.messages[0].text == "the material"
    # Extraction is not a place to be creative, and a plan that was rejected
    # should be reproducible from the same input.
    assert request.temperature == 0.0


async def test_an_unusable_reply_is_asked_about_once_more() -> None:
    provider = Scripted("no thanks", '{"name": "a", "count": 1}')

    result = await complete_json(
        registry_of(provider),
        ModelRole.UTILITY,
        Shape,
        system="be a shape",
        prompt="the material",
        tag="test",
    )

    assert result == Shape(name="a", count=1)
    assert len(provider.requests) == 2, "the repair attempt"
    # The failed reply is shown back, so the second attempt is a correction
    # rather than the same request run again.
    assert provider.requests[1].messages[1].text == "no thanks"


async def test_a_reply_that_validates_as_the_wrong_shape_is_also_repaired() -> None:
    """JSON is not the bar. `{"name": "a"}` parses perfectly and is not a
    `Shape`, and a caller handed a half-populated model has no way to tell."""
    provider = Scripted('{"name": "a"}', '{"name": "a", "count": 2}')

    result = await complete_json(
        registry_of(provider),
        ModelRole.UTILITY,
        Shape,
        system="be a shape",
        prompt="the material",
        tag="test",
    )

    assert result == Shape(name="a", count=2)


async def test_two_bad_replies_raise_rather_than_a_third_attempt() -> None:
    """The caller has a decision to make about the material. Spending a third
    call on a model that has failed twice is not it."""
    provider = Scripted("no", "still no")

    with pytest.raises(StructuredOutputError, match="two attempts"):
        await complete_json(
            registry_of(provider),
            ModelRole.UTILITY,
            Shape,
            system="be a shape",
            prompt="the material",
            tag="test",
        )

    assert len(provider.requests) == 2

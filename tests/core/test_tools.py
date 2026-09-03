"""Dispatch must never raise into the turn loop.

Every failure comes back as an error `tool_result` so the model can see it and
correct. The alternative — letting it propagate — strands an assistant
`tool_use` with no matching result, which poisons every later turn.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from kasa.core.tools import Tool, ToolRegistry, builtin_tools
from kasa.llm.types import ToolUseBlock

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


async def echo(args: dict[str, Any]) -> str:
    return f"weather in {args['city']}"


async def explode(args: dict[str, Any]) -> str:
    raise RuntimeError("upstream is down")


async def hang(args: dict[str, Any]) -> str:
    await asyncio.sleep(10)
    return "never"


def registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(name="weather", description="d", input_schema=SCHEMA, handler=echo),
            Tool(name="explode", description="d", input_schema=SCHEMA, handler=explode),
            Tool(name="hang", description="d", input_schema=SCHEMA, handler=hang, timeout=0.01),
        ]
    )


def use(name: str, **args: Any) -> ToolUseBlock:
    return ToolUseBlock(id="t1", name=name, input=args)


async def test_dispatch_returns_the_handler_result() -> None:
    result = await registry().dispatch(use("weather", city="Seoul"))
    assert result.content == "weather in Seoul"
    assert result.is_error is False
    assert result.tool_use_id == "t1"


async def test_unknown_tool_becomes_an_error_result() -> None:
    result = await registry().dispatch(use("nope", city="Seoul"))
    assert result.is_error
    # Naming the real tools lets the model recover on the next iteration.
    assert "weather" in result.content


async def test_schema_violations_become_error_results() -> None:
    result = await registry().dispatch(use("weather"))
    assert result.is_error
    assert "invalid arguments" in result.content


async def test_extra_properties_are_rejected() -> None:
    result = await registry().dispatch(use("weather", city="Seoul", extra=1))
    assert result.is_error


async def test_handler_exceptions_become_error_results() -> None:
    result = await registry().dispatch(use("explode", city="Seoul"))
    assert result.is_error
    assert "RuntimeError" in result.content
    assert "upstream is down" in result.content


async def test_a_hanging_tool_times_out() -> None:
    result = await registry().dispatch(use("hang", city="Seoul"))
    assert result.is_error
    assert "timed out" in result.content


async def test_cancellation_propagates() -> None:
    """The one exception: an aborted turn must not be reported to the model."""
    reg = ToolRegistry(
        [Tool(name="hang", description="d", input_schema=SCHEMA, handler=hang, timeout=10)]
    )
    task = asyncio.create_task(reg.dispatch(use("hang", city="Seoul")))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_definitions_are_sorted_for_prefix_stability() -> None:
    """Tool schemas sit in the cacheable prefix, so their order must not drift."""
    names = [d.name for d in registry().defs()]
    assert names == sorted(names)


def test_duplicate_registration_is_rejected() -> None:
    reg = registry()
    with pytest.raises(ValueError, match="already registered"):
        reg.register(Tool(name="weather", description="d", input_schema=SCHEMA, handler=echo))


def test_invalid_schema_is_rejected_at_registration() -> None:
    with pytest.raises(Exception):  # noqa: B017 - jsonschema's own error type
        ToolRegistry(
            [Tool(name="bad", description="d", input_schema={"type": "not-a-type"}, handler=echo)]
        )


async def test_builtin_current_time() -> None:
    reg = ToolRegistry(builtin_tools())
    result = await reg.dispatch(ToolUseBlock(id="t1", name="current_time", input={}))
    assert not result.is_error
    assert result.content.endswith("+00:00")

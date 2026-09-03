"""Tool registry and dispatch.

Dispatch never raises into the agent loop. A bad argument, an unknown tool, or a
handler that blows up all come back as a `tool_result` marked `is_error`, so the
model sees the failure and can correct it. Killing the turn instead would strand
an assistant `tool_use` with no matching result, which is unrecoverable state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import jsonschema

from kasa.llm.types import ToolDef, ToolResultBlock, ToolUseBlock

log = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]

#: A handler that hangs would hang the turn, and the user is waiting.
DEFAULT_TOOL_TIMEOUT = 30.0


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
    timeout: float = DEFAULT_TOOL_TIMEOUT

    def to_def(self) -> ToolDef:
        return ToolDef(name=self.name, description=self.description, input_schema=self.input_schema)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        jsonschema.Draft202012Validator.check_schema(tool.input_schema)
        self._tools[tool.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def defs(self) -> tuple[ToolDef, ...]:
        # Sorted so the serialized tool block is byte-stable across turns, which
        # keeps it inside the cacheable prefix.
        return tuple(t.to_def() for t in sorted(self._tools.values(), key=lambda t: t.name))

    async def dispatch(self, use: ToolUseBlock) -> ToolResultBlock:
        tool = self._tools.get(use.name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return _error(use, f"unknown tool {use.name!r}. Available tools: {known}")

        try:
            jsonschema.validate(use.input, tool.input_schema)
        except jsonschema.ValidationError as exc:
            return _error(use, f"invalid arguments: {exc.message}")

        try:
            async with asyncio.timeout(tool.timeout):
                result = await tool.handler(use.input)
        except TimeoutError:
            return _error(use, f"tool timed out after {tool.timeout:g}s")
        except asyncio.CancelledError:
            # The turn is being aborted; the loop persists its own error result.
            raise
        except Exception as exc:
            log.exception("tool %s failed", use.name)
            return _error(use, f"{type(exc).__name__}: {exc}")

        return ToolResultBlock(tool_use_id=use.id, content=result)


def _error(use: ToolUseBlock, message: str) -> ToolResultBlock:
    return ToolResultBlock(tool_use_id=use.id, content=message, is_error=True)


# -- built-ins ---------------------------------------------------------------


async def _current_time(args: dict[str, Any]) -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


CURRENT_TIME = Tool(
    name="current_time",
    description="Get the current date and time in UTC, ISO 8601.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_current_time,
)


def builtin_tools() -> list[Tool]:
    """Tools available in every session.

    Kept deliberately thin at v0. The memory tools land in v1 (#16), where they
    enqueue observations rather than writing anything directly.
    """
    return [CURRENT_TIME]

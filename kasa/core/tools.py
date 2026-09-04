"""Tool registry and dispatch.

Dispatch never raises into the agent loop. A bad argument, an unknown tool, or a
handler that blows up all come back as a `tool_result` marked `is_error`, so the
model sees the failure and can correct it. Killing the turn instead would strand
an assistant `tool_use` with no matching result, which is unrecoverable state.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import jsonschema

from kasa.llm.types import ToolDef, ToolResultBlock, ToolUseBlock

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Who is calling, and what they are allowed to see.

    Passed explicitly rather than carried in a context variable. `scope` decides
    whether a tool call may read a private memory, and a security-relevant value
    hidden in ambient state is one that eventually gets read from the wrong
    place. The model never supplies it; the session does.
    """

    session_id: str = "cli"
    scope: str = "workspace"
    #: Memory ids the tools pulled into this turn, in the order they were
    #: reached. Mutable inside a frozen context on purpose: the scope above is
    #: a permission and must not be reassignable, while this is a notebook the
    #: turn writes as it goes. It is what lets a 👍 on the answer reach the
    #: memories a `memory_search` found, not only the ones pre-injected (#36).
    recalled: list[str] = field(default_factory=list)


ToolHandler = Callable[[dict[str, Any], ToolContext], Awaitable[str]]

#: Applied to every tool result before it re-enters the transcript. Tool output
#: is the one part of a prompt Kasa does not write itself, so it is the one part
#: that can carry a credential back to a provider.
Scrubber = Callable[[str], str]
SecretResolver = Callable[[str], str | None]

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
    def __init__(
        self,
        tools: list[Tool] | None = None,
        *,
        scrub: Scrubber | None = None,
        resolve_secret: SecretResolver | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._scrub: Scrubber = scrub or (lambda text: text)
        self._resolve_secret = resolve_secret
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

    async def dispatch(
        self, use: ToolUseBlock, context: ToolContext | None = None
    ) -> ToolResultBlock:
        tool = self._tools.get(use.name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            return self._error(use, f"unknown tool {use.name!r}. Available tools: {known}")

        try:
            jsonschema.validate(use.input, tool.input_schema)
        except jsonschema.ValidationError as exc:
            return self._error(use, f"invalid arguments: {exc.message}")

        try:
            async with asyncio.timeout(tool.timeout):
                arguments = _substitute_vault_refs(use.input, self._resolve_secret)
                result = await tool.handler(arguments, context or ToolContext())
        except TimeoutError:
            return self._error(use, f"tool timed out after {tool.timeout:g}s")
        except asyncio.CancelledError:
            # The turn is being aborted; the loop persists its own error result.
            raise
        except Exception as exc:
            log.exception("tool %s failed", use.name)
            # The message can quote whatever the handler was holding, a token in
            # a URL included, so it is scrubbed like any other result.
            return self._error(use, f"{type(exc).__name__}: {exc}")

        return ToolResultBlock(tool_use_id=use.id, content=self._scrub(result))

    def _error(self, use: ToolUseBlock, message: str) -> ToolResultBlock:
        return ToolResultBlock(tool_use_id=use.id, content=self._scrub(message), is_error=True)


_VAULT_REF = re.compile(r"\{\{vault:([A-Za-z0-9_.-]+)\}\}")


def _substitute_vault_refs(value: Any, resolver: SecretResolver | None) -> Any:
    """Resolve vault references only at the last boundary before a tool call."""
    if resolver is None:
        return value
    if isinstance(value, dict):
        return {key: _substitute_vault_refs(item, resolver) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute_vault_refs(item, resolver) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        secret = resolver(name)
        if secret is None:
            raise ValueError(f"vault secret {name!r} does not exist")
        return secret

    return _VAULT_REF.sub(replace, value)


# -- built-ins ---------------------------------------------------------------


async def _current_time(args: dict[str, Any], context: ToolContext) -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


CURRENT_TIME = Tool(
    name="current_time",
    description="Get the current date and time in UTC, ISO 8601.",
    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    handler=_current_time,
)


def builtin_tools() -> list[Tool]:
    """Tools available in every session, with or without a memory repo."""
    return [CURRENT_TIME]

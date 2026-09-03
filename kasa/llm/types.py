"""The canonical wire-format-agnostic types every other subsystem speaks.

The rule this module exists to enforce: **no provider's representation leaks
past `kasa/llm/*_compat.py`**. The agent loop, the packer and the memory
subsystem only ever see the types defined here.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]

StopReason = Literal[
    "end_turn",
    "max_tokens",
    "tool_use",
    "stop_sequence",
    "content_filter",
]


class _Block(BaseModel):
    model_config = ConfigDict(frozen=True)


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(_Block):
    """Extended-thinking output.

    Preserved through the round-trip so it can be replayed on the next turn:
    Anthropic-compatible providers reject a tool-use continuation whose prior
    assistant turn dropped its thinking blocks.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None


class ToolUseBlock(_Block):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(_Block):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: tuple[ContentBlock, ...]

    @classmethod
    def user(cls, text: str) -> Self:
        return cls(role="user", content=(TextBlock(text=text),))

    @classmethod
    def assistant(cls, text: str) -> Self:
        return cls(role="assistant", content=(TextBlock(text=text),))

    @classmethod
    def tool_results(cls, results: list[ToolResultBlock]) -> Self:
        """Bundle tool results into a single turn.

        Deliberately one message holding every result: both provider families
        expect the results for one assistant turn to arrive together, and
        splitting them produces a hard 400 on Anthropic-compatible endpoints.
        """
        return cls(role="user", content=tuple(results))

    @property
    def text(self) -> str:
        """Concatenated text blocks. Thinking and tool blocks are excluded."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolUseBlock))

    @property
    def tool_results_in(self) -> tuple[ToolResultBlock, ...]:
        return tuple(b for b in self.content if isinstance(b, ToolResultBlock))


class ToolDef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...]
    system: str | None = None
    """Stable, cacheable prefix. Must be byte-identical across turns."""

    context: str | None = None
    """Per-turn prompt material (retrieved memory, episode summary).

    Kept out of `system` on purpose: it changes every turn, and burying it in
    the cached prefix would invalidate the cache on every request.
    """

    tools: tuple[ToolDef, ...] = ()
    max_tokens: int = 4096
    temperature: float | None = None
    stop_sequences: tuple[str, ...] = ()
    model: str | None = None
    """Override the provider's configured model. Rarely needed."""

    cache_system: bool = True
    """Mark the system prompt cacheable where the provider supports it.

    Only meaningful if the system prompt is byte-stable across turns, which is
    the packer's job to guarantee.
    """


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str

    @property
    def text(self) -> str:
        return self.message.text

    @property
    def tool_uses(self) -> tuple[ToolUseBlock, ...]:
        return self.message.tool_uses


# --- streaming deltas -------------------------------------------------------


class _Delta(BaseModel):
    model_config = ConfigDict(frozen=True)


class TextDelta(_Delta):
    type: Literal["text"] = "text"
    text: str


class ThinkingDelta(_Delta):
    type: Literal["thinking"] = "thinking"
    thinking: str


class ToolUseStart(_Delta):
    type: Literal["tool_use_start"] = "tool_use_start"
    id: str
    name: str


class ToolUseArgsDelta(_Delta):
    """A fragment of a tool call's JSON arguments.

    Both provider families stream tool arguments as partial JSON, so fragments
    are not individually parseable and must be concatenated per tool-use id.
    """

    type: Literal["tool_use_args"] = "tool_use_args"
    id: str
    partial_json: str


class ToolUseStop(_Delta):
    type: Literal["tool_use_stop"] = "tool_use_stop"
    id: str


class MessageStop(_Delta):
    type: Literal["message_stop"] = "message_stop"
    stop_reason: StopReason
    usage: Usage
    model: str


Delta = Annotated[
    TextDelta | ThinkingDelta | ToolUseStart | ToolUseArgsDelta | ToolUseStop | MessageStop,
    Field(discriminator="type"),
]

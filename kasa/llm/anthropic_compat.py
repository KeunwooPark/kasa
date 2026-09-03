"""Adapter for Anthropic-compatible endpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from kasa.errors import LLMError, ProviderProtocolError
from kasa.llm.base import HTTPProvider, collect
from kasa.llm.types import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    Delta,
    Message,
    MessageStop,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    ToolUseStop,
    Usage,
)

ANTHROPIC_VERSION = "2023-06-01"

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "max_tokens": "max_tokens",
    "tool_use": "tool_use",
    "stop_sequence": "stop_sequence",
    "refusal": "content_filter",
}


class AnthropicCompatProvider(HTTPProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
        name: str = "anthropic",
        client: httpx.AsyncClient | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        headers.update(extra_headers or {})
        super().__init__(name=name, model=model, base_url=base_url, headers=headers, client=client)

    # -- request mapping -----------------------------------------------------

    def _payload(self, req: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model or self.model,
            "messages": [self._message(m) for m in req.messages],
            # Required here, unlike the OpenAI family.
            "max_tokens": req.max_tokens,
        }
        if req.system or req.context:
            payload["system"] = self._system(req)
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.stop_sequences:
            payload["stop_sequences"] = list(req.stop_sequences)
        if req.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in req.tools
            ]
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _system(req: ChatRequest) -> Any:
        if not req.cache_system:
            return "\n\n".join(p for p in (req.system, req.context) if p)

        # The block form is what carries the cache marker. Only the stable
        # prefix gets it; per-turn context goes in a following, uncached block,
        # so retrieval changing every turn does not invalidate the cached prefix.
        blocks: list[dict[str, Any]] = []
        if req.system:
            blocks.append(
                {
                    "type": "text",
                    "text": req.system,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        if req.context:
            blocks.append({"type": "text", "text": req.context})
        return blocks

    @staticmethod
    def _message(msg: Message) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = []
        for block in msg.content:
            match block:
                case TextBlock():
                    blocks.append({"type": "text", "text": block.text})
                case ThinkingBlock():
                    # Only replay thinking that this API produced. A thinking
                    # block without a signature came from another provider's
                    # reasoning field, and echoing it back is a hard 400.
                    if block.signature:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": block.thinking,
                                "signature": block.signature,
                            }
                        )
                case ToolUseBlock():
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        }
                    )
                case ToolResultBlock():
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )
        # This API has no `tool` role; results ride inside a user turn.
        role = "assistant" if msg.role == "assistant" else "user"
        return {"role": role, "content": blocks}

    # -- response mapping ----------------------------------------------------

    @staticmethod
    def _usage(raw: dict[str, Any] | None) -> Usage:
        if not raw:
            return Usage()
        return Usage(
            input_tokens=int(raw.get("input_tokens") or 0),
            output_tokens=int(raw.get("output_tokens") or 0),
            cache_read_tokens=int(raw.get("cache_read_input_tokens") or 0),
            cache_write_tokens=int(raw.get("cache_creation_input_tokens") or 0),
        )

    async def complete(self, req: ChatRequest) -> ChatResponse:
        body = await self._post("/messages", self._payload(req, stream=False))
        blocks: list[ContentBlock] = []
        for raw in body.get("content") or []:
            match raw.get("type"):
                case "text":
                    blocks.append(TextBlock(text=str(raw.get("text") or "")))
                case "thinking":
                    blocks.append(
                        ThinkingBlock(
                            thinking=str(raw.get("thinking") or ""),
                            signature=raw.get("signature"),
                        )
                    )
                case "tool_use":
                    blocks.append(
                        ToolUseBlock(
                            id=str(raw.get("id") or ""),
                            name=str(raw.get("name") or ""),
                            input=dict(raw.get("input") or {}),
                        )
                    )

        return ChatResponse(
            message=Message(role="assistant", content=tuple(blocks)),
            stop_reason=_STOP_REASONS.get(str(body.get("stop_reason")), "end_turn"),
            usage=self._usage(body.get("usage")),
            model=str(body.get("model") or req.model or self.model),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        model = req.model or self.model
        stop: StopReason = "end_turn"
        usage = Usage()
        # Content-block index -> tool-use id, for routing input_json_delta.
        tools: dict[int, str] = {}

        async for event, data in self._post_sse("/messages", self._payload(req, stream=True)):
            kind = event or str(data.get("type") or "")
            match kind:
                case "error":
                    err = data.get("error") or {}
                    raise LLMError(
                        f"stream error: {err.get('message') or data}", provider=self.name
                    )
                case "message_start":
                    message = data.get("message") or {}
                    model = str(message.get("model") or model)
                    usage = self._usage(message.get("usage"))
                case "content_block_start":
                    index = int(data.get("index") or 0)
                    block = data.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        started = str(block.get("id") or f"tool_{index}")
                        tools[index] = started
                        yield ToolUseStart(id=started, name=str(block.get("name") or ""))
                    elif text := block.get("text"):
                        yield TextDelta(text=str(text))
                case "content_block_delta":
                    index = int(data.get("index") or 0)
                    delta = data.get("delta") or {}
                    match delta.get("type"):
                        case "text_delta":
                            yield TextDelta(text=str(delta.get("text") or ""))
                        case "thinking_delta":
                            yield ThinkingDelta(thinking=str(delta.get("thinking") or ""))
                        case "input_json_delta":
                            if (tool_id := tools.get(index)) is not None:
                                yield ToolUseArgsDelta(
                                    id=tool_id, partial_json=str(delta.get("partial_json") or "")
                                )
                case "content_block_stop":
                    index = int(data.get("index") or 0)
                    if (tool_id := tools.get(index)) is not None:
                        yield ToolUseStop(id=tool_id)
                case "message_delta":
                    delta = data.get("delta") or {}
                    if reason := delta.get("stop_reason"):
                        stop = _STOP_REASONS.get(str(reason), "end_turn")
                    # Output tokens are only final here; input tokens stay from
                    # message_start, so merge rather than replace.
                    if raw_usage := data.get("usage"):
                        tail = self._usage(raw_usage)
                        usage = Usage(
                            input_tokens=usage.input_tokens or tail.input_tokens,
                            output_tokens=tail.output_tokens or usage.output_tokens,
                            cache_read_tokens=usage.cache_read_tokens or tail.cache_read_tokens,
                            cache_write_tokens=usage.cache_write_tokens or tail.cache_write_tokens,
                        )

        yield MessageStop(stop_reason=stop, usage=usage, model=model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise ProviderProtocolError(
            "this API has no embeddings endpoint; point the embedding role at an "
            "OpenAI-compatible provider",
            provider=self.name,
        )

    async def complete_via_stream(self, req: ChatRequest) -> ChatResponse:
        return await collect(self.stream(req), model=req.model or self.model, provider=self.name)

"""Adapter for OpenAI-compatible endpoints.

Covers OpenAI itself plus Together, Groq, vLLM, Ollama, OpenRouter and
LM Studio, all of which speak `/chat/completions` with minor deviations.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from kasa.errors import ProviderProtocolError
from kasa.llm.base import HTTPProvider, collect
from kasa.llm.types import (
    ChatRequest,
    ChatResponse,
    Delta,
    Message,
    MessageStop,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    ToolUseStop,
    Usage,
)

_FINISH_REASONS: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "content_filter",
}


class OpenAICompatProvider(HTTPProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        name: str = "openai",
        embedding_dimensions: int | None = None,
        client: httpx.AsyncClient | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        headers.update(extra_headers or {})
        super().__init__(name=name, model=model, base_url=base_url, headers=headers, client=client)
        self._embedding_dimensions = embedding_dimensions

    # -- request mapping -----------------------------------------------------

    def _payload(self, req: ChatRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": req.model or self.model,
            "messages": self._messages(req),
            "max_tokens": req.max_tokens,
        }
        if req.temperature is not None:
            payload["temperature"] = req.temperature
        if req.stop_sequences:
            payload["stop"] = list(req.stop_sequences)
        if req.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in req.tools
            ]
        if stream:
            payload["stream"] = True
            # Without this, most compatible servers omit usage from streams
            # entirely and the cost meter silently records zeros.
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _messages(self, req: ChatRequest) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if req.system or req.context:
            # One system string here, so the stable prefix and the per-turn
            # context concatenate. Prefix first: implicit server-side caching
            # matches on prefixes, so the stable part has to lead.
            parts = [p for p in (req.system, req.context) if p]
            out.append({"role": "system", "content": "\n\n".join(parts)})

        for msg in req.messages:
            if msg.role == "assistant":
                out.append(self._assistant(msg))
                continue

            # Tool results become their own `role: "tool"` messages and must
            # precede any free text in the same turn.
            results = msg.tool_results_in
            for result in results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_use_id,
                        "content": result.content,
                    }
                )
            text = msg.text
            if text or not results:
                out.append({"role": msg.role, "content": text})
        return out

    @staticmethod
    def _assistant(msg: Message) -> dict[str, Any]:
        # Thinking blocks are dropped: this API has nowhere to put them, and
        # round-tripping them as text would corrupt the transcript.
        entry: dict[str, Any] = {"role": "assistant", "content": msg.text or None}
        if tool_uses := msg.tool_uses:
            entry["tool_calls"] = [
                {
                    "id": t.id,
                    "type": "function",
                    "function": {"name": t.name, "arguments": json.dumps(t.input)},
                }
                for t in tool_uses
            ]
        return entry

    # -- response mapping ----------------------------------------------------

    @staticmethod
    def _usage(raw: dict[str, Any] | None) -> Usage:
        if not raw:
            return Usage()
        details = raw.get("prompt_tokens_details") or {}
        return Usage(
            input_tokens=int(raw.get("prompt_tokens") or 0),
            output_tokens=int(raw.get("completion_tokens") or 0),
            cache_read_tokens=int(details.get("cached_tokens") or 0),
        )

    async def complete(self, req: ChatRequest) -> ChatResponse:
        body = await self._post("/chat/completions", self._payload(req, stream=False))
        choices = body.get("choices") or []
        if not choices:
            raise ProviderProtocolError("response contained no choices", provider=self.name)

        raw_msg = choices[0].get("message") or {}
        blocks: list[TextBlock | ThinkingBlock | ToolUseBlock | ToolResultBlock] = []

        # Some reasoning models expose thinking on a non-standard field; keep it
        # if it is there, since dropping it loses audit trail for no gain.
        if reasoning := raw_msg.get("reasoning_content"):
            blocks.append(ThinkingBlock(thinking=str(reasoning)))
        if content := raw_msg.get("content"):
            blocks.append(TextBlock(text=str(content)))
        for call in raw_msg.get("tool_calls") or []:
            fn = call.get("function") or {}
            blocks.append(
                ToolUseBlock(
                    id=str(call.get("id") or ""),
                    name=str(fn.get("name") or ""),
                    input=_loads_arguments(fn.get("arguments"), self.name),
                )
            )

        return ChatResponse(
            message=Message(role="assistant", content=tuple(blocks)),
            stop_reason=_FINISH_REASONS.get(str(choices[0].get("finish_reason")), "end_turn"),
            usage=self._usage(body.get("usage")),
            model=str(body.get("model") or req.model or self.model),
        )

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        model = req.model or self.model
        usage = Usage()
        stop: StopReason = "end_turn"
        # index -> (id, name); compatible servers send the id only on the first
        # fragment of each call, so the index is the only stable key.
        calls: dict[int, tuple[str, str]] = {}

        payload = self._payload(req, stream=True)
        async for _event, chunk in self._post_sse("/chat/completions", payload):
            if raw_usage := chunk.get("usage"):
                usage = self._usage(raw_usage)
            if model_name := chunk.get("model"):
                model = str(model_name)

            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if text := delta.get("content"):
                    yield TextDelta(text=str(text))

                for call in delta.get("tool_calls") or []:
                    index = int(call.get("index") or 0)
                    fn = call.get("function") or {}
                    if index not in calls:
                        call_id = str(call.get("id") or f"call_{index}")
                        calls[index] = (call_id, str(fn.get("name") or ""))
                        yield ToolUseStart(id=call_id, name=calls[index][1])
                    call_id, _ = calls[index]
                    if args := fn.get("arguments"):
                        yield ToolUseArgsDelta(id=call_id, partial_json=str(args))

                if reason := choice.get("finish_reason"):
                    stop = _FINISH_REASONS.get(str(reason), "end_turn")

        for call_id, _ in calls.values():
            yield ToolUseStop(id=call_id)
        yield MessageStop(stop_reason=stop, usage=usage, model=model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        if self._embedding_dimensions:
            payload["dimensions"] = self._embedding_dimensions
        body = await self._post("/embeddings", payload)
        data = body.get("data") or []
        if len(data) != len(texts):
            raise ProviderProtocolError(
                f"asked for {len(texts)} embeddings, got {len(data)}", provider=self.name
            )
        # The API documents index ordering but does not guarantee it on every
        # compatible server, so sort rather than trust.
        ordered = sorted(data, key=lambda d: int(d.get("index") or 0))
        return [[float(x) for x in item["embedding"]] for item in ordered]

    async def complete_via_stream(self, req: ChatRequest) -> ChatResponse:
        return await collect(self.stream(req), model=req.model or self.model, provider=self.name)


def _loads_arguments(raw: Any, provider: str) -> dict[str, Any]:
    if raw in (None, ""):
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ProviderProtocolError(
            f"tool arguments were not valid JSON: {str(raw)[:200]}", provider=provider
        ) from exc
    if not isinstance(parsed, dict):
        raise ProviderProtocolError(
            f"tool arguments were not an object: {str(raw)[:200]}", provider=provider
        )
    return parsed

"""The provider protocol, plus the HTTP and SSE machinery both adapters share."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from kasa.errors import (
    AuthError,
    ContentFilterError,
    ContextOverflowError,
    LLMError,
    ProviderProtocolError,
    RateLimitError,
    TransientError,
)
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
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    ToolUseStop,
    Usage,
)

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

#: Substrings that mean "your request was too long" across the providers and
#: proxies we have seen. Status codes alone do not distinguish this from an
#: ordinary bad request, and the caller needs to tell them apart to recover.
_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "prompt is too long",
    "reduce the length",
    "max_tokens_to_sample",
)


@runtime_checkable
class LLMProvider(Protocol):
    """What the rest of Kasa is allowed to assume about a model backend."""

    name: str
    model: str

    async def complete(self, req: ChatRequest) -> ChatResponse: ...

    def stream(self, req: ChatRequest) -> AsyncIterator[Delta]: ...

    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


class HTTPProvider:
    """Shared transport for HTTP-based providers.

    Deliberately talks to both APIs over plain httpx rather than each vendor's
    SDK. Two reasons: "OpenAI-compatible" servers deviate in small ways that an
    SDK will reject outright, and an SDK's types would leak into signatures that
    are supposed to be vendor-neutral.
    """

    name: str
    model: str

    def __init__(
        self,
        *,
        name: str,
        model: str,
        base_url: str,
        headers: Mapping[str, str],
        timeout: httpx.Timeout | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=dict(headers),
            timeout=timeout or DEFAULT_TIMEOUT,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- transport ----------------------------------------------------------

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=payload)
        except httpx.TimeoutException as exc:
            raise TransientError(f"request timed out: {exc}", provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"request failed: {exc}", provider=self.name) from exc

        if resp.status_code >= 400:
            raise self._error_for(resp.status_code, resp.text, resp.headers)
        try:
            body: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise ProviderProtocolError(
                f"response was not JSON: {resp.text[:200]}", provider=self.name
            ) from exc
        return body

    async def _post_sse(
        self, path: str, payload: dict[str, Any]
    ) -> AsyncIterator[tuple[str | None, dict[str, Any]]]:
        """Yield `(event_name, data)` pairs from a server-sent-event stream.

        `event_name` is None for streams that omit the `event:` line, which is
        how OpenAI-compatible endpoints behave.
        """
        try:
            async with self._client.stream("POST", path, json=payload) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    raise self._error_for(resp.status_code, body, resp.headers)

                event_name: str | None = None
                async for line in resp.aiter_lines():
                    if not line:
                        event_name = None
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        parsed = json.loads(data)
                    except ValueError as exc:
                        raise ProviderProtocolError(
                            f"malformed SSE payload: {data[:200]}", provider=self.name
                        ) from exc
                    yield event_name, parsed
        except httpx.TimeoutException as exc:
            raise TransientError(f"stream timed out: {exc}", provider=self.name) from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"stream failed: {exc}", provider=self.name) from exc

    def _error_for(
        self, status: int, body: str, headers: Mapping[str, str] | None = None
    ) -> LLMError:
        snippet = body[:500]
        lowered = body.lower()

        if status in (401, 403):
            return AuthError(
                f"authentication rejected: {snippet}", provider=self.name, status=status
            )
        if status == 429:
            retry_after: float | None = None
            if headers and (raw := headers.get("retry-after")):
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
            return RateLimitError(
                f"rate limited: {snippet}",
                provider=self.name,
                status=status,
                retry_after=retry_after,
            )
        if any(marker in lowered for marker in _OVERFLOW_MARKERS):
            return ContextOverflowError(
                f"context window exceeded: {snippet}", provider=self.name, status=status
            )
        if "content_filter" in lowered or "content policy" in lowered:
            return ContentFilterError(
                f"content filtered: {snippet}", provider=self.name, status=status
            )
        if status >= 500 or status == 408:
            return TransientError(f"server error: {snippet}", provider=self.name, status=status)
        return LLMError(
            f"request rejected ({status}): {snippet}", provider=self.name, status=status
        )


class StreamAccumulator:
    """Folds a delta stream back into a `ChatResponse`.

    Tool arguments arrive as partial JSON fragments that are not individually
    parseable, so they are buffered per tool-use id and decoded once at stop.
    """

    def __init__(self, model: str, provider: str) -> None:
        self._model = model
        self._provider = provider
        self._blocks: list[ContentBlock] = []
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._pending: dict[str, tuple[str, list[str]]] = {}
        self._order: list[str] = []
        self._stop: StopReason = "end_turn"
        self._usage = Usage()

    def feed(self, delta: Delta) -> None:
        match delta:
            case TextDelta():
                self._text.append(delta.text)
            case ThinkingDelta():
                self._thinking.append(delta.thinking)
            case ToolUseStart():
                self._pending[delta.id] = (delta.name, [])
                self._order.append(delta.id)
            case ToolUseArgsDelta():
                if delta.id in self._pending:
                    self._pending[delta.id][1].append(delta.partial_json)
            case ToolUseStop():
                pass
            case MessageStop():
                self._stop = delta.stop_reason
                self._usage = delta.usage
                if delta.model:
                    self._model = delta.model

    def finish(self) -> ChatResponse:
        if self._thinking:
            self._blocks.append(ThinkingBlock(thinking="".join(self._thinking)))
        if self._text:
            self._blocks.append(TextBlock(text="".join(self._text)))
        for tool_id in self._order:
            name, fragments = self._pending[tool_id]
            raw = "".join(fragments).strip() or "{}"
            try:
                parsed = json.loads(raw)
            except ValueError as exc:
                raise ProviderProtocolError(
                    f"tool {name!r} streamed unparseable arguments: {raw[:200]}",
                    provider=self._provider,
                ) from exc
            if not isinstance(parsed, dict):
                raise ProviderProtocolError(
                    f"tool {name!r} arguments were not an object: {raw[:200]}",
                    provider=self._provider,
                )
            self._blocks.append(ToolUseBlock(id=tool_id, name=name, input=parsed))

        return ChatResponse(
            message=Message(role="assistant", content=tuple(self._blocks)),
            stop_reason=self._stop,
            usage=self._usage,
            model=self._model,
        )


async def collect(stream: AsyncIterator[Delta], *, model: str, provider: str) -> ChatResponse:
    """Drain a delta stream into a single response."""
    acc = StreamAccumulator(model=model, provider=provider)
    async for delta in stream:
        acc.feed(delta)
    return acc.finish()

"""Role-based routing, retry, and fallback.

Kasa never picks "the model". It picks a *role*:

- `chat` dominates spend and is the only role a user waits on
- `utility` dominates call volume (summaries, extraction, classification)
- `embedding` may point at an entirely different vendor from the other two

Each role owns an ordered provider chain. The first entry is the primary; the
rest are fallbacks used when the primary is exhausted.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from kasa.errors import ConfigError, ContentFilterError, ContextOverflowError, LLMError
from kasa.llm.base import LLMProvider
from kasa.llm.cost import CostMeter, PriceBook
from kasa.llm.types import ChatRequest, ChatResponse, Delta, MessageStop, Usage


class ModelRole(StrEnum):
    CHAT = "chat"
    UTILITY = "utility"
    EMBEDDING = "embedding"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 20.0
    jitter: float = 0.25

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        raw = min(self.base_delay * (2**attempt), self.max_delay)
        return float(raw * (1 + random.uniform(-self.jitter, self.jitter)))


Sleeper = Callable[[float], Awaitable[None]]


class ProviderRegistry:
    def __init__(
        self,
        chains: dict[ModelRole, list[LLMProvider]],
        *,
        meter: CostMeter | None = None,
        retry: RetryPolicy | None = None,
        sleep: Sleeper = asyncio.sleep,
    ) -> None:
        self._chains = {role: list(providers) for role, providers in chains.items()}
        self._meter = meter or CostMeter(PriceBook())
        self._retry = retry or RetryPolicy()
        self._sleep = sleep

    # -- lookup --------------------------------------------------------------

    def chain(self, role: ModelRole) -> list[LLMProvider]:
        providers = self._chains.get(role)
        if not providers:
            raise ConfigError(f"no provider configured for role {role.value!r}")
        return providers

    def primary(self, role: ModelRole) -> LLMProvider:
        return self.chain(role)[0]

    @property
    def meter(self) -> CostMeter:
        return self._meter

    def describe(self) -> dict[str, list[str]]:
        return {
            role.value: [f"{p.name}:{p.model}" for p in providers]
            for role, providers in self._chains.items()
        }

    # -- calls ---------------------------------------------------------------

    async def complete(
        self, role: ModelRole, req: ChatRequest, *, tag: str | None = None
    ) -> ChatResponse:
        async def call(provider: LLMProvider) -> ChatResponse:
            started = time.monotonic()
            try:
                resp = await provider.complete(req)
            except LLMError as exc:
                await self._meter.record(
                    role=role.value,
                    provider=provider.name,
                    model=provider.model,
                    usage=Usage(),
                    latency_ms=_ms_since(started),
                    tag=tag,
                    ok=False,
                    error=type(exc).__name__,
                )
                raise
            await self._meter.record(
                role=role.value,
                provider=provider.name,
                model=resp.model,
                usage=resp.usage,
                latency_ms=_ms_since(started),
                tag=tag,
            )
            return resp

        return await self._with_fallback(role, call)

    async def stream(
        self, role: ModelRole, req: ChatRequest, *, tag: str | None = None
    ) -> AsyncIterator[Delta]:
        """Stream from the first provider that gets a delta out of the door.

        Retry and fallback only apply *before the first delta is yielded*. Once
        the caller has seen output there is no way to un-emit it, so a mid-stream
        failure propagates rather than silently restarting on another provider
        and duplicating text.
        """
        providers = self.chain(role)
        last: LLMError | None = None

        for provider in providers:
            for attempt in range(self._retry.attempts):
                started = time.monotonic()
                emitted = False
                try:
                    async for delta in provider.stream(req):
                        emitted = True
                        if isinstance(delta, MessageStop):
                            await self._meter.record(
                                role=role.value,
                                provider=provider.name,
                                model=delta.model,
                                usage=delta.usage,
                                latency_ms=_ms_since(started),
                                tag=tag,
                            )
                        yield delta
                    return
                except LLMError as exc:
                    await self._meter.record(
                        role=role.value,
                        provider=provider.name,
                        model=provider.model,
                        usage=Usage(),
                        latency_ms=_ms_since(started),
                        tag=tag,
                        ok=False,
                        error=type(exc).__name__,
                    )
                    if emitted or _is_terminal(exc):
                        raise
                    last = exc
                    if exc.retryable and attempt < self._retry.attempts - 1:
                        await self._sleep(self._retry.delay_for(attempt, _retry_after(exc)))
                        continue
                    break

        raise last or ConfigError(f"no provider produced a stream for role {role.value!r}")

    async def embed(
        self, texts: list[str], *, role: ModelRole = ModelRole.EMBEDDING
    ) -> list[list[float]]:
        async def call(provider: LLMProvider) -> list[list[float]]:
            return await provider.embed(texts)

        return await self._with_fallback(role, call)

    async def aclose(self) -> None:
        seen: set[int] = set()
        for providers in self._chains.values():
            for provider in providers:
                if id(provider) not in seen:
                    seen.add(id(provider))
                    await provider.aclose()

    # -- policy --------------------------------------------------------------

    async def _with_fallback[T](
        self, role: ModelRole, call: Callable[[LLMProvider], Awaitable[T]]
    ) -> T:
        providers = self.chain(role)
        last: LLMError | None = None

        for provider in providers:
            for attempt in range(self._retry.attempts):
                try:
                    return await call(provider)
                except LLMError as exc:
                    if _is_terminal(exc):
                        raise
                    last = exc
                    if exc.retryable and attempt < self._retry.attempts - 1:
                        await self._sleep(self._retry.delay_for(attempt, _retry_after(exc)))
                        continue
                    break  # not retryable, or attempts exhausted: next provider

        assert last is not None  # chain() guarantees at least one provider ran
        raise last


def _is_terminal(exc: LLMError) -> bool:
    """Errors where trying somewhere else cannot help.

    An overflowing request overflows everywhere, and a content refusal is a
    property of the request, not the vendor. Both need the caller to change
    something, so failing fast beats burning the whole fallback chain.
    """
    return isinstance(exc, ContextOverflowError | ContentFilterError)


def _retry_after(exc: LLMError) -> float | None:
    return getattr(exc, "retry_after", None)


def _ms_since(started: float) -> int:
    return int((time.monotonic() - started) * 1000)

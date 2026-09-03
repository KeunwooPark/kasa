from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from kasa.errors import AuthError, ContextOverflowError, LLMError, TransientError
from kasa.llm.cost import CallRecord, CostMeter, Price, PriceBook
from kasa.llm.registry import ModelRole, ProviderRegistry, RetryPolicy
from kasa.llm.types import (
    ChatRequest,
    ChatResponse,
    Delta,
    Message,
    MessageStop,
    TextDelta,
    Usage,
)

REQUEST = ChatRequest(messages=(Message.user("hi"),))


def ok(text: str = "hi", model: str = "m") -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(text),
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=2),
        model=model,
    )


class FakeProvider:
    """Replays a script of responses and failures."""

    def __init__(self, name: str, script: list[object], *, model: str = "m") -> None:
        self.name = name
        self.model = model
        self.script = list(script)
        self.calls = 0
        self.closed = False

    def _next(self) -> object:
        self.calls += 1
        item = self.script.pop(0) if self.script else self.script
        if isinstance(item, BaseException):
            raise item
        return item

    async def complete(self, req: ChatRequest) -> ChatResponse:
        result = self._next()
        assert isinstance(result, ChatResponse)
        return result

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        item = self.script.pop(0) if self.script else None
        self.calls += 1
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, list):
            # A partial stream: emit deltas, then fail. Models the case where a
            # provider dies after the caller has already seen output.
            for delta in item[:-1]:
                yield delta
            raise item[-1]
        assert isinstance(item, ChatResponse)
        yield TextDelta(text=item.text)
        yield MessageStop(stop_reason="end_turn", usage=item.usage, model=item.model)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._next()
        assert isinstance(result, list)
        return result

    async def aclose(self) -> None:
        self.closed = True


async def no_sleep(_seconds: float) -> None:
    return None


def registry(*providers: FakeProvider, attempts: int = 3, meter: CostMeter | None = None):
    return ProviderRegistry(
        {ModelRole.CHAT: list(providers), ModelRole.EMBEDDING: list(providers)},
        retry=RetryPolicy(attempts=attempts, base_delay=0),
        sleep=no_sleep,
        meter=meter,
    )


async def test_retries_a_transient_failure_on_the_same_provider() -> None:
    provider = FakeProvider("p1", [TransientError("blip"), ok("recovered")])
    resp = await registry(provider).complete(ModelRole.CHAT, REQUEST)

    assert resp.text == "recovered"
    assert provider.calls == 2


async def test_gives_up_and_falls_back_after_exhausting_attempts() -> None:
    primary = FakeProvider("p1", [TransientError("down")] * 3)
    backup = FakeProvider("p2", [ok("from backup")])

    resp = await registry(primary, backup, attempts=3).complete(ModelRole.CHAT, REQUEST)

    assert resp.text == "from backup"
    assert primary.calls == 3
    assert backup.calls == 1


async def test_non_retryable_failure_moves_on_immediately() -> None:
    """A rejected key will still be rejected on the next attempt."""
    primary = FakeProvider("p1", [AuthError("bad key")])
    backup = FakeProvider("p2", [ok("from backup")])

    resp = await registry(primary, backup).complete(ModelRole.CHAT, REQUEST)

    assert resp.text == "from backup"
    assert primary.calls == 1


async def test_terminal_errors_do_not_burn_the_fallback_chain() -> None:
    """An overflowing request overflows everywhere; failing fast is the point."""
    primary = FakeProvider("p1", [ContextOverflowError("too long")])
    backup = FakeProvider("p2", [ok("never reached")])

    with pytest.raises(ContextOverflowError):
        await registry(primary, backup).complete(ModelRole.CHAT, REQUEST)

    assert backup.calls == 0


async def test_last_error_propagates_when_every_provider_fails() -> None:
    primary = FakeProvider("p1", [TransientError("a")] * 3)
    backup = FakeProvider("p2", [TransientError("b")] * 3)

    with pytest.raises(LLMError, match="b"):
        await registry(primary, backup).complete(ModelRole.CHAT, REQUEST)


async def test_stream_falls_back_before_the_first_delta() -> None:
    primary = FakeProvider("p1", [TransientError("dead")] * 3)
    backup = FakeProvider("p2", [ok("streamed")])

    chunks = [d async for d in registry(primary, backup).stream(ModelRole.CHAT, REQUEST)]

    assert any(isinstance(c, TextDelta) and c.text == "streamed" for c in chunks)


async def test_stream_failing_mid_flight_does_not_restart() -> None:
    """Deltas already yielded cannot be un-yielded.

    Silently retrying on another provider would duplicate the visible answer,
    so a mid-stream failure has to propagate.
    """
    primary = FakeProvider("p1", [[TextDelta(text="par"), TransientError("cut off")]])
    backup = FakeProvider("p2", [ok("would duplicate")])
    reg = registry(primary, backup)

    seen: list[Delta] = []
    with pytest.raises(TransientError):
        async for delta in reg.stream(ModelRole.CHAT, REQUEST):
            seen.append(delta)

    assert [d.text for d in seen if isinstance(d, TextDelta)] == ["par"]
    assert backup.calls == 0


async def test_records_usage_and_cost() -> None:
    records: list[CallRecord] = []

    async def sink(record: CallRecord) -> None:
        records.append(record)

    meter = CostMeter(PriceBook({"m": Price(input=1.0, output=2.0)}), sink=sink)
    await registry(FakeProvider("p1", [ok()]), meter=meter).complete(
        ModelRole.CHAT, REQUEST, tag="test"
    )

    assert len(records) == 1
    assert records[0].tag == "test"
    assert records[0].usage.input_tokens == 10
    assert records[0].cost_usd == pytest.approx((10 * 1.0 + 2 * 2.0) / 1_000_000)


async def test_failed_calls_are_recorded_too() -> None:
    records: list[CallRecord] = []

    async def sink(record: CallRecord) -> None:
        records.append(record)

    meter = CostMeter(PriceBook(), sink=sink)
    with pytest.raises(AuthError):
        await registry(FakeProvider("p1", [AuthError("nope")]), meter=meter).complete(
            ModelRole.CHAT, REQUEST
        )

    assert [(r.ok, r.error) for r in records] == [(False, "AuthError")]


async def test_unpriced_models_still_record_tokens() -> None:
    """A stale price table is worse than none, so unknown models report no cost."""
    meter = CostMeter(PriceBook())
    await registry(FakeProvider("p1", [ok()]), meter=meter).complete(ModelRole.CHAT, REQUEST)

    assert meter.total.input_tokens == 10
    assert meter.total_usd == 0.0


async def test_price_lookup_prefers_the_longest_matching_prefix() -> None:
    book = PriceBook({"claude": Price(input=1.0), "claude-opus": Price(input=9.0)})
    assert book.cost_usd("claude-opus-5-20260101", Usage(input_tokens=1_000_000)) == 9.0


async def test_aclose_closes_each_provider_once() -> None:
    shared = FakeProvider("p1", [])
    reg = ProviderRegistry({ModelRole.CHAT: [shared], ModelRole.UTILITY: [shared]})
    await reg.aclose()
    assert shared.closed

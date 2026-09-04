"""Token and spend accounting.

Prices are configuration, never a hardcoded guess: vendor pricing changes far
faster than this repo will, and a stale built-in table produces confidently
wrong numbers. An unpriced model still records tokens; only the USD figure is
`None`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from kasa.llm.types import Usage


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens."""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class PriceBook:
    def __init__(self, prices: dict[str, Price] | None = None) -> None:
        # Keyed by model-name prefix so `claude-opus-5` matches
        # `claude-opus-5-20260101` without a config edit on every point release.
        self._prices = dict(prices or {})

    def lookup(self, model: str) -> Price | None:
        best: tuple[int, Price] | None = None
        for prefix, price in self._prices.items():
            if model.startswith(prefix) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), price)
        return best[1] if best else None

    def cost_usd(self, model: str, usage: Usage) -> float | None:
        price = self.lookup(model)
        if price is None:
            return None
        return (
            usage.input_tokens * price.input
            + usage.output_tokens * price.output
            + usage.cache_read_tokens * price.cache_read
            + usage.cache_write_tokens * price.cache_write
        ) / 1_000_000


@dataclass(frozen=True, slots=True)
class CallRecord:
    role: str
    provider: str
    model: str
    usage: Usage
    latency_ms: int
    cost_usd: float | None
    tag: str | None
    ok: bool
    error: str | None = None
    session_id: str | None = None


#: Where a completed call goes. The store supplies the real one; tests and the
#: CLI can pass a collector.
CostSink = Callable[[CallRecord], Awaitable[None]]
SpendSource = Callable[[str], Awaitable[float]]


async def null_sink(record: CallRecord) -> None:
    return None


class CostMeter:
    def __init__(
        self,
        price_book: PriceBook,
        sink: CostSink = null_sink,
        *,
        daily_usd_ceiling: float | None = None,
        spent_since: SpendSource | None = None,
    ) -> None:
        self._prices = price_book
        self._sink = sink
        self.total = Usage()
        self.total_usd = 0.0
        self._sessions: dict[str, Usage] = {}
        self.daily_usd_ceiling = daily_usd_ceiling
        self._spent_since = spent_since
        self._day = _utc_day()
        self._daily_usd = 0.0

    def session_usage(self, session_id: str) -> Usage:
        return self._sessions.get(session_id, Usage())

    def session_cache_hit_rate(self, session_id: str) -> float:
        return self.session_usage(session_id).cache_hit_rate

    async def daily_ceiling_reached(self) -> bool:
        ceiling = self.daily_usd_ceiling
        if ceiling is None:
            return False
        day = _utc_day()
        if self._spent_since is not None:
            spent = await self._spent_since(day)
        else:
            spent = self._daily_usd if self._day == day else 0.0
        return spent >= ceiling

    async def record(
        self,
        *,
        role: str,
        provider: str,
        model: str,
        usage: Usage,
        latency_ms: int,
        tag: str | None = None,
        ok: bool = True,
        error: str | None = None,
        session_id: str | None = None,
        cost_usd: float | None = None,
    ) -> CallRecord:
        # An explicit cost wins over the token table. Not everything metered
        # here is priced per token — web search is billed per call — and a
        # per-call price looked up by token count would always come back zero.
        cost = cost_usd if cost_usd is not None else self._prices.cost_usd(model, usage)
        self.total = self.total + usage
        if session_id is not None:
            self._sessions[session_id] = self.session_usage(session_id) + usage
        if cost is not None:
            self.total_usd += cost
            day = _utc_day()
            if day != self._day:
                self._day = day
                self._daily_usd = 0.0
            self._daily_usd += cost
        record = CallRecord(
            role=role,
            provider=provider,
            model=model,
            usage=usage,
            latency_ms=latency_ms,
            cost_usd=cost,
            tag=tag,
            ok=ok,
            error=error,
            session_id=session_id,
        )
        await self._sink(record)
        return record


def _utc_day() -> str:
    return datetime.now(UTC).date().isoformat()

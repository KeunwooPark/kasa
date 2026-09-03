"""A five-field cron expression, and when it next fires.

Small on purpose. The schedules this has to express are the ones in
`docs/DESIGN.md` §6 — hourly, nightly, weekly — and a dependency taken on for
that is a dependency to keep current forever.

Times are UTC, because that is what the database stores and what a daemon on a
server runs in. `0 3 * * *` is three in the morning UTC, which is not three in
the morning where you are; say what you mean in UTC.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Self

from kasa.errors import KasaError

#: minute, hour, day-of-month, month, day-of-week — the inclusive range each
#: field may take, in the order they are written.
RANGES = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))

#: How far ahead to look before deciding an expression never fires again. Four
#: years, so `0 0 29 2 *` still finds its leap day.
HORIZON = timedelta(days=366 * 4)


class CronError(KasaError):
    """A cron expression this does not understand."""


@dataclass(frozen=True, slots=True)
class Cron:
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]
    #: Whether day-of-month and day-of-week were *both* narrowed. Cron reads
    #: that as "either day matches", not "both" — `0 0 1 * 1` is the first of
    #: the month and every Monday, which is surprising until you have been
    #: caught by it once.
    either_day: bool
    expression: str

    @classmethod
    def parse(cls, expression: str) -> Self:
        fields = expression.split()
        if len(fields) != 5:
            raise CronError(
                f"{expression!r} has {len(fields)} field(s); a cron expression has five "
                "(minute hour day-of-month month day-of-week)"
            )
        values = [
            _field(text, low, high, expression)
            for text, (low, high) in zip(fields, RANGES, strict=True)
        ]
        return cls(
            minutes=values[0],
            hours=values[1],
            days=values[2],
            months=values[3],
            weekdays=values[4],
            either_day=fields[2] != "*" and fields[4] != "*",
            expression=expression,
        )

    def matches(self, moment: datetime) -> bool:
        return (
            moment.minute in self.minutes
            and moment.hour in self.hours
            and moment.month in self.months
            and self._day_matches(moment)
        )

    def next_after(self, moment: datetime) -> datetime:
        """The first minute strictly after `moment` that this fires on.

        Walks forward, but skips: a month that does not match costs one step,
        not forty-four thousand.
        """
        candidate = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
        limit = candidate + HORIZON
        while candidate < limit:
            if candidate.month not in self.months:
                candidate = _next_month(candidate)
            elif not self._day_matches(candidate):
                candidate = (candidate + timedelta(days=1)).replace(hour=0, minute=0)
            elif candidate.hour not in self.hours:
                candidate = (candidate + timedelta(hours=1)).replace(minute=0)
            elif candidate.minute not in self.minutes:
                candidate += timedelta(minutes=1)
            else:
                return candidate
        raise CronError(f"{self.expression!r} does not fire within four years")

    def _day_matches(self, moment: datetime) -> bool:
        # cron counts weekdays from Sunday; Python counts from Monday.
        by_month = moment.day in self.days
        by_week = moment.isoweekday() % 7 in self.weekdays
        return (by_month or by_week) if self.either_day else (by_month and by_week)


def _next_month(moment: datetime) -> datetime:
    start = moment.replace(day=1, hour=0, minute=0)
    return (start + timedelta(days=32)).replace(day=1, hour=0, minute=0)


def _field(text: str, low: int, high: int, expression: str) -> frozenset[int]:
    values: set[int] = set()
    for part in text.split(","):
        values |= _part(part, low, high, expression)
    if not values:
        raise CronError(f"{text!r} in {expression!r} matches nothing")
    return frozenset(values)


def _part(part: str, low: int, high: int, expression: str) -> set[int]:
    spec, _, step_text = part.partition("/")
    try:
        step = int(step_text) if step_text else 1
    except ValueError:
        raise CronError(f"{part!r} in {expression!r} has a step that is not a number") from None
    if step < 1:
        raise CronError(f"{part!r} in {expression!r} has a step below 1")

    if spec == "*":
        start, stop = low, high
    elif "-" in spec:
        start_text, _, stop_text = spec.partition("-")
        start, stop = _number(start_text, part, expression), _number(stop_text, part, expression)
    else:
        start = stop = _number(spec, part, expression)
        # `5/15` is not a range, so a step on a bare number is a typo for
        # `5-59/15` or for `*/15`, and guessing which would be worse.
        if step_text:
            raise CronError(f"{part!r} in {expression!r} steps from a single value")

    if not (low <= start <= high and low <= stop <= high):
        raise CronError(f"{part!r} in {expression!r} is outside {low}-{high}")
    if start > stop:
        raise CronError(f"{part!r} in {expression!r} counts backwards")
    return set(range(start, stop + 1, step))


def _number(text: str, part: str, expression: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise CronError(f"{part!r} in {expression!r} is not a number") from None


#: The three the design actually asks for (§6), named so a registration reads
#: as its intent rather than as five fields somebody has to decode.
HOURLY = "0 * * * *"
NIGHTLY = "0 3 * * *"
WEEKLY = "0 4 * * 0"

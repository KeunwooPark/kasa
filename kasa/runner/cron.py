"""A five-field cron expression, and when it next fires.

Small on purpose. The schedules this has to express are the ones in
`docs/DESIGN.md` §6 — hourly, nightly, weekly — and a dependency taken on for
that is a dependency to keep current forever.

Times are UTC unless an expression is given a zone. That default is the right
one for the jobs this was written for: nobody asks what time zone `reindex`
runs in, and `0 3 * * *` in a registration means three in the morning on the
server. It stops being the right one the moment a *person* names a time.
"Nine every morning" is nine where they are, it moves twice a year, and nobody
should have to do the offset arithmetic to say it — so `parse` takes an
optional IANA zone, the walk happens on that zone's wall clock, and only the
answer is converted back. The boundary is unchanged either way: an aware UTC
instant in, an aware UTC instant out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
    #:
    #: "Narrowed" is decided the way Vixie decides it: a field *beginning* with
    #: `*` is unrestricted, whatever follows. So `*/2` in day-of-month keeps
    #: the relation AND, and `0 0 */2 * 1` fires on Mondays that fall on an odd
    #: day rather than on every other day *or* every Monday. Reading it as
    #: restricted, which the obvious `!= "*"` does, yields a much larger set.
    either_day: bool
    expression: str
    #: The wall clock the fields are read against. None is UTC, and is what
    #: every schedule that ships with Kasa uses.
    tz: ZoneInfo | None = None

    @classmethod
    def parse(cls, expression: str, *, tz: str | ZoneInfo | None = None) -> Self:
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
            either_day=not fields[2].startswith("*") and not fields[4].startswith("*"),
            expression=expression,
            tz=_zone(tz),
        )

    @property
    def label(self) -> str:
        """How to name this in a log line or a listing.

        Five fields alone are ambiguous once a zone is possible, and the reader
        of "could not schedule 0 9 * * 1-5" has no way to ask which nine.
        """
        return self.expression if self.tz is None else f"{self.expression} ({self.tz.key})"

    def matches(self, moment: datetime) -> bool:
        local = self._wall(moment)
        return (
            local.minute in self.minutes
            and local.hour in self.hours
            and local.month in self.months
            and self._day_matches(local)
        )

    def next_after(self, moment: datetime) -> datetime:
        """The first minute strictly after `moment` that this fires on.

        Walks forward, but skips: a month that does not match costs one step,
        not forty-four thousand. With a zone, the walk is over that zone's wall
        clock — which is the whole point, since nine in the morning stays nine
        in the morning on both sides of a DST boundary — and each candidate is
        turned back into an instant before it is offered.
        """
        candidate = self._wall(moment).replace(second=0, microsecond=0) + timedelta(minutes=1)
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
            elif (fire_at := self._instant(candidate, after=moment)) is not None:
                return fire_at
            else:
                candidate += timedelta(minutes=1)
        raise CronError(f"{self.label!r} does not fire within four years")

    def _day_matches(self, moment: datetime) -> bool:
        # cron counts weekdays from Sunday; Python counts from Monday.
        by_month = moment.day in self.days
        by_week = moment.isoweekday() % 7 in self.weekdays
        return (by_month or by_week) if self.either_day else (by_month and by_week)

    def _wall(self, moment: datetime) -> datetime:
        """`moment` as the naive wall clock the fields are read against.

        Naive on purpose: the walk adds days and hours, and adding a day to an
        aware datetime across a DST boundary moves the wall clock by 23 or 25
        hours, which is exactly the arithmetic a schedule must not do.
        """
        if self.tz is None:
            return moment.replace(tzinfo=None)
        if moment.tzinfo is None:
            raise CronError(
                f"{self.label!r} counts from an instant, and a naive datetime does not name one"
            )
        return moment.astimezone(self.tz).replace(tzinfo=None)

    def _instant(self, local: datetime, *, after: datetime) -> datetime | None:
        """The UTC instant `local` names, or None if the walk should go on.

        None on two counts, and the caller treats them the same way — advance a
        minute and keep looking:

        * The instant is not after `after`. Only reachable in the hour a
          fall-back repeats: the clock ticks in the second pass over a wall
          time that already fired in the first, and `next_after` promises an
          instant that is strictly later than the one it was handed. Skipping
          is also what makes a repeated wall time fire *once* rather than
          twice.
        * The wall time is inside a spring-forward gap and has no far side
          within the gap — which cannot happen, but is not worth asserting.

        Without a zone there is nothing to resolve and nothing to skip: the
        walk started a minute after `after` and every step moves forward, so
        the candidate is already the answer, in the frame it was asked in.
        """
        if self.tz is None:
            return local.replace(tzinfo=after.tzinfo)
        # fold=0 is the earlier of the two passes over a repeated wall time,
        # which is the one that fires.
        fire_at = _to_utc(local, self.tz) or _after_gap(local, self.tz)
        return fire_at if fire_at is not None and fire_at > after else None


def _to_utc(local: datetime, tz: ZoneInfo) -> datetime | None:
    """The instant a wall time names, or None if it names none.

    A wall time inside a spring-forward gap still *converts* — PEP 495 gives it
    the offset from before the transition — so the check is the round trip: an
    hour that does not exist comes back as a different hour.
    """
    instant = local.replace(tzinfo=tz).astimezone(UTC)
    return instant if instant.astimezone(tz).replace(tzinfo=None) == local else None


def _after_gap(local: datetime, tz: ZoneInfo) -> datetime | None:
    """The first wall time on the far side of the gap that swallowed `local`.

    A schedule whose hour a zone skips still fires that day: `0 2 * * *` where
    02:00 does not exist runs at 03:00, not tomorrow. Walking a minute at a
    time is bounded by the gap itself — an hour nearly everywhere, and a whole
    day only for Apia in 2011.
    """
    ahead = local.replace(tzinfo=tz, fold=1).utcoffset()
    behind = local.replace(tzinfo=tz).utcoffset()
    if ahead is None or behind is None or ahead <= behind:  # pragma: no cover - not a gap
        return None
    probe, edge = local, local + (ahead - behind)
    while probe < edge:
        probe += timedelta(minutes=1)
        if (instant := _to_utc(probe, tz)) is not None:
            return instant
    return None


def _zone(tz: str | ZoneInfo | None) -> ZoneInfo | None:
    if tz is None or isinstance(tz, ZoneInfo):
        return tz
    try:
        return ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise CronError(
            f"{tz!r} is not a time zone this machine knows. Names come from the IANA "
            "database — 'Asia/Seoul', not 'KST' — and a platform that ships without one "
            "needs the 'tzdata' package installed."
        ) from exc


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

"""Five fields, and when they next fire.

UTC unless a test says otherwise; the zone-aware half is at the bottom.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from kasa.runner.cron import HOURLY, NIGHTLY, WEEKLY, Cron, CronError

NOW = datetime(2026, 9, 3, 10, 30, tzinfo=UTC)  # a Thursday


def fires(expression: str, after: datetime = NOW) -> datetime:
    return Cron.parse(expression).next_after(after)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (HOURLY, datetime(2026, 9, 3, 11, 0, tzinfo=UTC)),
        (NIGHTLY, datetime(2026, 9, 4, 3, 0, tzinfo=UTC)),
        (WEEKLY, datetime(2026, 9, 6, 4, 0, tzinfo=UTC)),  # the next Sunday
        ("*/15 * * * *", datetime(2026, 9, 3, 10, 45, tzinfo=UTC)),
        ("0,30 * * * *", datetime(2026, 9, 3, 11, 0, tzinfo=UTC)),
        ("35 10 * * *", datetime(2026, 9, 3, 10, 35, tzinfo=UTC)),
        ("0 9 * * 1-5", datetime(2026, 9, 4, 9, 0, tzinfo=UTC)),
        ("0 0 1 * *", datetime(2026, 10, 1, 0, 0, tzinfo=UTC)),
        ("0 0 29 2 *", datetime(2028, 2, 29, 0, 0, tzinfo=UTC)),
    ],
)
def test_the_schedules_this_has_to_express(expression: str, expected: datetime) -> None:
    assert fires(expression) == expected


def test_the_next_fire_is_strictly_after_the_moment_given() -> None:
    """Otherwise a scheduler that ticks on the minute re-queues the occurrence
    it has just run, forever."""
    on_the_hour = datetime(2026, 9, 3, 11, 0, tzinfo=UTC)

    assert fires(HOURLY, on_the_hour) == datetime(2026, 9, 3, 12, 0, tzinfo=UTC)


def test_seconds_do_not_delay_the_next_fire() -> None:
    assert fires(HOURLY, NOW.replace(minute=59, second=59)) == datetime(
        2026, 9, 3, 11, 0, tzinfo=UTC
    )


def test_a_restricted_day_of_month_and_weekday_mean_either() -> None:
    """Cron's oldest surprise: `0 0 1 * 1` is the first of the month *or* any
    Monday, not the first of a month that is a Monday."""
    both = Cron.parse("0 0 1 * 1")

    assert both.next_after(NOW) == datetime(2026, 9, 7, 0, 0, tzinfo=UTC)  # the Monday
    assert both.matches(datetime(2026, 10, 1, 0, 0, tzinfo=UTC))  # a Thursday the 1st


def test_only_one_of_them_restricted_is_not_either() -> None:
    assert fires("0 0 4 * *") == datetime(2026, 9, 4, 0, 0, tzinfo=UTC)


def test_a_stepped_day_field_is_unrestricted_and_keeps_the_relation_and() -> None:
    """Vixie sets its star flags on any field *beginning* with `*`, so `*/2` in
    day-of-month does not flip the relation. Reading it as restricted turns
    "every other day that is a Monday" into "every other day, or any Monday",
    which is a much larger set."""
    stepped = Cron.parse("0 0 */2 * 1")

    assert stepped.either_day is False
    # Mondays in September 2026 are the 7th, 14th, 21st and 28th. Only the odd
    # ones survive `*/2`.
    assert stepped.next_after(NOW) == datetime(2026, 9, 7, 0, 0, tzinfo=UTC)
    assert not stepped.matches(datetime(2026, 9, 14, 0, 0, tzinfo=UTC)), "a Monday, even day"
    assert not stepped.matches(datetime(2026, 9, 5, 0, 0, tzinfo=UTC)), "an odd day, Saturday"
    assert stepped.matches(datetime(2026, 9, 21, 0, 0, tzinfo=UTC)), "a Monday, odd day"


def test_a_stepped_weekday_is_unrestricted_too() -> None:
    """The same rule on the other side of the relation."""
    assert Cron.parse("0 0 1 * */2").either_day is False


def test_a_range_on_a_day_field_is_still_restricted() -> None:
    """Only a leading `*` is a star. `1-7` narrows, and the relation flips."""
    assert Cron.parse("0 0 1-7 * 1").either_day is True


def test_a_step_inside_a_range() -> None:
    assert Cron.parse("0-30/10 * * * *").minutes == frozenset({0, 10, 20, 30})


@pytest.mark.parametrize(
    ("expression", "complaint"),
    [
        ("0 0 * *", "has 4 field"),
        ("0 0 * * * *", "has 6 field"),
        ("60 * * * *", "outside 0-59"),
        ("0 24 * * *", "outside 0-23"),
        ("30-10 * * * *", "counts backwards"),
        ("banana * * * *", "not a number"),
        ("*/0 * * * *", "step below 1"),
        ("5/15 * * * *", "steps from a single value"),
    ],
)
def test_an_expression_that_will_not_do_says_why(expression: str, complaint: str) -> None:
    """A cron expression is written once and read at three in the morning."""
    with pytest.raises(CronError) as caught:
        Cron.parse(expression)
    assert complaint in str(caught.value)


# -- zones -------------------------------------------------------------------
#
# Nine in the morning is nine in the morning where the person who said it
# lives. Everything below is that sentence, held against the four ways a zone
# makes it hard: an offset, a half-offset, an offset that changed once, and the
# two hours a year that do not behave.

SEOUL = ZoneInfo("Asia/Seoul")


def local(zone: str, stamp: str) -> datetime:
    return datetime.fromisoformat(stamp).replace(tzinfo=ZoneInfo(zone))


def utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp).replace(tzinfo=UTC)


def test_a_zone_moves_the_fire_time_not_the_expression() -> None:
    """`0 9 * * 1-5` in Seoul is nine in Seoul, which is midnight here."""
    weekdays = Cron.parse("0 9 * * 1-5", tz=SEOUL)

    assert weekdays.next_after(NOW) == utc("2026-09-04 00:00")  # the Friday


def test_a_zone_name_is_enough() -> None:
    assert Cron.parse("0 9 * * *", tz="Asia/Seoul").tz == SEOUL


def test_no_zone_is_utc_and_unchanged() -> None:
    """The six jobs that ship with Kasa pass no zone and must not move."""
    assert Cron.parse(NIGHTLY).tz is None
    assert fires(NIGHTLY) == Cron.parse(NIGHTLY, tz="UTC").next_after(NOW)


def test_a_half_hour_offset() -> None:
    assert Cron.parse("0 9 * * *", tz="Asia/Kolkata").next_after(NOW) == utc("2026-09-04 03:30")


def test_an_offset_that_changed_once() -> None:
    """Moscow left UTC+4 for good on 26 October 2014. Nine in the morning did
    not move; the instant it names did."""
    moscow = Cron.parse("0 9 * * *", tz="Europe/Moscow")

    assert moscow.next_after(utc("2014-10-24 12:00")) == utc("2014-10-25 05:00")
    assert moscow.next_after(utc("2014-10-26 12:00")) == utc("2014-10-27 06:00")


def test_the_answer_is_an_instant_whatever_the_zone() -> None:
    """What the jobs table stores, and what `scheduled_id` is derived from."""
    fire_at = Cron.parse("0 9 * * *", tz=SEOUL).next_after(NOW)

    assert fire_at.tzinfo is UTC
    assert fire_at.astimezone(SEOUL) == local("Asia/Seoul", "2026-09-04 09:00")


@pytest.mark.parametrize(
    ("zone", "start", "expected"),
    [
        # 02:00 does not exist on 8 March 2026 in New York, nor on 4 October in
        # Sydney. The schedule fires anyway, at the first wall time on the far
        # side of the gap — 03:00 — rather than skipping the day.
        ("America/New_York", "2026-03-07 12:00", "2026-03-08 07:00"),
        ("Australia/Sydney", "2026-10-03 05:00", "2026-10-03 16:00"),
    ],
)
def test_an_hour_a_zone_skips_still_fires_that_day(zone: str, start: str, expected: str) -> None:
    fire_at = Cron.parse("0 2 * * *", tz=zone).next_after(utc(start))

    assert fire_at == utc(expected)
    assert fire_at.astimezone(ZoneInfo(zone)).hour == 3


@pytest.mark.parametrize(
    ("zone", "expression", "before", "first", "next_day"),
    [
        # 01:00 happens twice on 1 November 2026 in New York; 02:00 twice on
        # 5 April in Sydney. Once, on the first pass — a daily digest posted
        # twice because the clocks went back is a bug nobody can reproduce for
        # another year.
        (
            "America/New_York",
            "0 1 * * *",
            "2026-10-31 12:00",
            "2026-11-01 05:00",
            "2026-11-02 06:00",
        ),
        (
            "Australia/Sydney",
            "0 2 * * *",
            "2026-04-04 05:00",
            "2026-04-04 15:00",
            "2026-04-05 16:00",
        ),
    ],
)
def test_an_hour_a_zone_repeats_fires_once(
    zone: str, expression: str, before: str, first: str, next_day: str
) -> None:
    daily = Cron.parse(expression, tz=zone)

    assert daily.next_after(utc(before)) == utc(first)
    assert daily.next_after(utc(first)) == utc(next_day), "the second pass is not a second fire"


def test_a_tick_inside_the_repeated_hour_does_not_look_backwards() -> None:
    """The clock ticks every thirty seconds, so it ticks in the hour that
    happens twice — and an occurrence dated before the tick that produced it is
    one the scheduler has already run."""
    daily = Cron.parse("0 1 * * *", tz="America/New_York")
    inside = utc("2026-11-01 06:15")  # 01:15 for the second time

    assert daily.next_after(inside) == utc("2026-11-02 06:00")


def test_a_zoned_expression_names_its_zone_when_something_goes_wrong() -> None:
    assert Cron.parse("0 9 * * 1-5", tz=SEOUL).label == "0 9 * * 1-5 (Asia/Seoul)"
    assert Cron.parse("0 9 * * 1-5").label == "0 9 * * 1-5"


def test_a_zone_this_machine_does_not_know_says_so() -> None:
    """'KST' is what a person types, and it is not an IANA name."""
    with pytest.raises(CronError) as caught:
        Cron.parse("0 9 * * *", tz="KST")
    assert "Asia/Seoul" in str(caught.value)


def test_a_zoned_expression_counts_from_an_instant() -> None:
    with pytest.raises(CronError) as caught:
        Cron.parse("0 9 * * *", tz=SEOUL).next_after(datetime(2026, 9, 3, 10, 30))
    assert "does not name one" in str(caught.value)

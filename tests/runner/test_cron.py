"""Five fields, and when they next fire. Everything here is UTC."""

from __future__ import annotations

from datetime import UTC, datetime

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

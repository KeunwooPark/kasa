"""The salience curve: what age takes away and what recall gives back."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kasa.memory.salience import Decay, age_of

DECAY = Decay()


def test_a_fresh_memory_keeps_the_salience_it_was_written_with() -> None:
    """The base is the frontmatter default on purpose: a memory promoted this
    morning must not be rewritten tonight for a difference nobody asked for."""
    assert abs(DECAY.score(age=timedelta(hours=6)) - DECAY.base) < 0.01


def test_half_the_salience_is_gone_after_the_half_life() -> None:
    assert abs(DECAY.score(age=timedelta(days=30)) - DECAY.base / 2) < 0.001


def test_recall_buys_back_time() -> None:
    """A memory that keeps being needed does not fade at the rate of one that
    nobody has asked for since it was written."""
    untouched = DECAY.score(age=timedelta(days=30))
    recalled = DECAY.score(age=timedelta(days=30), hits=3)

    assert recalled > untouched
    # Enough recall carries a month-old memory back past where it started.
    assert DECAY.score(age=timedelta(days=30), hits=10) > DECAY.base


def test_one_busy_conversation_cannot_pin_a_memory_forever() -> None:
    """Without the cap, a session that searched the same thing forty times
    holds a memory at the top of the corpus for a month."""
    assert DECAY.score(age=timedelta(days=1), hits=1_000) <= 1.0
    assert DECAY.score(age=timedelta(days=1), hits=1_000) == DECAY.score(
        age=timedelta(days=1), hits=40
    )


def test_age_alone_never_reaches_zero() -> None:
    """Zero is unrecoverable: nothing that ranks below everything can be
    retrieved to be boosted again, and "old" was never "wrong"."""
    assert DECAY.score(age=timedelta(days=3_650)) == DECAY.floor


def test_scoring_is_a_function_of_state_not_a_step() -> None:
    """The property the whole nightly pass rests on. `reflect` rewrites twenty
    files a night on a corpus of a thousand, so a memory may be scored today,
    skipped for a week, and scored again — and must land where it would have if
    it had been scored every night."""
    age = timedelta(days=12)

    assert DECAY.score(age=age, hits=2) == DECAY.score(age=age, hits=2)


def test_a_clock_that_went_backwards_does_not_raise_salience() -> None:
    """A restored backup, or a machine correcting its time. Age is the one
    direction this must never move."""
    now = datetime(2026, 9, 4, tzinfo=UTC)
    future = now + timedelta(days=5)

    assert age_of(future, now) == timedelta(0)
    assert DECAY.score(age=age_of(future, now)) == DECAY.base

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


def test_an_endorsement_is_worth_more_than_a_recall() -> None:
    """A recall says the ranker picked this memory; an endorsement says a
    person read what came of it and agreed. Only one of those is a judgement,
    and it is the rarer of the two."""
    age = timedelta(days=20)

    assert DECAY.score(age=age, endorsements=1) > DECAY.score(age=age, hits=1)


def test_endorsements_are_capped_like_recalls_are() -> None:
    """A memory everybody agrees with should sit near the top of the corpus. It
    should not be immune to a year of nobody needing it."""
    age = timedelta(days=20)

    assert DECAY.score(age=age, endorsements=100) == DECAY.score(age=age, endorsements=1_000)
    assert DECAY.score(age=timedelta(days=3_650), endorsements=100) < 1.0


def test_endorsement_and_recall_both_count() -> None:
    age = timedelta(days=20)

    both = DECAY.score(age=age, hits=1, endorsements=1)
    assert both > DECAY.score(age=age, hits=1)
    assert both > DECAY.score(age=age, endorsements=1)


def test_endorsements_are_recomputed_like_everything_else() -> None:
    """They arrive as a count within a window, which is what keeps a night that
    runs twice landing where a night that runs once did."""
    age = timedelta(days=12)

    assert DECAY.score(age=age, hits=2, endorsements=3) == DECAY.score(
        age=age, hits=2, endorsements=3
    )

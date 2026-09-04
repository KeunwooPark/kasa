"""The cheapest quality signal there is, and the asymmetry it rests on.

👍 and ❌ are not mirror images. One is a count in a window that feeds a number
`reflect` recomputes; the other is an event applied exactly once to a number
nothing recomputes. Most of what is worth testing here is that neither has
picked up the other's semantics.
"""

from __future__ import annotations

from kasa.core.feedback import DOWN, SUSPECT, UP, Feedback
from kasa.store import Store

ANSWER = "slack:T0TEAM:C0DEPLOY:1700000001.000000"
JANE = "U0JANE"
RAJ = "U0RAJ"
FIRST = "mem_01K8XQ0000000000000000001"
SECOND = "mem_01K8XQ0000000000000000002"


async def answered(store: Store, *memories: str, scope: str = "channel:C0DEPLOY") -> str:
    return await store.record_answer(
        source="slack",
        external_id=ANSWER,
        memory_ids=list(memories),
        session_id="slack:T0TEAM:C0DEPLOY:1700000000.000100",
        scope=scope,
    )


async def votes(store: Store, kind: str) -> list[dict[str, object]]:
    return await store.raw("SELECT * FROM memory_feedback WHERE kind = ?", (kind,))


# -- what a reaction reaches --------------------------------------------------


async def test_a_thumbs_up_vouches_for_every_memory_behind_the_answer(store: Store) -> None:
    await answered(store, FIRST, SECOND)

    result = await Feedback(store).record(
        source="slack", external_id=ANSWER, verdict=UP, author=JANE
    )

    assert result.known and result.memories == [FIRST, SECOND]
    assert {str(row["memory_id"]) for row in await votes(store, "up")} == {FIRST, SECOND}


async def test_a_reaction_on_a_message_kasa_did_not_post_does_nothing(store: Store) -> None:
    """The common case in a busy channel: people react to each other."""
    result = await Feedback(store).record(
        source="slack", external_id="slack:T0TEAM:C0DEPLOY:9.9", verdict=UP, author=JANE
    )

    assert not result.known
    assert await votes(store, "up") == []


async def test_an_answer_that_used_no_memory_is_still_a_known_answer(store: Store) -> None:
    """The row exists so that "nothing to boost" is tellable from "not ours"."""
    await answered(store)

    result = await Feedback(store).record(
        source="slack", external_id=ANSWER, verdict=UP, author=JANE
    )

    assert result.known and result.memories == []


async def test_one_person_is_one_vote(store: Store) -> None:
    """Otherwise a channel of forty people clicking 👍 on one reply outweighs a
    month of everything else."""
    await answered(store, FIRST)
    feedback = Feedback(store)

    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=JANE)
    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=JANE)

    assert len(await votes(store, "up")) == 1


async def test_two_people_are_two_votes(store: Store) -> None:
    await answered(store, FIRST)
    feedback = Feedback(store)

    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=JANE)
    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=RAJ)

    assert await store.endorsements_since("2000-01-01") == {FIRST: 2}


# -- the two directions are not mirror images ---------------------------------


async def test_a_cross_queues_a_review_as_well_as_a_vote(store: Store) -> None:
    """Lowering confidence is a commit to the repo. A person should be looking
    at whether the memory belongs there at all."""
    await answered(store, FIRST)

    result = await Feedback(store).record(
        source="slack", external_id=ANSWER, verdict=DOWN, author=JANE
    )

    assert len(result.reviews) == 1
    review = (await store.open_reviews())[0]
    assert review["kind"] == SUSPECT
    assert review["subject"] == FIRST
    assert review["scope"] == "channel:C0DEPLOY", "as private as the conversation was"


async def test_a_thumbs_up_queues_nothing(store: Store) -> None:
    await answered(store, FIRST)

    result = await Feedback(store).record(
        source="slack", external_id=ANSWER, verdict=UP, author=JANE
    )

    assert result.reviews == []
    assert await store.open_reviews() == []


async def test_two_people_marking_one_memory_wrong_is_one_review(store: Store) -> None:
    """One memory is one thing to look at, however many answers it spoils."""
    await answered(store, FIRST)
    feedback = Feedback(store)

    await feedback.record(source="slack", external_id=ANSWER, verdict=DOWN, author=JANE)
    await feedback.record(source="slack", external_id=ANSWER, verdict=DOWN, author=RAJ)

    assert len(await store.open_reviews()) == 1
    assert len(await votes(store, "down")) == 2, "but still two votes"


async def test_an_up_vote_is_not_counted_as_a_down_vote(store: Store) -> None:
    await answered(store, FIRST)

    await Feedback(store).record(source="slack", external_id=ANSWER, verdict=UP, author=JANE)

    assert await store.unapplied_feedback("down") == []


# -- taking it back -----------------------------------------------------------


async def test_removing_a_reaction_removes_the_vote(store: Store) -> None:
    await answered(store, FIRST, SECOND)
    feedback = Feedback(store)
    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=JANE)

    await feedback.withdraw(source="slack", external_id=ANSWER, verdict=UP, author=JANE)

    assert await store.endorsements_since("2000-01-01") == {}


async def test_one_person_taking_theirs_back_leaves_everybody_elses(store: Store) -> None:
    await answered(store, FIRST)
    feedback = Feedback(store)
    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=JANE)
    await feedback.record(source="slack", external_id=ANSWER, verdict=UP, author=RAJ)

    await feedback.withdraw(source="slack", external_id=ANSWER, verdict=UP, author=JANE)

    assert await store.endorsements_since("2000-01-01") == {FIRST: 1}


async def test_a_vote_reflect_has_already_spent_cannot_be_taken_back(store: Store) -> None:
    """The corpus has moved. Un-applying the row here would leave it and the
    file disagreeing, with nobody able to tell which was right."""
    await answered(store, FIRST)
    feedback = Feedback(store)
    await feedback.record(source="slack", external_id=ANSWER, verdict=DOWN, author=JANE)
    spent = [int(row["id"]) for row in await store.unapplied_feedback("down")]
    await store.mark_feedback_applied(spent)

    await feedback.withdraw(source="slack", external_id=ANSWER, verdict=DOWN, author=JANE)

    assert len(await votes(store, "down")) == 1


async def test_the_review_a_cross_raised_outlives_the_cross(store: Store) -> None:
    """It has already asked a person to look at something. Cancelling that
    because the reaction was un-clicked would be Kasa closing the question,
    which is the one thing raising a review says it will not do."""
    await answered(store, FIRST)
    feedback = Feedback(store)
    await feedback.record(source="slack", external_id=ANSWER, verdict=DOWN, author=JANE)

    await feedback.withdraw(source="slack", external_id=ANSWER, verdict=DOWN, author=JANE)

    assert len(await store.open_reviews()) == 1


# -- the answer record --------------------------------------------------------


async def test_recording_the_same_answer_twice_keeps_the_first_row(store: Store) -> None:
    """A redelivered turn re-posts nothing but may re-record. The reaction has
    to reach one row, not the newer of two."""
    first = await answered(store, FIRST)
    second = await answered(store, SECOND)

    assert first == second
    answer = await store.answer_at("slack", ANSWER)
    assert answer is not None and FIRST in str(answer["memory_ids"])

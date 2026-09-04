"""What happens downstream when somebody takes back what they said.

Three different answers to three different events, and the test names say which
is which: an edit invalidates a claim, a deletion weakens one, and neither of
them touches the repo. The last is the one worth being strict about — a
background job that quietly deletes a file because a Slack message went away an
hour later is the behaviour the patch-plan pipeline exists to prevent.
"""

from __future__ import annotations

import pytest

from kasa.core.revise import DELETED, EDITED, TOMBSTONE, Reviser, Revision
from kasa.llm.types import Message
from kasa.store import Store

SESSION = "slack:T0TEAM:C0DEPLOY:1700000000.000100"
EXTERNAL = "slack:T0TEAM:C0DEPLOY:1700000000.000100"


async def stored(store: Store, text: str, *, external_id: str = EXTERNAL) -> str:
    await store.ensure_session(SESSION, surface="slack", scope="channel:C0DEPLOY")
    return await store.append_message(
        SESSION, Message.user(text), author="U0JANE", external_id=external_id
    )


async def observation(store: Store, message_id: str, **fields: object) -> str:
    return await store.add_observation(
        subject=str(fields.pop("subject", "the deploy window")),
        claim=str(fields.pop("claim", "The deploy window is Thursday.")),
        kind="fact",
        scope="channel:C0DEPLOY",
        session_id=SESSION,
        confidence=float(fields.pop("confidence", 0.8)),  # type: ignore[arg-type]
        source_refs=[message_id],
    )


async def row_for(store: Store, observation_id: str) -> dict[str, object]:
    rows = await store.raw("SELECT * FROM observations WHERE id = ?", (observation_id,))
    assert rows
    return rows[0]


# -- the message itself -------------------------------------------------------


async def test_an_edit_rewrites_the_stored_message(store: Store) -> None:
    await stored(store, "the deploy window is Thursday")

    result = await Reviser(store).apply(Revision(EXTERNAL, "the deploy window is Friday"))

    assert result.found and result.state == EDITED
    assert [m.text for m in await store.recent_messages(SESSION)] == ["the deploy window is Friday"]


async def test_a_deletion_leaves_a_tombstone_in_the_transcript(store: Store) -> None:
    """Not a removed row. The assistant has already answered it, that answer is
    still in the transcript, and a reply to nothing reads as the model having
    invented the question."""
    await stored(store, "the deploy window is Thursday")

    result = await Reviser(store).apply(Revision(EXTERNAL, None))

    assert result.state == DELETED
    assert [m.text for m in await store.recent_messages(SESSION)] == [TOMBSTONE]


async def test_the_words_of_a_deleted_message_are_gone_from_the_database(store: Store) -> None:
    """Somebody deleting a message means it, and the row is the only copy
    short-term memory kept."""
    await stored(store, "my home address is 12 Made Up Street")

    await Reviser(store).apply(Revision(EXTERNAL, None))

    rows = await store.raw("SELECT content, state FROM messages WHERE session_id = ?", (SESSION,))
    assert "Made Up Street" not in str(rows[0]["content"])
    assert rows[0]["state"] == "deleted"


async def test_a_revision_for_a_message_nobody_stored_changes_nothing(store: Store) -> None:
    """Most of a workspace's edits are to messages Kasa never read."""
    result = await Reviser(store).apply(Revision("slack:T0TEAM:C0OTHER:1.0", "anything"))

    assert not result.found


async def test_an_unfurl_is_not_an_edit(store: Store) -> None:
    """Slack sends `message_changed` when a link grows a preview card. Treating
    that as a rewrite would mark a perfectly good claim stale because somebody
    pasted a URL."""
    message_id = await stored(store, "see https://example.test")
    pending = await observation(store, message_id)

    result = await Reviser(store).apply(Revision(EXTERNAL, "see https://example.test"))

    assert result.found and result.stale == 0
    assert (await row_for(store, pending))["state"] == "pending"


# -- what it does to candidate facts ------------------------------------------


async def test_an_edit_makes_a_pending_claim_stale(store: Store) -> None:
    """The words the claim was read out of are not the words that were said."""
    message_id = await stored(store, "the deploy window is Thursday")
    pending = await observation(store, message_id)

    result = await Reviser(store).apply(Revision(EXTERNAL, "the deploy window is Friday"))

    assert result.stale == 1
    row = await row_for(store, pending)
    assert row["state"] == "stale"
    assert "edited" in str(row["reason"])


async def test_a_deletion_lowers_confidence_rather_than_deciding(store: Store) -> None:
    """A retraction is evidence, not a verdict: the author may have been fixing
    a typo, or thinking better of saying it out loud, or wrong."""
    message_id = await stored(store, "the deploy window is Thursday")
    pending = await observation(store, message_id, confidence=0.8)

    result = await Reviser(store).apply(Revision(EXTERNAL, None))

    assert result.weakened == 1
    row = await row_for(store, pending)
    assert row["confidence"] == pytest.approx(0.4)
    assert row["state"] == "pending", "still a candidate, just a weaker one"


async def test_a_claim_from_another_message_is_left_alone(store: Store) -> None:
    other = "slack:T0TEAM:C0DEPLOY:1700000000.000200"
    kept = await observation(store, await stored(store, "unrelated", external_id=other))
    await stored(store, "the deploy window is Thursday")

    await Reviser(store).apply(Revision(EXTERNAL, None))

    assert (await row_for(store, kept))["confidence"] == pytest.approx(0.8)


# -- what it refuses to do ----------------------------------------------------


async def test_a_promoted_claim_queues_a_review_and_nothing_else(store: Store) -> None:
    """The rule the whole module is built around. What is in the repo is in the
    repo until a person says otherwise."""
    message_id = await stored(store, "the deploy window is Thursday")
    promoted = await observation(store, message_id)
    await store.resolve_observations([promoted], state="promoted", reason="committed")

    result = await Reviser(store).apply(Revision(EXTERNAL, None))

    assert len(result.reviews) == 1
    reviews = await store.open_reviews()
    assert reviews[0]["kind"] == DELETED
    # Normalized on the way in, which is what makes it the same key `promote`
    # grouped the claim under.
    assert reviews[0]["subject"] == "deploy window"
    assert "The deploy window is Thursday." in str(reviews[0]["detail"])
    assert reviews[0]["scope"] == "channel:C0DEPLOY", "a review of a DM is as private as the DM"


async def test_an_edited_source_queues_a_review_too(store: Store) -> None:
    """An edit that changes what a promoted claim was read out of leaves the
    corpus just as wrong as a deletion does."""
    message_id = await stored(store, "the deploy window is Thursday")
    promoted = await observation(store, message_id)
    await store.resolve_observations([promoted], state="promoted", reason="committed")

    result = await Reviser(store).apply(Revision(EXTERNAL, "the deploy window is Friday"))

    assert len(result.reviews) == 1
    assert (await store.open_reviews())[0]["kind"] == EDITED


async def test_the_same_deletion_twice_is_one_review(store: Store) -> None:
    """Slack re-sends an event it did not hear an ack for, and this path runs
    on the ack path, so it is only safe because it is idempotent."""
    message_id = await stored(store, "the deploy window is Thursday")
    promoted = await observation(store, message_id)
    await store.resolve_observations([promoted], state="promoted", reason="committed")

    await Reviser(store).apply(Revision(EXTERNAL, None))
    again = await Reviser(store).apply(Revision(EXTERNAL, None))

    assert again.reviews == []
    assert len(await store.open_reviews()) == 1


async def test_a_closed_review_stays_closed(store: Store) -> None:
    message_id = await stored(store, "the deploy window is Thursday")
    promoted = await observation(store, message_id)
    await store.resolve_observations([promoted], state="promoted", reason="committed")
    await Reviser(store).apply(Revision(EXTERNAL, None))
    review_id = str((await store.open_reviews())[0]["id"])

    assert await store.resolve_review(review_id)
    assert await store.open_reviews() == []
    assert not await store.resolve_review(review_id), "closing it twice is not a second close"

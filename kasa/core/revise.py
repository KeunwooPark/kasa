"""When the conversation changes under a memory that was built from it.

Short-term memory is a copy of a transcript, and a copy is only honest while
the original still says the same thing. Slack lets anybody edit or delete what
they said an hour ago, and a system that remembers on purpose has to have an
answer for that beyond keeping the old words.

Three things happen to a revised message, and they are deliberately different
from each other:

**The message itself** is rewritten in place — new words for an edit, a
tombstone for a deletion. In place rather than removed: the assistant has
already answered it, that answer is still in the transcript, and a reply to a
message that is no longer there reads as the model having invented the
question.

**Candidate facts drawn from it** are treated by what the change means. An edit
says the words this claim was read out of are not the words that were said, so
the claim is stale — it was never promoted, and it should not be. A deletion is
weaker evidence than that: somebody retracting a message may be correcting a
typo, may have thought better of saying it out loud, or may simply have
finished with it. So a deletion lowers confidence and lets `promote` weigh the
claim against everything else that was said, rather than settling it here.

**Anything already in the repo** is left exactly as it is, and a review is
queued instead. A background job that quietly deletes a file because somebody
removed a Slack message an hour later is the behaviour the whole patch-plan
pipeline exists to prevent: the memory may have been merged, rewritten or built
on since, and a retraction is not a correction. Kasa says what it noticed and
stops.

Surface-agnostic on purpose. Slack is what motivates it, and every surface that
lets a person take something back needs the same three answers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from kasa.llm.types import Message
from kasa.store import Store
from kasa.store.db import message_from_row

log = logging.getLogger(__name__)

#: What a deleted message says in the transcript from now on. Written as the
#: person's own message because that is whose turn it occupies — the sequence
#: has to keep working, and what changed is the content, not who spoke.
TOMBSTONE = "[this message was deleted by its author]"

#: What a retraction does to the confidence of a claim drawn from it. Halving
#: rather than zeroing: this is evidence, not a verdict. A claim that several
#: messages support survives one of them being taken back, which is the right
#: outcome and not one a boolean can express.
WEAKEN_BY = 0.5

EDITED = "source_edited"
DELETED = "source_deleted"


@dataclass(frozen=True, slots=True)
class Revision:
    """What a surface says happened to a message it already delivered.

    `external_id` is the surface's own key for the message — the same one the
    inbox dedupes on, which is what lets this find the row without knowing how
    the message was stored. `text` of None means the message is gone.
    """

    external_id: str
    text: str | None

    @property
    def deleted(self) -> bool:
        return self.text is None


@dataclass(slots=True)
class Revised:
    """What a revision actually changed."""

    #: False when the message was never stored: a revision in a channel Kasa
    #: does not read, or one older than the database. The common case, and not
    #: a problem — most of a workspace's edits are to messages nobody kept.
    found: bool = False
    state: str = ""
    stale: int = 0
    weakened: int = 0
    #: Reviews raised, meaning claims already in the corpus whose source moved.
    reviews: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.found:
            return "no stored message for that revision"
        parts = [self.state]
        if self.stale:
            parts.append(f"{self.stale} observation(s) stale")
        if self.weakened:
            parts.append(f"{self.weakened} observation(s) weakened")
        if self.reviews:
            parts.append(f"{len(self.reviews)} review(s) queued")
        return ", ".join(parts)


class Reviser:
    """Applies one revision to everything downstream of the message."""

    def __init__(self, store: Store, *, weaken_by: float = WEAKEN_BY) -> None:
        self._store = store
        self._weaken_by = weaken_by

    async def apply(self, revision: Revision) -> Revised:
        row = await self._store.message_by_external_id(revision.external_id)
        if row is None:
            return Revised()

        message_id = str(row["id"])
        state = DELETED if revision.deleted else EDITED
        content = Message.user(TOMBSTONE if revision.deleted else revision.text or "")
        if not revision.deleted and content.text == message_from_row(row).text:
            # Slack sends `message_changed` for things that are not edits — an
            # unfurl attaching itself, a reaction on a file. Treating those as
            # a rewrite would mark a perfectly good claim stale because
            # somebody's link got a preview card.
            return Revised(found=True, state=str(row["state"]))

        await self._store.revise_message(
            message_id, content=content, state="deleted" if revision.deleted else "edited"
        )

        observations = await self._store.observations_from(message_id)
        pending = [o for o in observations if str(o["state"]) == "pending"]
        outcome = Revised(found=True, state=state)

        if revision.deleted:
            outcome.weakened = await self._store.weaken_observations(
                [str(o["id"]) for o in pending],
                factor=self._weaken_by,
                reason="the message it came from was deleted",
            )
        else:
            outcome.stale = await self._store.resolve_observations(
                [str(o["id"]) for o in pending],
                state="stale",
                reason="the message it came from was edited",
            )

        for observation in observations:
            if str(observation["state"]) != "promoted":
                continue
            if (review := await self._queue(observation, state)) is not None:
                outcome.reviews.append(review)

        log.info("revision %s: %s", revision.external_id, outcome.summary())
        return outcome

    async def _queue(self, observation: dict[str, object], kind: str) -> str | None:
        """Ask a person to look at a memory whose source moved.

        Keyed on the observation rather than the message: one claim is one
        thing to check, and a message that is edited twice is still that.
        """
        what = "deleted" if kind == DELETED else "edited"
        return await self._store.queue_review(
            kind=kind,
            key=str(observation["id"]),
            subject=str(observation["subject"]),
            detail=(
                f"This was promoted to long-term memory, and the message it came from "
                f"has since been {what}. The claim was: {observation['claim']}"
            ),
            refs=[str(observation["id"])],
            scope=str(observation["scope"]),
        )

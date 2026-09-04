"""What somebody thought of an answer, and what long-term memory does with it.

The cheapest quality signal there is. A 👍 costs one click and says the thing
no amount of retrieval tuning can work out on its own — that what was recalled
was the right thing to recall. An ❌ says the opposite, and a corpus that
nobody corrects has no other way to find out that one of its files is wrong.

It only works because the answer is still connected to the memories behind it
when the reaction arrives, which may be days later and is certainly after the
process that produced it has forgotten everything. `answers` is that record,
and this module is what reads it back.

The two directions are not mirror images, and the asymmetry is the point.

**Up is a count in a window.** It feeds `salience`, which is recomputed nightly
from age, recall and endorsement rather than adjusted — the property that lets
a bounded pass converge instead of double-counting (`kasa/memory/salience.py`).
Nothing here has to be applied exactly once, so nothing here tracks whether it
was.

**Down is an event, applied once.** It lowers `confidence`, which is not
derived from anything: a number a model set, that nothing recomputes. Treating
one ❌ as a nightly subtraction would walk it to zero inside a fortnight. And
because lowering it is a commit to the repo, an ❌ also raises a review — the
same reasoning as #25's deletions, that a background job quietly rewriting what
a person wrote is what the patch-plan pipeline exists to prevent. The
difference is that here a person is deliberately signalling, so the confidence
change goes ahead through `reflect`'s validated commit; the review is what
makes sure somebody looks at whether the memory should be there at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from kasa.store import Store

log = logging.getLogger(__name__)

Verdict = Literal["up", "down"]

UP: Verdict = "up"
DOWN: Verdict = "down"

#: What an ❌ does to a memory's confidence, once. Multiplicative and gentle:
#: one person disagreeing with one answer is a reason to trust a memory less,
#: not a reason to stop believing it — the memory may have been right and the
#: answer wrong about which memory to use.
SUSPECT_FACTOR = 0.7

#: The review an ❌ raises, in `reviews.kind`.
SUSPECT = "memory_suspect"


@dataclass(slots=True)
class Recorded:
    """What one reaction actually did."""

    #: False when the message reacted to is not an answer Kasa recorded — a
    #: reaction on somebody else's message, or on a reply from before this
    #: build. The common case in a busy channel, and not a problem.
    known: bool = False
    verdict: str = ""
    memories: list[str] = field(default_factory=list)
    reviews: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.known:
            return "a reaction on something that is not one of Kasa's answers"
        if not self.memories:
            return f"{self.verdict} on an answer that used no memory"
        parts = [f"{self.verdict} on {len(self.memories)} memory(s)"]
        if self.reviews:
            parts.append(f"{len(self.reviews)} review(s) queued")
        return ", ".join(parts)


class Feedback:
    """Applies one person's verdict to the memories behind one answer."""

    def __init__(self, store: Store) -> None:
        self._store = store

    async def record(
        self, *, source: str, external_id: str, verdict: Verdict, author: str = ""
    ) -> Recorded:
        """Take a verdict on the answer posted at `external_id`."""
        answer = await self._store.answer_at(source, external_id)
        if answer is None:
            return Recorded()

        memories = _memory_ids(answer)
        outcome = Recorded(known=True, verdict=verdict, memories=memories)
        for memory_id in memories:
            fresh = await self._store.add_memory_feedback(
                memory_id=memory_id,
                kind=verdict,
                answer_id=str(answer["id"]),
                author=author,
            )
            if not fresh or verdict != DOWN:
                continue
            if (review := await self._review(memory_id, answer)) is not None:
                outcome.reviews.append(review)
        log.info("feedback on %s: %s", external_id, outcome.summary())
        return outcome

    async def withdraw(
        self, *, source: str, external_id: str, verdict: Verdict, author: str = ""
    ) -> Recorded:
        """Take a verdict back, if `reflect` has not spent it yet.

        The review an ❌ raised is deliberately left standing. It has already
        asked a person to look at something, and cancelling that because the
        reaction was un-clicked would be Kasa deciding the question is closed —
        which is the one thing raising a review says it will not do.
        """
        answer = await self._store.answer_at(source, external_id)
        if answer is None:
            return Recorded()
        dropped = await self._store.drop_memory_feedback(
            answer_id=str(answer["id"]), author=author, kind=verdict
        )
        return Recorded(
            known=True, verdict=verdict, memories=_memory_ids(answer) if dropped else []
        )

    async def _review(self, memory_id: str, answer: dict[str, object]) -> str | None:
        return await self._store.queue_review(
            kind=SUSPECT,
            # One memory is one thing to look at, however many answers it goes
            # on to get an ❌ on.
            key=memory_id,
            subject=memory_id,
            detail=(
                "Somebody marked an answer that used this memory as wrong. Its "
                "confidence has been lowered; check whether the memory itself is "
                "wrong, out of date, or was simply the wrong one to recall."
            ),
            refs=[memory_id],
            scope=str(answer["scope"]),
        )


def _memory_ids(answer: dict[str, object]) -> list[str]:
    raw = answer.get("memory_ids")
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except ValueError:
        return []
    return [str(one) for one in parsed] if isinstance(parsed, list) else []

"""What a memory's importance is, given its age and how often it was needed.

`salience` is the number `forget` will eventually read to decide what stops
being worth keeping, so the arithmetic that sets it is worth having in one
place that can be read and tested away from the job that schedules it.

**It is recomputed, not adjusted.** `score` is a pure function of how old a
memory is and how often it was recalled in the recent window — running it twice
gives the same answer, and running it after a week's gap gives the same answer
as running it every night. That property is what makes a bounded nightly pass
correct: `reflect` can rewrite twenty files a night on a corpus of a thousand
and reach the same place, eventually, that rewriting all of them would have.

The alternative — decaying the value that is already there — needs a record of
when each memory was last decayed, or it double-counts every time the pass is
interrupted, runs twice, or skips a file because the commit was full.

Two forces, deliberately asymmetric. Time takes importance away from
everything and nothing has to happen for it to; only evidence gives it back.
There are two kinds of evidence and they are not worth the same: a recall says
the ranker picked this memory, and an endorsement (#36) says a person read what
came of it and said it was right. The second is rarer, harder to fake, and the
only input to this number that somebody chose to give.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class Decay:
    """The curve, and what a recall is worth against it."""

    #: Where an untouched memory starts. The same as `Frontmatter.salience`'s
    #: default, so a memory written today is not rewritten tonight for a
    #: difference nobody asked about.
    base: float = 0.5
    #: How long a memory nobody needs takes to lose half its salience. A month,
    #: so something that mattered in April is still findable in May and has
    #: faded by the autumn.
    half_life_days: float = 30.0
    #: Added per recall in the window. One is worth about ten days of decay at
    #: the default half-life, so a memory retrieved a couple of times a month
    #: holds its place and one retrieved daily climbs.
    per_hit: float = 0.08
    #: The most recalls may add, however busy the window was. Without it, one
    #: conversation that searched the same thing forty times pins a memory at
    #: the top of the corpus for a month.
    max_boost: float = 0.3
    #: Added per person who put a 👍 on an answer this memory produced (#36).
    #: Worth more than twice a recall, because it is: a recall says the ranker
    #: picked this memory, and an endorsement says a person read what came of
    #: it and agreed. It is the one input to salience that somebody chose.
    per_endorsement: float = 0.2
    #: And the cap on that, for the same reason `max_boost` exists. A memory
    #: everybody agrees with should sit near the top of the corpus; it should
    #: not be immune to a year of nobody needing it.
    max_endorsement_boost: float = 0.4
    #: Age alone never drives salience to zero. Zero is unrecoverable — nothing
    #: that ranks below everything can be retrieved to be boosted again — and
    #: "old" was never the same claim as "wrong".
    floor: float = 0.05

    def score(self, *, age: timedelta, hits: int = 0, endorsements: int = 0) -> float:
        """The salience this memory should have, given its age and its evidence.

        Still a pure function of its inputs, which is the property the whole
        module rests on: endorsements arrive as a count within the same window
        recalls are counted in, so a night that runs twice reaches the same
        answer as a night that runs once.
        """
        days = max(age.total_seconds(), 0.0) / 86_400.0
        decayed = self.base * math.pow(0.5, days / self.half_life_days)
        earned = (
            min(hits, _cap(self.max_boost, self.per_hit)) * self.per_hit
            + min(endorsements, _cap(self.max_endorsement_boost, self.per_endorsement))
            * self.per_endorsement
        )
        return _clamp(max(decayed + earned, self.floor))


def age_of(updated: datetime, now: datetime) -> timedelta:
    """How long since a memory last changed, never negative.

    A clock that went backwards — a restored backup, a machine correcting its
    time — would otherwise *raise* salience across the whole corpus, which is
    the one direction age must never move it.
    """
    return max(now - updated, timedelta(0))


def _cap(ceiling: float, per: float) -> float:
    """How many of a thing worth `per` each it takes to reach `ceiling`."""
    return ceiling / per if per else 0.0


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)

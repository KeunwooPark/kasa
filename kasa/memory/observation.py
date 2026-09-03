"""A candidate fact, before anything durable has happened to it.

An observation is the unit that crosses from short-term to long-term memory:
`episode_close` extracts them from a conversation, `promote` reconciles them
against the corpus and turns them into a patch plan. Between those two it is a
row in SQLite that nobody has committed to believing yet.

The draft type is here rather than in the store because two callers build one —
the `memory_write` tool and the extractor — and both have to agree on what the
fields mean. `scope` in particular: it is inherited from the session, never
chosen, because an observation made in a DM must not become workspace knowledge
by way of a model deciding it would be more useful there.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, get_args

#: What an observation can be. `fact` is the default shape; the others are
#: separate because `promote` has to treat them differently — a `task` ages
#: out, a `decision` supersedes rather than merges.
ObservationKind = Literal["fact", "preference", "decision", "task", "relation"]

#: The same set as a tuple, for the places that need values rather than a type:
#: a JSON Schema `enum`, and a validity check on a tool argument. Derived from
#: the Literal so the two cannot drift.
OBSERVATION_KINDS: tuple[str, ...] = get_args(ObservationKind)


@dataclass(frozen=True, slots=True)
class ObservationDraft:
    """One candidate fact, ready to be written to `observations`."""

    subject: str
    claim: str
    #: One of `OBSERVATION_KINDS`, checked by whoever built the draft. Not the
    #: `ObservationKind` Literal: both builders start from a plain string — a
    #: tool argument, or a model's JSON — and a `cast` at each of them would
    #: assert exactly what the check beside it has just established.
    kind: str
    scope: str
    confidence: float = 0.7
    source_refs: Sequence[str] = field(default_factory=tuple)

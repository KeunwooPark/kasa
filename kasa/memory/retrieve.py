"""Lexical retrieval: query, candidates, fusion, scope filter, pack.

Hybrid retrieval arrives in #31. This is the lexical half, and it has to be
good enough to be useful on its own, because a memory system that only works
once embeddings are configured is a memory system that does not work.

It has one known blind spot, and it is the ordinary one for lexical search:
derivational morphology. Porter stemming turns "deploying" into "deploy" but
leaves "deployments" alone, so a memory about the "deploy pipeline" does not
answer "who handles deployments?". Inflection is handled; word formation is not.
That gap is the case for #31 rather than for a bigger stopword list.

Two things here are not merely engineering preferences:

**The scope filter runs in SQL, before ranking.** Not after fusion, not during
packing. A memory the requester may not see must never enter the ranked pool at
all, because every later step is a place somebody could forget to re-check. The
one exception is `explain=True`, which re-runs the query unfiltered purely to
report what was excluded — and marks those rows so they cannot be packed.

**Every step is recorded.** Retrieval you cannot debug is retrieval you cannot
improve, and essentially every complaint about this system will arrive as "why
did it not remember X". `kasa why` prints what this module recorded.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from kasa.llm.tokens import Tokenizer
from kasa.store import Store

Row = dict[str, Any]

#: Reciprocal-rank-fusion constant. 60 is the value from the original paper and
#: is not worth tuning until there is a second list to fuse with.
RRF_K = 60

#: How long it takes a memory's recency boost to halve. Long, because long-term
#: memory is supposed to be long-term; recency breaks ties, it does not rank.
RECENCY_HALF_LIFE_DAYS = 120.0

#: How many chunks each candidate source contributes before fusion.
SOURCE_LIMIT = 30

DEFAULT_LIMIT = 8

#: A message shorter than this, or one that opens with an anaphor, is not a
#: standalone query — it needs the conversation around it.
_SELF_CONTAINED_MIN_WORDS = 4

_ANAPHORA = frozenset(
    {
        "it",
        "that",
        "this",
        "they",
        "them",
        "those",
        "these",
        "he",
        "she",
        "him",
        "her",
        "his",
        "hers",
        "there",
        "then",
    }
)

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)

_WORD = re.compile(r"[A-Za-z0-9_'-]+")

#: Rewrites a message plus recent turns into a standalone query. Supplied by the
#: caller so the utility model stays out of this module; None means the
#: heuristic below.
Rewriter = Callable[[str, Sequence[str]], Awaitable[str]]


@dataclass(frozen=True, slots=True)
class Candidate:
    memory_id: str
    path: str
    chunk_id: str
    ordinal: int
    text: str
    scope: str
    salience: float
    pinned: bool
    updated_at: str

    #: Where it came from, and how highly each source ranked it.
    lexical_rank: int | None = None
    header_rank: int | None = None
    bm25: float | None = None

    fused: float = 0.0
    recency: float = 1.0
    final: float = 0.0
    #: Set only on candidates surfaced by `explain`, which are never packed.
    denied: str | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        found = []
        if self.lexical_rank is not None:
            found.append("bm25")
        if self.header_rank is not None:
            found.append("title/tag")
        if self.pinned:
            found.append("pinned")
        return tuple(found)


@dataclass(slots=True)
class RetrievalTrace:
    question: str
    query: str
    rewritten: bool
    scope: str
    match_expression: str
    candidates: list[Candidate] = field(default_factory=list)
    denied: list[Candidate] = field(default_factory=list)
    kept: list[Candidate] = field(default_factory=list)
    budget_tokens: int = 0
    used_tokens: int = 0


@dataclass(slots=True)
class Retrieval:
    snippets: list[str] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    trace: RetrievalTrace | None = None

    @property
    def memory_ids(self) -> list[str]:
        return [c.memory_id for c in (self.trace.kept if self.trace else [])]


def permits(requester_scope: str, memory_scope: str) -> bool:
    """Whether a session in `requester_scope` may see a memory in `memory_scope`.

    Workspace memories are visible everywhere; anything narrower is visible only
    from inside exactly that scope. The number-one failure mode of a shared
    memory is repeating something from a DM in a public channel, so this errs
    towards refusing.
    """
    return memory_scope == "workspace" or memory_scope == requester_scope


def is_self_contained(message: str) -> bool:
    """Whether a message can be used as a retrieval query without rewriting.

    "What did Jane say about the deploy pipeline?" is. "What about it?" is not,
    and rewriting it costs a model call that the first case does not need.
    """
    words = _WORD.findall(message.lower())
    if len(words) < _SELF_CONTAINED_MIN_WORDS:
        return False
    if not [w for w in words if w not in _STOPWORDS]:
        return False
    return not (set(words[:3]) & _ANAPHORA)


def build_match(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression.

    Every term is quoted and OR-ed. Quoting matters for correctness, not
    tidiness: an unquoted apostrophe or a bare `NEAR` in someone's message is a
    syntax error inside FTS5, and a syntax error here is a failed turn.
    """
    terms = [t for t in _WORD.findall(query.lower()) if t not in _STOPWORDS and len(t) > 1]
    seen = list(dict.fromkeys(terms))
    return " OR ".join(f'"{term}"' for term in seen)


class Retriever:
    def __init__(
        self,
        store: Store,
        *,
        tokenizer: Tokenizer,
        budget_tokens: int = 4_000,
        limit: int = DEFAULT_LIMIT,
        rewriter: Rewriter | None = None,
        now: datetime | None = None,
    ) -> None:
        self._store = store
        self._tok = tokenizer
        self._budget = budget_tokens
        self._limit = limit
        self._rewriter = rewriter
        self._now = now

    async def retrieve(
        self,
        question: str,
        *,
        scope: str = "workspace",
        recent: Sequence[str] = (),
        explain: bool = False,
    ) -> Retrieval:
        query, rewritten = await self._build_query(question, recent)
        match = build_match(query)
        trace = RetrievalTrace(
            question=question,
            query=query,
            rewritten=rewritten,
            scope=scope,
            match_expression=match,
            budget_tokens=self._budget,
        )

        candidates = await self._candidates(match, scope)
        pinned = await self._pinned(scope)
        merged = _merge(candidates + pinned)
        for candidate in merged:
            trace.candidates.append(candidate)

        ranked = sorted(merged, key=lambda c: c.final, reverse=True)
        trace.candidates = ranked
        if explain:
            trace.denied = await self._denied(match, scope)

        return self._pack(ranked, trace)

    # -- query ---------------------------------------------------------------

    async def _build_query(self, question: str, recent: Sequence[str]) -> tuple[str, bool]:
        if is_self_contained(question) or not recent:
            return question, False
        if self._rewriter is not None:
            return await self._rewriter(question, recent), True
        # No utility model wired in: carry the salient words of the recent turns
        # so that "what about it?" still has something to search for.
        context_terms = [
            term
            for turn in list(recent)[-3:]
            for term in _WORD.findall(turn.lower())
            if term not in _STOPWORDS and len(term) > 2
        ]
        return f"{question} {' '.join(dict.fromkeys(context_terms))}".strip(), True

    # -- candidates ----------------------------------------------------------

    async def _candidates(self, match: str, scope: str) -> list[Candidate]:
        if not match:
            return []
        body = await self._match(match, scope, header_only=False)
        header = await self._match(match, scope, header_only=True)

        found: dict[str, Candidate] = {}
        for rank, row in enumerate(body, start=1):
            found[str(row["id"])] = _candidate(
                row, lexical_rank=rank, bm25=float(str(row["score"]))
            )
        for rank, row in enumerate(header, start=1):
            chunk_id = str(row["id"])
            existing = found.get(chunk_id)
            found[chunk_id] = (
                replace(existing, header_rank=rank)
                if existing is not None
                else _candidate(row, header_rank=rank, bm25=float(str(row["score"])))
            )
        return [self._score(c) for c in found.values()]

    async def _match(self, match: str, scope: str, *, header_only: bool) -> list[Row]:
        # The scope filter is part of the query, not a later pass. A memory the
        # requester may not see never enters the ranked pool.
        return await self._store.raw(
            "SELECT c.id, c.memory_id, c.path, c.ordinal, c.text, c.scope, c.salience,"
            "       c.pinned, c.updated_at, bm25(chunks_fts) AS score"
            " FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid"
            " WHERE chunks_fts MATCH ?"
            "   AND (c.scope = 'workspace' OR c.scope = ?)"
            + ("   AND c.ordinal = 0" if header_only else "")
            + " ORDER BY score LIMIT ?",
            (match, scope, SOURCE_LIMIT),
        )

    async def _pinned(self, scope: str) -> list[Candidate]:
        rows = await self._store.raw(
            "SELECT id, memory_id, path, ordinal, text, scope, salience, pinned, updated_at"
            " FROM chunks WHERE pinned = 1 AND (scope = 'workspace' OR scope = ?)"
            " ORDER BY memory_id, ordinal",
            (scope,),
        )
        return [self._score(_candidate(row)) for row in rows]

    async def _denied(self, match: str, scope: str) -> list[Candidate]:
        """What the scope filter excluded. Explanation only — never packed."""
        if not match:
            return []
        rows = await self._store.raw(
            "SELECT c.id, c.memory_id, c.path, c.ordinal, c.text, c.scope, c.salience,"
            "       c.pinned, c.updated_at, bm25(chunks_fts) AS score"
            " FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid"
            " WHERE chunks_fts MATCH ? AND c.scope != 'workspace' AND c.scope != ?"
            " ORDER BY score LIMIT ?",
            (match, scope, SOURCE_LIMIT),
        )
        return [
            _candidate(row, denied=f"scope {row['scope']!r} is not visible from {scope!r}")
            for row in rows
        ]

    # -- scoring -------------------------------------------------------------

    def _score(self, candidate: Candidate) -> Candidate:
        fused = 0.0
        if candidate.lexical_rank is not None:
            fused += 1.0 / (RRF_K + candidate.lexical_rank)
        if candidate.header_rank is not None:
            # A hit on the title or tags is a stronger signal than one buried in
            # prose, so the header list fuses in as its own ranking.
            fused += 1.0 / (RRF_K + candidate.header_rank)
        if candidate.pinned:
            fused += 1.0 / RRF_K

        recency = self._recency(candidate.updated_at)
        return replace(
            candidate,
            fused=fused,
            recency=recency,
            final=fused * candidate.salience * recency,
        )

    def _recency(self, updated_at: str) -> float:
        try:
            updated = datetime.fromisoformat(updated_at)
        except ValueError:
            return 1.0
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=UTC)
        now = self._now or datetime.now(UTC)
        age_days = max((now - updated).total_seconds() / 86_400, 0.0)
        return math.exp(-age_days * math.log(2) / RECENCY_HALF_LIFE_DAYS)

    # -- packing -------------------------------------------------------------

    def _pack(self, ranked: Sequence[Candidate], trace: RetrievalTrace) -> Retrieval:
        result = Retrieval(trace=trace)
        used = 0
        seen: set[str] = set()

        for candidate in ranked:
            if candidate.denied is not None:
                continue  # belt and braces: explanation rows are never packed
            # One chunk per memory. Two chunks of the same file crowd out a
            # second opinion, and the agent can read the whole file with
            # `memory_read` when it wants more.
            if candidate.memory_id in seen or len(seen) >= self._limit:
                continue
            snippet = render_snippet(candidate)
            cost = self._tok.count(snippet)
            if used + cost > self._budget:
                continue
            seen.add(candidate.memory_id)
            used += cost
            trace.kept.append(candidate)
            (result.pinned if candidate.pinned else result.snippets).append(snippet)

        trace.used_tokens = used
        return result


def render_snippet(candidate: Candidate) -> str:
    return f"[[{candidate.memory_id}]] ({candidate.path})\n{candidate.text.strip()}"


def _candidate(
    row: Row,
    *,
    lexical_rank: int | None = None,
    header_rank: int | None = None,
    bm25: float | None = None,
    denied: str | None = None,
) -> Candidate:
    return Candidate(
        memory_id=str(row["memory_id"]),
        path=str(row["path"]),
        chunk_id=str(row["id"]),
        ordinal=int(str(row["ordinal"])),
        text=str(row["text"]),
        scope=str(row["scope"]),
        salience=float(str(row["salience"])),
        pinned=bool(row["pinned"]),
        updated_at=str(row["updated_at"]),
        lexical_rank=lexical_rank,
        header_rank=header_rank,
        bm25=bm25,
        denied=denied,
    )


def _merge(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Collapse the candidate sources into one list, keyed by chunk."""
    merged: dict[str, Candidate] = {}
    for candidate in candidates:
        if (existing := merged.get(candidate.chunk_id)) is None:
            merged[candidate.chunk_id] = candidate
            continue
        merged[candidate.chunk_id] = replace(
            existing,
            lexical_rank=existing.lexical_rank or candidate.lexical_rank,
            header_rank=existing.header_rank or candidate.header_rank,
            fused=max(existing.fused, candidate.fused),
            final=max(existing.final, candidate.final),
        )
    return list(merged.values())

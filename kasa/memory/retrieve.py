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

**Recalled text is scrubbed on the way out.** A secret that reaches long-term
memory is replayed to the provider on every turn that retrieves it, so the
redactor belongs on this boundary and not only on the tool path — see `scrub`
on `Retriever` and #67.
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

#: Replaces secret material in recalled text. `kasa.redact.Redactor.scrub`.
Scrubber = Callable[[str], str]

#: Reciprocal-rank-fusion constant. 60 is the value from the original paper and
#: is not worth tuning until there is a second list to fuse with.
RRF_K = 60

#: How long it takes a memory's recency boost to halve. Long, because long-term
#: memory is supposed to be long-term; recency breaks ties, it does not rank.
RECENCY_HALF_LIFE_DAYS = 120.0

#: How much a title/tag hit outweighs one buried in prose. The header list is a
#: separate ranking over a much smaller field, and a memory whose *title* is
#: about the question is usually the memory that was wanted. Until #41 this
#: weight was an accident — the header chunk matched both source queries and so
#: fused twice — and 2.0 keeps the ranking it produced now that the hit is
#: carried by the body instead.
HEADER_WEIGHT = 2.0

#: How many chunks each candidate source contributes before fusion.
SOURCE_LIMIT = 30

#: How many memories belong in a prompt that also has to hold the conversation.
#: A packing bound, not a ceiling on what retrieval can find — a caller asking an
#: explicit question passes its own limit to `retrieve` (#61).
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

#: Words too common to discriminate. A lexicon of function words and the
#: generic verbs conversation is made of, not a frequency list — `build_match`
#: ORs whatever survives, so one word like "should" in a question is enough to
#: pull in a memory that happens to contain it (#47). Morphology is FTS's job:
#: these are matched literally, and the porter tokenizer stems what is left.
_STOPWORDS = frozenset(
    {
        "a",
        "about",
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
        "could",
        "did",
        "do",
        "does",
        "else",
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
        "know",
        "me",
        "might",
        "more",
        "must",
        "my",
        "need",
        "of",
        "on",
        "or",
        "our",
        "say",
        "should",
        "so",
        "tell",
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
        "want",
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
        "would",
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
    #: Split for the context packer, which files pinned material into the
    #: cacheable prefix and the rest into the per-turn block. The split is a
    #: destination, not an order — read `kept` when rank is what matters.
    snippets: list[str] = field(default_factory=list)
    pinned: list[str] = field(default_factory=list)
    #: Everything packed, best first, whichever block it went to.
    kept: list[Candidate] = field(default_factory=list)
    trace: RetrievalTrace | None = None

    @property
    def memory_ids(self) -> list[str]:
        return [c.memory_id for c in self.kept]


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

    The anaphor may be anywhere. Scanning only the opening words caught the
    docstring's own example and missed the shape people actually type — "what
    else do we know about them?", "can you say more about her?" — which was
    #44: the rewriter was never reached, `build_match` was left with whatever
    generic words remained, and the turn retrieved nothing at all.

    Scanning the whole message does mean a question that names its subject
    *and* uses a pronoun is rewritten when it did not need to be. That is the
    error worth making: an unnecessary rewrite costs one utility-model call and
    still returns the right memories — and costs nothing at all in the common
    setup, where no utility model is configured and `_build_query` falls back
    to carrying terms from the recent turns. A missed rewrite costs the answer.
    """
    words = _WORD.findall(message.lower())
    if len(words) < _SELF_CONTAINED_MIN_WORDS:
        return False
    if not [w for w in words if w not in _STOPWORDS]:
        return False
    return not (set(words) & _ANAPHORA)


def build_match(query: str) -> str:
    """Turn free text into an FTS5 MATCH expression.

    Every term is quoted and OR-ed. Quoting matters for correctness, not
    tidiness: an unquoted apostrophe or a bare `NEAR` in someone's message is a
    syntax error inside FTS5, and a syntax error here is a failed turn.

    OR rather than AND, deliberately, even though #47 makes the case that ORing
    generic terms drags in unrelated memories. AND would make one unusual word
    in a question zero the whole result, and the golden set exists because
    people do not phrase questions in the words the note was written in. The
    discrimination belongs downstream: `bm25` scores a chunk matching three
    terms above one matching one, and `_score` fuses on that ranking. What ORing
    could not survive was a stopword list that let "should" through — so the
    list is the fix, not the operator.
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
        scrub: Scrubber | None = None,
        record_hits: bool = False,
        now: datetime | None = None,
    ) -> None:
        self._store = store
        self._tok = tokenizer
        self._budget = budget_tokens
        self._limit = limit
        self._rewriter = rewriter
        # Defaulting to no scrubbing keeps the retriever usable without a
        # config — every caller that builds a prompt passes one, and there is a
        # test that says so.
        self._scrub = scrub
        # Off by default, because most retrievers are not conversations. `kasa
        # why` traces what *would* be recalled and `promote` reads competition
        # for a plan; counting either as a recall would let a background job
        # and a debugging session decide what stays in long-term memory.
        self._record_hits = record_hits
        self._now = now

    async def retrieve(
        self,
        question: str,
        *,
        scope: str = "workspace",
        recent: Sequence[str] = (),
        explain: bool = False,
        include_pinned: bool = True,
        limit: int | None = None,
    ) -> Retrieval:
        """Rank memories against `question`.

        `include_pinned` adds every pinned memory to the pool whether or not it
        matched, which is what the context packer wants: standing instructions
        should arrive in the prompt unasked. A caller that is answering an
        explicit query wants the opposite — see `memory_search`. Pinned
        memories that *do* match are found by the lexical query either way, and
        keep their pinned bonus.

        `limit` overrides how many memories are packed. The retriever's own
        limit is a budget for the prompt: it is how many memories fit alongside
        a conversation, and it is the right default for the pre-injected
        retrieval every turn pays for. A caller that has decided the prompt did
        not have enough is asking a different question, and gets to say how many
        answers it wants. The token budget still caps both.

        When `include_pinned` is on, pinned memories are packed against their
        own allowance of `limit` rather than competing with the matched ones —
        a standing instruction is not an answer to the question, and it must
        not be evicted by one. See `_pack`.

        Every candidate's text is scrubbed before it is ranked into anything a
        caller can read, so the trace, the snippets and `kept` all carry the
        same redacted text.
        """
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
        pinned = await self._pinned(scope) if include_pinned else []
        merged = [self._redact(c) for c in _merge(candidates + pinned)]
        for candidate in merged:
            trace.candidates.append(candidate)

        ranked = sorted(merged, key=lambda c: c.final, reverse=True)
        trace.candidates = ranked
        if explain:
            trace.denied = await self._denied(match, scope)

        retrieval = self._pack(
            ranked,
            trace,
            limit if limit is not None else self._limit,
            reserve_pinned=include_pinned,
        )
        if self._record_hits:
            # What was packed, not what was ranked: a memory that lost to the
            # budget was not recalled, and salience is meant to measure being
            # useful rather than being nearly useful. Deduplicated because one
            # memory can contribute several chunks to one prompt, and that is
            # one recall.
            await self._store.record_memory_hits(list(dict.fromkeys(retrieval.memory_ids)))
        return retrieval

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
            await self._carry_header_rank(found, row, rank, scope)
        return [self._score(c) for c in found.values()]

    async def _carry_header_rank(
        self, found: dict[str, Candidate], row: Row, rank: int, scope: str
    ) -> None:
        """Attach a title/tag hit to the memory's prose rather than to itself.

        The header chunk is a locator. It exists so that a memory titled "Deploy
        pipeline ownership" can be found by that phrase, and it holds the title
        and the tags — which is to say, nothing the model could not already
        guess from the file path. Packing it in place of the body was the bug in
        #41: because the header matched both source queries it fused twice and
        outranked its own prose, and `_pack` keeps one chunk per memory.

        So the rank lands on every body chunk of the memory, boosting the file
        the way a title hit should, and the header itself is packed only by a
        memory that has no prose to offer instead.
        """
        memory_id = str(row["memory_id"])
        bodies = [c for c in found.values() if c.memory_id == memory_id and c.ordinal > 0]
        if bodies:
            for candidate in bodies:
                found[candidate.chunk_id] = replace(candidate, header_rank=rank)
            return

        lead = await self._lead_body(memory_id, scope)
        source = lead if lead is not None else row
        found[str(source["id"])] = _candidate(
            source, header_rank=rank, bm25=float(str(row["score"]))
        )

    async def _match(self, match: str, scope: str, *, header_only: bool) -> list[Row]:
        # The scope filter is part of the query, not a later pass. A memory the
        # requester may not see never enters the ranked pool.
        return await self._store.raw(
            "SELECT c.id, c.memory_id, c.path, c.ordinal, c.text, c.scope, c.salience,"
            "       c.pinned, c.updated_at, bm25(chunks_fts) AS score"
            " FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid"
            " WHERE chunks_fts MATCH ?"
            "   AND (c.scope = 'workspace' OR c.scope = ?)"
            + ("   AND c.ordinal = 0" if header_only else "   AND c.ordinal > 0")
            + " ORDER BY score LIMIT ?",
            (match, scope, SOURCE_LIMIT),
        )

    async def _lead_body(self, memory_id: str, scope: str) -> Row | None:
        """The first prose chunk of a memory located by its title or tags."""
        rows = await self._store.raw(
            "SELECT id, memory_id, path, ordinal, text, scope, salience, pinned, updated_at"
            " FROM chunks WHERE memory_id = ? AND ordinal > 0"
            "   AND (scope = 'workspace' OR scope = ?)"
            " ORDER BY ordinal LIMIT 1",
            (memory_id, scope),
        )
        return rows[0] if rows else None

    async def _pinned(self, scope: str) -> list[Candidate]:
        rows = await self._store.raw(
            "SELECT id, memory_id, path, ordinal, text, scope, salience, pinned, updated_at"
            " FROM chunks WHERE pinned = 1 AND (scope = 'workspace' OR scope = ?)"
            " ORDER BY memory_id, ordinal",
            (scope,),
        )
        # Pinned chunks all score alike, so packing takes the first one listed.
        # Without this that is always the header, and a standing instruction
        # arrives in every single prompt as its own title.
        return [self._score(_candidate(row)) for row in _prefer_body(rows)]

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
            # The memory's title or tags matched, which is a stronger signal
            # than a hit buried in prose, so that list fuses in as its own
            # ranking. It is carried by the body — see `_carry_header_rank`.
            fused += HEADER_WEIGHT / (RRF_K + candidate.header_rank)
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

    # -- redaction -----------------------------------------------------------

    def _redact(self, candidate: Candidate) -> Candidate:
        """Scrub a candidate's text once, before anything downstream reads it.

        Here rather than in `render_snippet` or in the context packer because
        this is the single point every consumer of recalled text passes
        through: the pre-injected snippets, `memory_search`, `kept`, and the
        trace `kasa why` prints. #67 was that the redactor sat only on
        `ToolRegistry`, so the same memory was scrubbed when the model *asked*
        for it and sent verbatim when the model was *given* it — and being
        given it is the path every turn takes.

        After scoring, so redaction cannot change what ranks; before packing,
        so the token budget is charged for the text actually sent.

        The conversation itself is deliberately not scrubbed here. A key the
        user just pasted is theirs, sent to the provider in that turn by their
        own act; a key sitting in long-term memory is replayed unbidden on
        every turn that retrieves it, forever. Only the second is this
        module's to fix.
        """
        if self._scrub is None:
            return candidate
        scrubbed = self._scrub(candidate.text)
        return candidate if scrubbed == candidate.text else replace(candidate, text=scrubbed)

    # -- packing -------------------------------------------------------------

    def _pack(
        self,
        ranked: Sequence[Candidate],
        trace: RetrievalTrace,
        limit: int,
        *,
        reserve_pinned: bool,
    ) -> Retrieval:
        """Choose what goes in the prompt, best first.

        `limit` is two allowances, not one. Pinned memories are taken first and
        against their own count; matched memories are taken against theirs.
        Sharing one bound was #75: on any question that matched eight things
        well, the standing instruction was simply gone — and because pinned
        material lives in the cacheable prefix, the prefix then flipped between
        two shapes turn to turn, which is the one thing `ContextPacker` says it
        exists to prevent.

        Pinned still has a bound rather than none, so that pinning a great deal
        cannot starve the answer to the question of every token in the budget.
        The context packer's own `pinned` share is the ceiling on how much of
        this actually reaches the prompt.

        `reserve_pinned` is off for an explicit search. There, a pinned memory
        is a result like any other and competes for the caller's `limit` — the
        caller asked what matched, and the pinned block is already in the
        prompt above the answer (#57).
        """
        result = Retrieval(trace=trace)
        used = 0
        seen: set[str] = set()
        chosen: dict[str, str] = {}

        def take(candidate: Candidate) -> bool:
            nonlocal used
            snippet = render_snippet(candidate)
            cost = self._tok.count(snippet)
            if used + cost > self._budget:
                return False
            used += cost
            # One chunk per memory. Two chunks of the same file crowd out a
            # second opinion, and the agent can read the whole file with
            # `memory_read` when it wants more.
            seen.add(candidate.memory_id)
            chosen[candidate.chunk_id] = snippet
            return True

        # `denied is not None` is belt and braces: explanation rows are never
        # packed.
        if reserve_pinned:
            taken = 0
            for candidate in ranked:
                if taken >= limit:
                    break
                if candidate.denied is not None or not candidate.pinned:
                    continue
                if candidate.memory_id not in seen and take(candidate):
                    taken += 1

        taken = 0
        for candidate in ranked:
            if taken >= limit:
                break
            if candidate.denied is not None or candidate.memory_id in seen:
                continue
            if reserve_pinned and candidate.pinned:
                # It had its own pass. Letting it take a matched slot as well
                # would starve the answer just as thoroughly as sharing one
                # bound did — with the eviction running the other way.
                continue
            if take(candidate):
                taken += 1

        # Rank order, whichever pass put it there: `kept` is what ranked, and
        # the pinned/snippets split is only a destination.
        for candidate in ranked:
            snippet = chosen.get(candidate.chunk_id)
            if snippet is None:
                continue
            result.kept.append(candidate)
            (result.pinned if candidate.pinned else result.snippets).append(snippet)

        trace.kept = result.kept
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


def _prefer_body(rows: Sequence[Row]) -> list[Row]:
    """Drop each memory's header row when it has prose to offer instead."""
    by_memory: dict[str, list[Row]] = {}
    for row in rows:
        by_memory.setdefault(str(row["memory_id"]), []).append(row)
    kept: list[Row] = []
    for group in by_memory.values():
        kept.extend([r for r in group if int(str(r["ordinal"])) > 0] or group)
    return kept


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

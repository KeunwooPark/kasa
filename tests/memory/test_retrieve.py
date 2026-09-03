"""Lexical retrieval, including the golden set #15 asks for.

The golden set is twenty questions against a fixture corpus with the memory each
one should surface. It is the only test here that measures whether retrieval is
any *good*, as opposed to whether it is correct — and it is the one that will
catch a scoring change that quietly makes recall worse.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kasa.llm.tokens import HeuristicTokenizer, Tokenizer
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.explain import render_trace
from kasa.memory.index import MemoryIndex
from kasa.memory.retrieve import Retriever, build_match, is_self_contained, permits
from kasa.store import Store

NOW = datetime(2026, 9, 3, tzinfo=UTC)


# -- the fixture corpus ------------------------------------------------------

CORPUS: list[dict[str, object]] = [
    {
        "type": "person",
        "title": "Jane Okafor",
        "tags": ["infra", "ownership"],
        "body": "Jane owns the deploy pipeline and reviews every change to it. "
        "She prefers async review over meetings.",
    },
    {
        "type": "person",
        "title": "Bob Nkemelu",
        "tags": ["oncall"],
        "body": "Bob runs the incident rota and is the escalation point overnight.",
    },
    {
        "type": "project",
        "title": "Deploy pipeline",
        "tags": ["infra"],
        "body": "Deploys run nightly at 02:00 UTC from the main branch. "
        "A failed deploy rolls back automatically and pages the rota.",
    },
    {
        "type": "project",
        "title": "Billing migration",
        "tags": ["billing"],
        "body": "Moving invoicing off the legacy Postgres cluster onto the new "
        "ledger service. Blocked on the reconciliation script.",
    },
    {
        "type": "topic",
        "title": "Code review conventions",
        "tags": ["process"],
        "body": "Two approvals for anything touching money. One approval "
        "otherwise. Reviews are expected within a working day.",
    },
    {
        "type": "topic",
        "title": "Postgres upgrade plan",
        "tags": ["database"],
        "body": "The cluster moves from 14 to 16 in Q4. Logical replication "
        "first, then a cutover during the Sunday maintenance window.",
    },
    {
        "type": "fact",
        "title": "Staging credentials rotate monthly",
        "tags": ["security"],
        "body": "Staging database credentials rotate on the first of each month, "
        "automatically, via the secrets operator.",
    },
    {
        "type": "fact",
        "title": "The office wifi password is on the whiteboard",
        "tags": ["office"],
        "body": "Guests should ask reception rather than reading the whiteboard.",
    },
    {
        "type": "topic",
        "title": "Incident retrospective format",
        "tags": ["process", "oncall"],
        "body": "Blameless. Written within two working days. Timeline first, "
        "then contributing factors, then actions with owners.",
    },
    {
        "type": "fact",
        "title": "Kasa runs on a single workspace",
        "tags": ["kasa"],
        "body": "Multi-tenancy is explicitly out of scope for v1.",
    },
]

#: question -> the title of the memory it should surface.
GOLDEN: list[tuple[str, str]] = [
    ("Who owns the deploy pipeline?", "Jane Okafor"),
    ("Who should I ask about deploys?", "Jane Okafor"),
    ("Does Jane like meetings?", "Jane Okafor"),
    ("Who is on the incident rota?", "Bob Nkemelu"),
    ("Who do I escalate to overnight?", "Bob Nkemelu"),
    ("When do deploys run?", "Deploy pipeline"),
    ("What happens when a deploy fails?", "Deploy pipeline"),
    ("What is blocking the billing migration?", "Billing migration"),
    ("Are we still on the legacy invoicing cluster?", "Billing migration"),
    ("How many approvals does a change need?", "Code review conventions"),
    ("What are the code review conventions?", "Code review conventions"),
    ("How long should a review take?", "Code review conventions"),
    ("When is the Postgres upgrade?", "Postgres upgrade plan"),
    ("How are we upgrading the database cluster?", "Postgres upgrade plan"),
    ("How often do staging credentials rotate?", "Staging credentials rotate monthly"),
    ("Is credential rotation automatic?", "Staging credentials rotate monthly"),
    ("What is the wifi password situation?", "The office wifi password is on the whiteboard"),
    ("What format do we use for retrospectives?", "Incident retrospective format"),
    ("Are retrospectives blameless?", "Incident retrospective format"),
    ("Does Kasa support multiple workspaces?", "Kasa runs on a single workspace"),
]


@pytest.fixture
def tokenizer() -> Tokenizer:
    return HeuristicTokenizer()


def write(root: Path, doc: MemoryDoc, path: str | None = None) -> MemoryDoc:
    target = root / (path or doc.suggested_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    return doc


@pytest.fixture
async def corpus(tmp_path: Path, store: Store) -> dict[str, str]:
    """The fixture corpus, indexed. Returns title -> memory id."""
    bootstrap(tmp_path)
    by_title = {}
    for entry in CORPUS:
        doc = MemoryDoc.new(**entry)  # type: ignore[arg-type]
        write(tmp_path, doc)
        by_title[str(entry["title"])] = doc.id
    await MemoryIndex(store, tmp_path).reindex()
    return by_title


def retriever(store: Store, tokenizer: Tokenizer, **kwargs: object) -> Retriever:
    return Retriever(store, tokenizer=tokenizer, now=NOW, **kwargs)  # type: ignore[arg-type]


# -- the golden set ----------------------------------------------------------


async def test_golden_set_recall_at_5(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """Twenty questions, each with the memory it should surface in the top five."""
    search = retriever(store, tokenizer)
    misses = []

    for question, expected_title in GOLDEN:
        retrieval = await search.retrieve(question)
        top5 = retrieval.memory_ids[:5]
        if corpus[expected_title] not in top5:
            misses.append((question, expected_title))

    recall = (len(GOLDEN) - len(misses)) / len(GOLDEN)
    # Currently 20/20. One unit of slack, so an unrelated corpus edit does not
    # fail the build, and no more — this is the number the whole system is for.
    assert recall >= 0.95, f"recall@5 was {recall:.0%}; missed {misses}"


async def test_golden_set_recall_at_1_is_respectable(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """Not a hard requirement, but a floor that catches a scoring regression."""
    search = retriever(store, tokenizer)
    hits = 0
    for question, expected_title in GOLDEN:
        retrieval = await search.retrieve(question)
        if retrieval.memory_ids[:1] == [corpus[expected_title]]:
            hits += 1
    assert hits / len(GOLDEN) >= 0.75, f"recall@1 was {hits}/{len(GOLDEN)}"


async def test_questions_do_not_have_to_match_the_notes_word_for_word(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """Stemming. Nobody conjugates their question to match what was written down."""
    search = retriever(store, tokenizer)

    plural = await search.retrieve("Who should I ask about deploys?")
    assert corpus["Jane Okafor"] in plural.memory_ids[:5]

    inflected = await search.retrieve("Is credential rotation automatic?")
    assert corpus["Staging credentials rotate monthly"] in inflected.memory_ids[:5]


async def test_derivational_morphology_is_a_known_gap(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """A characterization test, not an endorsement.

    Porter stems "deploying" to "deploy" but leaves "deployments" alone, so this
    question misses a memory that is plainly about it. Inflection is handled;
    word formation is not, and no amount of stopword tuning fixes that — it is
    the case for hybrid retrieval in #31.

    When #31 lands this test should start failing. Delete it then.
    """
    found = await retriever(store, tokenizer).retrieve("Who handles deployments?")
    assert corpus["Jane Okafor"] not in found.memory_ids

    # The same question in the words the memory uses works fine.
    rephrased = await retriever(store, tokenizer).retrieve("Who handles deploying?")
    assert corpus["Jane Okafor"] in rephrased.memory_ids[:5]


# -- query construction ------------------------------------------------------


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Who owns the deploy pipeline?", True),
        ("What did Jane decide about the rota?", True),
        ("what about it?", False),
        ("it?", False),
        ("them", False),
        ("the of and", False),
        ("thanks", False),
    ],
)
def test_self_containment_is_detected(message: str, expected: bool) -> None:
    assert is_self_contained(message) is expected


async def test_a_self_contained_message_is_not_rewritten(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """Rewriting costs a model call the common case does not need."""
    retrieval = await retriever(store, tokenizer).retrieve(
        "Who owns the deploy pipeline?", recent=["earlier turn"]
    )
    assert retrieval.trace is not None
    assert retrieval.trace.rewritten is False
    assert retrieval.trace.query == "Who owns the deploy pipeline?"


async def test_a_follow_up_borrows_terms_from_the_conversation(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    retrieval = await retriever(store, tokenizer).retrieve(
        "who owns it?", recent=["Tell me about the deploy pipeline"]
    )
    assert retrieval.trace is not None
    assert retrieval.trace.rewritten is True
    assert "deploy" in retrieval.trace.query
    assert corpus["Jane Okafor"] in retrieval.memory_ids[:3]


async def test_a_supplied_rewriter_is_used(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """The utility model plugs in here without this module knowing about it."""

    async def rewrite(message: str, recent: list[str]) -> str:
        return "postgres upgrade"

    search = retriever(store, tokenizer, rewriter=rewrite)
    retrieval = await search.retrieve("what about it?", recent=["something"])

    assert retrieval.memory_ids[:1] == [corpus["Postgres upgrade plan"]]


@pytest.mark.parametrize(
    "text",
    ["it's a NEAR thing", 'say "hello"', "a AND b OR c", "^caret *star", "", "-- dashes"],
)
def test_awkward_queries_do_not_break_fts(
    text: str, corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """An FTS syntax error here is a failed turn, not a bad result."""
    build_match(text)  # must not raise


async def test_awkward_queries_return_something_rather_than_erroring(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    for text in ["it's a NEAR thing", 'say "hello"', "a AND b OR c", "*", ""]:
        await retriever(store, tokenizer).retrieve(text)


# -- scope: the filter that must never be forgotten --------------------------


@pytest.mark.parametrize(
    ("requester", "memory", "allowed"),
    [
        ("workspace", "workspace", True),
        ("channel:C1", "workspace", True),
        ("private:U1", "workspace", True),
        ("workspace", "channel:C1", False),
        ("workspace", "private:U1", False),
        ("channel:C1", "channel:C1", True),
        ("channel:C1", "channel:C2", False),
        ("private:U1", "private:U2", False),
        ("channel:C1", "private:U1", False),
    ],
)
def test_scope_visibility_rules(requester: str, memory: str, allowed: bool) -> None:
    assert permits(requester, memory) is allowed


async def test_a_private_memory_never_reaches_a_public_scope(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """The number-one failure mode of a shared-memory bot, guarded directly."""
    bootstrap(tmp_path)
    secret = write(
        tmp_path,
        MemoryDoc.new(
            type="fact",
            title="Salary negotiation",
            body="They asked for a raise to 180k.",
            visibility="private:U01",
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("salary raise", scope="workspace")

    assert secret.id not in retrieval.memory_ids
    assert all(secret.id not in s for s in retrieval.snippets)


async def test_the_owner_of_a_private_memory_still_sees_it(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    bootstrap(tmp_path)
    secret = write(
        tmp_path,
        MemoryDoc.new(
            type="fact", title="Salary negotiation", body="Raise to 180k.", visibility="private:U01"
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("salary raise", scope="private:U01")
    assert secret.id in retrieval.memory_ids


async def test_scope_filtering_happens_before_ranking(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """A denied memory must not appear in the candidate pool at all."""
    bootstrap(tmp_path)
    write(
        tmp_path,
        MemoryDoc.new(
            type="fact", title="Private deploys", body="Deploy secret.", visibility="private:U01"
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("deploy", scope="workspace")

    assert retrieval.trace is not None
    assert retrieval.trace.candidates == [], "it never entered ranking, not just packing"


async def test_explain_reports_what_scope_removed_without_packing_it(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    bootstrap(tmp_path)
    secret = write(
        tmp_path,
        MemoryDoc.new(
            type="fact", title="Private deploys", body="Deploy secret.", visibility="private:U01"
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve(
        "deploy", scope="workspace", explain=True
    )

    assert retrieval.trace is not None
    # Chunk-level: a trace that collapsed to one row per memory would hide which
    # part of the file matched, which is usually the thing you are debugging.
    assert {c.memory_id for c in retrieval.trace.denied} == {secret.id}
    assert retrieval.snippets == [], "explained, not injected"
    assert "not visible from" in render_trace(retrieval)


# -- scoring and packing -----------------------------------------------------


async def test_pinned_memories_are_always_retrieved(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """Standing instructions have to arrive whether or not anyone asked."""
    bootstrap(tmp_path)
    pinned = write(
        tmp_path,
        MemoryDoc.new(
            type="fact", title="Standing instruction", body="Always answer in metric.", pinned=True
        ),
    )
    write(tmp_path, MemoryDoc.new(type="fact", title="Unrelated", body="Something else."))
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("a question about nothing in particular")

    assert pinned.id in retrieval.memory_ids
    assert retrieval.pinned, "and it is packed as pinned, not as ordinary retrieval"


async def test_salience_breaks_ties(tmp_path: Path, store: Store, tokenizer: Tokenizer) -> None:
    bootstrap(tmp_path)
    dull = write(
        tmp_path,
        MemoryDoc.new(type="fact", title="Deploy note one", body="Deploys happen.", salience=0.1),
    )
    sharp = write(
        tmp_path,
        MemoryDoc.new(type="fact", title="Deploy note two", body="Deploys happen.", salience=0.9),
    )
    await MemoryIndex(store, tmp_path).reindex()

    ids = (await retriever(store, tokenizer).retrieve("deploys")).memory_ids
    assert ids.index(sharp.id) < ids.index(dull.id)


async def test_recency_decays(tmp_path: Path, store: Store, tokenizer: Tokenizer) -> None:
    bootstrap(tmp_path)
    old = MemoryDoc.new(type="fact", title="Rota note one", body="The rota rotates.")
    old = old.model_copy(
        update={
            "frontmatter": old.frontmatter.model_copy(update={"updated": NOW - timedelta(days=800)})
        }
    )
    write(tmp_path, old)
    fresh = write(
        tmp_path, MemoryDoc.new(type="fact", title="Rota note two", body="The rota rotates.")
    )
    await MemoryIndex(store, tmp_path).reindex()

    ids = (await retriever(store, tokenizer).retrieve("rota")).memory_ids
    assert ids.index(fresh.id) < ids.index(old.id)


async def test_results_are_deduped_by_memory(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """Two chunks of one file would crowd out a second opinion."""
    bootstrap(tmp_path)
    body = "\n\n".join(f"## Section {i}\n\ndeploy pipeline detail " + "x" * 300 for i in range(4))
    doc = write(tmp_path, MemoryDoc.new(type="topic", title="Deploys", body=body))
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("deploy pipeline")
    assert retrieval.memory_ids.count(doc.id) == 1


async def test_the_retrieval_budget_is_respected(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    search = retriever(store, tokenizer, budget_tokens=40)
    retrieval = await search.retrieve("deploy pipeline rota review postgres billing")

    assert retrieval.trace is not None
    assert retrieval.trace.used_tokens <= 40
    assert len(retrieval.snippets) < len(CORPUS)


async def test_the_result_limit_is_respected(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    search = retriever(store, tokenizer, limit=2)
    retrieval = await search.retrieve("deploy pipeline rota review postgres billing")
    assert len(retrieval.memory_ids) <= 2


async def test_snippets_carry_the_memory_id_so_the_agent_can_read_more(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    retrieval = await retriever(store, tokenizer).retrieve("Who owns the deploy pipeline?")
    assert f"[[{corpus['Jane Okafor']}]]" in "\n".join(retrieval.snippets)


async def test_an_empty_index_retrieves_nothing_without_erroring(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    bootstrap(tmp_path)
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("anything at all")
    assert retrieval.snippets == []


# -- the trace ---------------------------------------------------------------


async def test_the_trace_explains_the_whole_pipeline(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    retrieval = await retriever(store, tokenizer).retrieve(
        "Who owns the deploy pipeline?", explain=True
    )
    rendered = render_trace(retrieval)

    assert "QUERY" in rendered
    assert "CANDIDATES" in rendered
    assert "SCOPE FILTER" in rendered
    assert "PACKED" in rendered
    assert corpus["Jane Okafor"] in rendered
    assert "final" in rendered and "recency" in rendered


async def test_the_trace_marks_what_was_packed(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    retrieval = await retriever(store, tokenizer, limit=1).retrieve("deploy pipeline")
    assert "* = packed into the prompt" in render_trace(retrieval)
    assert retrieval.trace is not None
    assert len(retrieval.trace.kept) == 1


async def test_a_question_that_finds_nothing_says_so(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    rendered = render_trace(
        await retriever(store, tokenizer).retrieve("zygomorphic quinquagenarian", explain=True)
    )
    assert "none matched" in rendered
    assert "nothing was injected" in rendered

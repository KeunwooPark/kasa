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

from kasa.core.context import PINNED_HEADER, ContextPacker
from kasa.llm.tokens import HeuristicTokenizer, Tokenizer
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.explain import render_trace
from kasa.memory.index import MemoryIndex
from kasa.memory.retrieve import (
    DEFAULT_LIMIT,
    Retriever,
    build_match,
    is_self_contained,
    permits,
    render_snippet,
)
from kasa.redact import Redactor
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
        # #44: the anaphor is not always at the front, and these were all
        # judged self-contained, so the rewriter never saw them.
        ("What else do we know about them?", False),
        ("Tell me more about that", False),
        ("Can you say more about her?", False),
        ("Does that change the rota?", False),
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


async def test_a_follow_up_with_a_trailing_pronoun_still_reaches_the_rewriter(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """The two-turn shape from #44, which used to retrieve nothing at all.

    "What else do we know about them?" was judged self-contained because the
    anaphor is the sixth word, so the query was built from "else", "know" and
    "about" — which match nothing, from a corpus that plainly has the answer.
    """
    search = retriever(store, tokenizer)
    first = await search.retrieve("Who owns the deploy pipeline?")
    assert corpus["Jane Okafor"] in first.memory_ids

    second = await search.retrieve(
        "What else do we know about them?", recent=["Who owns the deploy pipeline?"]
    )

    assert second.trace is not None
    assert second.trace.rewritten is True
    assert corpus["Jane Okafor"] in second.memory_ids


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
    ("question", "unwanted"),
    [
        # #47: one generic word surviving into an OR is enough to drag a whole
        # unrelated memory into the pool — and, at the time, into the prompt.
        ("Who owns the deploy pipeline, and how should they be contacted?", "should"),
        ("What do we know about postgres?", "know"),
        ("Can you tell me more about the rota?", "tell"),
        ("What else needs to happen before the cutover?", "else"),
    ],
)
def test_generic_words_do_not_survive_into_the_query(question: str, unwanted: str) -> None:
    assert unwanted not in build_match(question)


async def test_a_generic_verb_does_not_drag_in_an_unrelated_memory(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """The exact pair from #47, reduced to the two memories involved."""
    bootstrap(tmp_path)
    wanted = write(
        tmp_path,
        MemoryDoc.new(
            type="person",
            title="Jane Okafor",
            body="Jane owns the deploy pipeline and prefers to be paged on Signal.",
        ),
    )
    unrelated = write(
        tmp_path,
        MemoryDoc.new(
            type="topic",
            title="Postgres",
            body="MySQL is legacy and no new service should use it.",
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve(
        "Who owns the deploy pipeline, and how should they be contacted?"
    )

    assert wanted.id in retrieval.memory_ids
    assert unrelated.id not in retrieval.memory_ids


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


async def test_a_title_hit_injects_the_body_not_the_title(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """The header chunk locates a memory; it is not what gets injected.

    Regression for #41: the synthetic title/tag chunk matched both source
    queries, fused twice, and beat its own prose into the one slot `_pack`
    allows each memory — so the model received a file name where the answer
    should have been.
    """
    bootstrap(tmp_path)
    write(
        tmp_path,
        MemoryDoc.new(
            type="person",
            title="Jane owns the deploy pipeline",
            tags=["infra", "ownership", "deploy"],
            body="Jane Kowalski owns the deploy pipeline and the release runbook. "
            "She prefers to be paged on Signal rather than by phone.",
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve(
        "Who owns the deploy pipeline, and how should they be contacted?"
    )

    injected = "\n".join(retrieval.snippets)
    assert "Signal" in injected, "the question is answerable from what was injected"
    assert "infra ownership deploy" not in injected, "the tag list is not content"


async def test_a_memory_found_only_by_its_title_still_arrives_with_its_prose(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """The header exists for exactly this case, and must not be the answer to it."""
    bootstrap(tmp_path)
    write(
        tmp_path,
        MemoryDoc.new(
            type="topic",
            title="Deploy pipeline ownership",
            body="Ask Jane. She reviews everything that touches it.",
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("deploy pipeline ownership")

    assert "Ask Jane" in "\n".join(retrieval.snippets)


async def test_a_pinned_memory_arrives_as_its_body(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """A standing instruction is in every prompt. Its title is not the instruction."""
    bootstrap(tmp_path)
    write(
        tmp_path,
        MemoryDoc.new(
            type="fact",
            title="Standing instruction",
            tags=["style"],
            body="Always answer in metric units, and never round a currency amount.",
            pinned=True,
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("a question about nothing in particular")

    assert "Always answer in metric units" in "\n".join(retrieval.pinned)


async def test_pinned_memories_can_be_left_out_of_the_pool(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """#57. `memory_search` answers a question, so it does not want them unasked."""
    bootstrap(tmp_path)
    write(
        tmp_path,
        MemoryDoc.new(
            type="fact",
            title="Standing instruction",
            body="Always answer in metric units, and never round a currency amount.",
            pinned=True,
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()
    search = retriever(store, tokenizer)

    unasked = await search.retrieve("a question about nothing in particular", include_pinned=False)
    assert unasked.kept == []

    # Still findable, and still ranked as pinned, when the query is about it.
    asked = await search.retrieve("metric units currency", include_pinned=False)
    assert "Always answer in metric units" in "\n".join(asked.pinned)
    assert asked.kept[0].pinned


async def test_kept_reports_every_packed_memory_in_rank_order(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    retrieval = await retriever(store, tokenizer).retrieve("Who owns the deploy pipeline?")

    scores = [c.final for c in retrieval.kept]
    assert scores == sorted(scores, reverse=True)
    assert len(retrieval.kept) == len(retrieval.snippets) + len(retrieval.pinned)


async def test_a_memory_with_no_body_falls_back_to_its_header(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """A title-only memory has nothing else to give, and is still worth finding."""
    bootstrap(tmp_path)
    titled = write(
        tmp_path, MemoryDoc.new(type="fact", title="The wifi password is on the whiteboard")
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("what is the wifi password")

    assert titled.id in retrieval.memory_ids
    assert "whiteboard" in "\n".join(retrieval.snippets)


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


async def test_a_caller_can_ask_for_more_memories_than_a_prompt_holds(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """#61. `DEFAULT_LIMIT` bounds what fits in a prompt, not what can be found."""
    search = retriever(store, tokenizer)
    question = "deploy billing postgres review incident wifi kasa credentials retrospective"

    default = await search.retrieve(question)
    assert len(default.kept) == DEFAULT_LIMIT

    wider = await search.retrieve(question, limit=10)
    assert len(wider.kept) == 10
    # Still one chunk per memory, and still in rank order.
    assert wider.memory_ids == sorted(set(wider.memory_ids), key=wider.memory_ids.index)
    assert wider.kept[: len(default.kept)] == default.kept


async def test_a_smaller_limit_still_takes_the_best_ones(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    search = retriever(store, tokenizer)
    narrow = await search.retrieve("who owns the deploy pipeline", limit=2)
    assert len(narrow.kept) == 2
    assert corpus["Jane Okafor"] in narrow.memory_ids


# -- redaction (#67) ---------------------------------------------------------


LEAKY = {
    "type": "topic",
    "title": "Staging deploy key rotation",
    "tags": ["infra", "credentials"],
    "body": "The staging runner authenticates with AKIAIOSFODNN7EXAMPLE and the "
    "deploy key below.\n\n"
    "-----BEGIN RSA PRIVATE KEY-----\n"
    "MIIEpAIBAAKCAQEA1n2Xa9wZk3v4Qb8sT0pLmNoPqRsTuVwXyZ0123456789abcd\n"
    "-----END RSA PRIVATE KEY-----\n\n"
    "Rotate it when the migration lands.",
}


@pytest.fixture
async def leaky(tmp_path: Path, store: Store) -> str:
    """A single memory carrying credentials, indexed. Returns its id."""
    bootstrap(tmp_path)
    doc = MemoryDoc.new(**LEAKY)  # type: ignore[arg-type]
    write(tmp_path, doc)
    await MemoryIndex(store, tmp_path).reindex()
    return doc.id


async def test_a_credential_in_memory_never_reaches_the_prompt(
    leaky: str, store: Store, tokenizer: Tokenizer
) -> None:
    """#67. The injection path is the default path, and it was the unscrubbed one."""
    retrieval = await retriever(store, tokenizer, scrub=Redactor().scrub).retrieve(
        "how does the staging runner authenticate", explain=True
    )

    assert retrieval.memory_ids == [leaky], "the memory has to actually be found"
    packed = "\n".join([*retrieval.pinned, *retrieval.snippets])
    assert "AKIAIOSFODNN7EXAMPLE" not in packed
    assert "MIIEpAIB" not in packed
    assert "Rotate it when the migration lands." in packed, "the useful text survives"


async def test_every_view_of_a_candidate_is_scrubbed_not_just_the_snippet(
    leaky: str, store: Store, tokenizer: Tokenizer
) -> None:
    """`memory_search` renders from `kept`, and `kasa why` from the trace.

    Scrubbing only where snippets are built would leave both of those reading
    the raw text off the same candidate — which is the shape of the original
    bug, one funnel short.
    """
    retrieval = await retriever(store, tokenizer, scrub=Redactor().scrub).retrieve(
        "staging runner deploy key", explain=True
    )
    assert retrieval.trace is not None

    surfaces = [
        *(render_snippet(c) for c in retrieval.kept),
        *(c.text for c in retrieval.trace.candidates),
        render_trace(retrieval),
    ]
    for text in surfaces:
        assert "AKIAIOSFODNN7EXAMPLE" not in text
        assert "MIIEpAIB" not in text


async def test_redaction_does_not_change_what_ranks(
    corpus: dict[str, str], store: Store, tokenizer: Tokenizer
) -> None:
    """Scrubbing runs after scoring, so a corpus with no secrets ranks identically."""
    question = "who owns the deploy pipeline"
    plain = await retriever(store, tokenizer).retrieve(question)
    scrubbed = await retriever(store, tokenizer, scrub=Redactor().scrub).retrieve(question)

    assert scrubbed.memory_ids == plain.memory_ids
    assert scrubbed.snippets == plain.snippets


async def test_the_budget_is_charged_for_the_redacted_text(
    leaky: str, store: Store, tokenizer: Tokenizer
) -> None:
    """A token budget counted before redaction would be counting text nobody sends."""
    retrieval = await retriever(store, tokenizer, scrub=Redactor().scrub).retrieve(
        "how does the staging runner authenticate"
    )
    assert retrieval.trace is not None
    assert retrieval.trace.used_tokens == sum(
        tokenizer.count(s) for s in [*retrieval.pinned, *retrieval.snippets]
    )


async def test_a_retriever_with_no_scrubber_is_unchanged(
    leaky: str, store: Store, tokenizer: Tokenizer
) -> None:
    """The default stays raw: a retriever built without a config is still usable,
    and every caller that builds a prompt passes one."""
    retrieval = await retriever(store, tokenizer).retrieve("staging runner deploy key")
    assert "AKIAIOSFODNN7EXAMPLE" in "\n".join(retrieval.snippets)


async def test_the_scope_block_names_each_memory_once(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """#70. A header chunk and a body chunk are two rows in the trace and one
    fact on the page: the document's scope. Printed per chunk they came out as
    the same sentence twice, with nothing to tell the two lines apart."""
    bootstrap(tmp_path)
    secret = write(
        tmp_path,
        MemoryDoc.new(
            type="topic",
            title="Deploy incident",
            body="Deploy went wrong.\n\n## After\n\nThe deploy rota was paged.",
            visibility="private:U01",
        ),
    )
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve(
        "deploy", scope="workspace", explain=True
    )
    assert retrieval.trace is not None
    assert len(retrieval.trace.denied) > 1, "more than one chunk of one memory was excluded"

    block = render_trace(retrieval).partition("SCOPE FILTER")[2].partition("PACKED")[0]
    named = [line for line in block.splitlines() if secret.id in line]
    assert len(named) == 1, f"one line per memory, got:\n{block}"
    assert f"({len(retrieval.trace.denied)} chunks)" in named[0]
    assert f"{len(retrieval.trace.denied)} chunk(s) never entered" in block


# -- pinned memories keep their own allowance (#75) ---------------------------


@pytest.fixture
async def crowded(tmp_path: Path, store: Store) -> str:
    """One standing instruction, and twelve memories that bury it. Returns its id."""
    bootstrap(tmp_path)
    pinned = write(
        tmp_path,
        MemoryDoc.new(
            type="fact",
            title="Standing instruction",
            body="Always answer in metric units, and never round a currency amount.",
            pinned=True,
        ),
    )
    for n in range(12):
        write(
            tmp_path,
            MemoryDoc.new(
                type="topic",
                title=f"Escalation runbook step {n + 1}",
                body=f"Escalation runbook step {n + 1}: page the rota and open an incident.",
            ),
        )
    await MemoryIndex(store, tmp_path).reindex()
    return pinned.id


async def test_a_standing_instruction_is_not_evicted_by_the_answer(
    crowded: str, store: Store, tokenizer: Tokenizer
) -> None:
    """#75. Twelve better matches used to take all eight slots and the pinned
    memory with them — on an ordinary question about a well-covered topic."""
    search = retriever(store, tokenizer)

    retrieval = await search.retrieve("escalation runbook step incident rota page")

    assert len(retrieval.snippets) == DEFAULT_LIMIT, "the matched allowance is still eight"
    assert crowded in retrieval.memory_ids
    assert "Always answer in metric units" in "\n".join(retrieval.pinned)


async def test_the_cacheable_prefix_does_not_flip_with_the_question(
    crowded: str, store: Store, tokenizer: Tokenizer
) -> None:
    """The pinned block lives in `system`. If it comes and goes with the query,
    every alternation is a full cache miss on the prefix — the one thing
    `ContextPacker` says it exists to prevent."""
    search = retriever(store, tokenizer)
    packer = ContextPacker(tokenizer=tokenizer)

    prefixes = set()
    for question in ("hello there", "every escalation runbook step", "hello again"):
        got = await search.retrieve(question)
        prefixes.add(
            packer.pack(system_prompt="S", pinned=got.pinned, retrieved=got.snippets).system
        )

    assert len(prefixes) == 1
    assert PINNED_HEADER in prefixes.pop()


async def test_pinning_a_great_deal_cannot_starve_the_answer(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """The allowance is separate, not unlimited. Twenty pinned memories must
    still leave room for what the question was about."""
    bootstrap(tmp_path)
    for n in range(20):
        write(
            tmp_path,
            MemoryDoc.new(
                type="fact",
                title=f"Standing instruction {n}",
                body=f"Rule {n} applies.",
                pinned=True,
            ),
        )
    write(tmp_path, MemoryDoc.new(type="fact", title="Wifi", body="The wifi password is hunter2."))
    await MemoryIndex(store, tmp_path).reindex()

    retrieval = await retriever(store, tokenizer).retrieve("what is the wifi password")

    assert len(retrieval.pinned) == DEFAULT_LIMIT, "capped at its own allowance, not unbounded"
    assert "hunter2" in "\n".join(retrieval.snippets), "the question still got an answer"


async def test_an_explicit_search_still_counts_a_pinned_hit_against_its_limit(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """#57's decision stands: `memory_search` asked what matched, so a pinned
    memory that matches is a result like any other and does not come free.

    The separate allowance is for the pre-injected retrieval, where a standing
    instruction is not competing with the answer. Here it is one of the
    answers, and `limit` means what the caller asked it to mean."""
    bootstrap(tmp_path)
    for n in range(3):
        write(
            tmp_path,
            MemoryDoc.new(
                type="fact",
                title=f"Standing instruction {n}",
                body="Always answer about the deploy pipeline in metric units.",
                pinned=True,
            ),
        )
    for n in range(3):
        write(
            tmp_path,
            MemoryDoc.new(
                type="fact", title=f"Note {n}", body="The deploy pipeline runs nightly in metric."
            ),
        )
    await MemoryIndex(store, tmp_path).reindex()

    asked = await retriever(store, tokenizer).retrieve(
        "deploy pipeline metric", include_pinned=False, limit=3
    )

    assert len(asked.kept) == 3, "no free slot for the pinned hits"
    assert any(c.pinned for c in asked.kept), "and they did compete, rather than being excluded"

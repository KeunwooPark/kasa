"""The agent's memory tools, and the two acceptance criteria for #16.

The interesting cases are not "does search return results". They are: can the
agent recover when pre-injection missed, and can `memory_write` be talked into
touching the repository. The answers have to be yes and no.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from kasa.core.agent import Agent
from kasa.core.context import ContextPacker
from kasa.core.memory_tools import memory_tools
from kasa.core.tools import ToolContext, ToolRegistry
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.tokens import HeuristicTokenizer, Tokenizer
from kasa.llm.types import (
    ChatRequest,
    ChatResponse,
    Delta,
    MessageStop,
    TextDelta,
    ToolResultBlock,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    ToolUseStop,
    Usage,
)
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc, new_memory_id
from kasa.memory.gitcmd import GitRepo
from kasa.memory.index import MemoryIndex
from kasa.memory.ltm import MemoryStore
from kasa.memory.manifest import Manifest
from kasa.memory.retrieve import Retriever
from kasa.store import Store


@pytest.fixture
def tokenizer() -> Tokenizer:
    return HeuristicTokenizer()


class Memory:
    """A memory repo, its index, and the tools bound to both."""

    def __init__(self, root: Path, store: Store, registry: ToolRegistry, repo: GitRepo) -> None:
        self.root = root
        self.store = store
        self.registry = registry
        self.repo = repo

    async def call(self, name: str, args: dict[str, Any], **context: str) -> str:
        result = await self.registry.dispatch(
            ToolUseBlock(id="t1", name=name, input=args), ToolContext(**context)
        )
        return result.content


@pytest.fixture
async def memory(tmp_path: Path, store: Store, tokenizer: Tokenizer) -> AsyncIterator[Memory]:
    root = tmp_path / "ltm"
    repo = GitRepo.init(root, branch="main")
    bootstrap(root)

    for doc in (
        MemoryDoc.new(
            type="person",
            title="Jane Okafor",
            tags=["infra"],
            body="Jane owns the deploy pipeline and reviews every change to it.",
        ),
        MemoryDoc.new(
            type="fact",
            title="Salary review outcome",
            body="They were moved to band 5 in July.",
            visibility="private:U01",
        ),
    ):
        target = root / doc.suggested_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.render())
    Manifest.rebuild(root)[0].save(root)
    repo.commit("memory: seed")
    await MemoryIndex(store, root).reindex()
    ltm = MemoryStore(repo, store, branch="main", push=False)
    registry = ToolRegistry(
        memory_tools(
            retriever=Retriever(store, tokenizer=tokenizer),
            memory=ltm,
            store=store,
        )
    )
    yield Memory(root, store, registry, repo)


def id_of(memory: Memory, path: str) -> str:
    return MemoryDoc.parse((memory.root / path).read_text()).id


# -- memory_search -----------------------------------------------------------


async def test_search_returns_snippets_with_ids(memory: Memory) -> None:
    result = await memory.call("memory_search", {"query": "who owns the deploy pipeline"})

    assert "Jane" in result
    assert f"[[{id_of(memory, 'memory/people/jane-okafor.md')}]]" in result


async def test_search_says_so_when_nothing_matches(memory: Memory) -> None:
    result = await memory.call("memory_search", {"query": "zygomorphic quinquagenarian"})
    assert "No memories matched" in result


async def test_search_respects_the_session_scope(memory: Memory) -> None:
    """A DM-scoped memory must not surface in a workspace conversation."""
    secret = id_of(memory, "memory/facts/salary-review-outcome.md")

    public = await memory.call("memory_search", {"query": "salary band"})
    assert secret not in public

    private = await memory.call("memory_search", {"query": "salary band"}, scope="private:U01")
    assert secret in private


async def test_a_scope_hint_cannot_widen_access(memory: Memory) -> None:
    """The one argument a model could use to escape its own scope."""
    result = await memory.call(
        "memory_search", {"query": "salary band", "scope_hint": "private:U01"}
    )

    assert id_of(memory, "memory/facts/salary-review-outcome.md") not in result
    assert "cannot search the scope" in result


async def test_a_scope_hint_may_narrow(memory: Memory) -> None:
    result = await memory.call(
        "memory_search",
        {"query": "salary band", "scope_hint": "private:U01"},
        scope="private:U01",
    )
    assert id_of(memory, "memory/facts/salary-review-outcome.md") in result


async def test_the_search_limit_is_capped(memory: Memory) -> None:
    result = await memory.call("memory_search", {"query": "deploy", "limit": 9999})
    assert result  # the schema caps it; the handler clamps it too


async def stock(memory: Memory, count: int, term: str) -> None:
    """Add `count` memories that all match `term`, and reindex."""
    for n in range(count):
        doc = MemoryDoc.new(
            type="fact",
            title=f"{term.title()} runbook step {n}",
            body=f"Step {n} of the {term} runbook is owned by the platform team.",
        )
        target = memory.root / doc.suggested_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(doc.render())
    await MemoryIndex(memory.store, memory.root).reindex()


def snippet_count(result: str) -> int:
    return result.count("[[mem_")


async def test_search_returns_as_many_results_as_it_asked_for(memory: Memory) -> None:
    """#61. The schema advertises 20; the retriever's packing limit capped it at 8."""
    await stock(memory, 15, "escalation")

    twelve = await memory.call("memory_search", {"query": "escalation runbook", "limit": 12})
    assert snippet_count(twelve) == 12


async def test_search_returns_five_when_it_asks_for_nothing(memory: Memory) -> None:
    """The default is the tool's, and passing it down must not have changed it."""
    await stock(memory, 15, "escalation")

    default = await memory.call("memory_search", {"query": "escalation runbook"})
    assert snippet_count(default) == 5


async def test_a_limit_below_one_is_refused_before_the_handler_sees_it(memory: Memory) -> None:
    """Why the handler needs no floor: `limit` now bounds packing, and a zero
    there would report an empty pack as "nothing matched"."""
    result = await memory.call("memory_search", {"query": "deploy pipeline", "limit": 0})
    assert "less than the minimum" in result


async def pin(memory: Memory, **fields: Any) -> str:
    """Add a pinned memory to the corpus and reindex, returning its id."""
    doc = MemoryDoc.new(pinned=True, **fields)
    target = memory.root / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    await MemoryIndex(memory.store, memory.root).reindex()
    return doc.id


async def test_search_does_not_lead_with_an_unrelated_pinned_memory(memory: Memory) -> None:
    """#57. The tool promises ranked snippets, so the top hit must be the best one."""
    standing = await pin(
        memory,
        type="fact",
        title="House style",
        body="Answer in British English and keep replies under four sentences.",
    )

    result = await memory.call("memory_search", {"query": "who owns the deploy pipeline"})

    assert result.startswith(f"[[{id_of(memory, 'memory/people/jane-okafor.md')}]]")
    assert standing not in result


async def test_a_pinned_memory_still_ranks_when_it_matches(memory: Memory) -> None:
    """Excluded from the pool, not from the results: the query can still find it."""
    standing = await pin(
        memory,
        type="fact",
        title="Deploy freeze",
        body="No deploys go out between the 20th of December and the 2nd of January.",
    )

    result = await memory.call("memory_search", {"query": "deploy freeze over christmas"})
    assert standing in result


async def test_pinned_memories_do_not_crowd_out_the_answer(memory: Memory) -> None:
    """Five standing instructions used to fill the default limit exactly."""
    for n in range(5):
        await pin(
            memory,
            type="fact",
            title=f"House style {n}",
            body=f"Standing instruction number {n}, about nothing in particular.",
        )

    result = await memory.call("memory_search", {"query": "who owns the deploy pipeline"})
    assert "Jane" in result


# -- memory_read -------------------------------------------------------------


async def test_read_returns_the_whole_file(memory: Memory) -> None:
    jane = id_of(memory, "memory/people/jane-okafor.md")
    result = await memory.call("memory_read", {"memory_id": jane})

    assert result.startswith("---")
    assert "reviews every change" in result


async def test_reading_a_private_memory_from_a_public_scope_is_refused(memory: Memory) -> None:
    """And refused the same way a missing one is, so the id itself leaks nothing."""
    secret = id_of(memory, "memory/facts/salary-review-outcome.md")

    refused = await memory.call("memory_read", {"memory_id": secret})
    missing = await memory.call("memory_read", {"memory_id": new_memory_id()})

    assert "band 5" not in refused
    assert refused.split()[:-1] == missing.split()[:-1], "indistinguishable but for the id"


async def test_reading_a_private_memory_from_its_own_scope_works(memory: Memory) -> None:
    secret = id_of(memory, "memory/facts/salary-review-outcome.md")
    result = await memory.call("memory_read", {"memory_id": secret}, scope="private:U01")
    assert "band 5" in result


@pytest.mark.parametrize(
    "memory_id", ["../../etc/passwd", "memory/people/jane.md", "", "not-an-id", "mem_short"]
)
async def test_read_only_accepts_memory_ids(memory: Memory, memory_id: str) -> None:
    result = await memory.call("memory_read", {"memory_id": memory_id})
    assert "is not a memory id" in result


async def test_read_follows_the_supersedes_chain(memory: Memory, store: Store) -> None:
    """A snippet quoted from an old conversation still resolves after a merge."""
    old_id = new_memory_id()
    successor = MemoryDoc.new(type="topic", title="Merged", body="Now here.", supersedes=[old_id])
    target = memory.root / successor.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(successor.render())
    Manifest.rebuild(memory.root)[0].save(memory.root)

    assert "Now here." in await memory.call("memory_read", {"memory_id": old_id})


# -- memory_write ------------------------------------------------------------


async def test_write_enqueues_an_observation(memory: Memory, store: Store) -> None:
    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "Bob", "claim": "Bob runs the incident rota."},
        session_id="cli:1",
    )

    assert "queued" in result
    pending = await store.pending_observations()
    # The subject is stored normalized: it is the key `promote` groups by, and
    # "Bob" typed here has to meet "Bob's" extracted from a conversation. The
    # claim is not — it is prose a person will read in a memory file.
    assert [(o["subject"], o["claim"], o["kind"]) for o in pending] == [
        ("bob", "Bob runs the incident rota.", "fact")
    ]


async def test_write_never_produces_a_commit(memory: Memory) -> None:
    """The acceptance criterion. The agent proposes; `promote` disposes."""
    before = memory.repo.head()

    await memory.call(
        "memory_write", {"kind": "decision", "subject": "Deploys", "claim": "Nightly at 02:00."}
    )

    assert memory.repo.head() == before
    assert not memory.repo.is_dirty(), "nothing was written to the working copy either"


async def test_an_observation_inherits_the_session_scope(memory: Memory, store: Store) -> None:
    """Something said in a DM must not become general knowledge."""
    await store.ensure_session("slack:U01", surface="slack", scope="private:U01")
    await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "Pay", "claim": "They asked for a raise."},
        scope="private:U01",
        session_id="slack:U01",
    )

    observation = (await store.pending_observations())[0]
    assert observation["scope"] == "private:U01"
    assert observation["session_id"] == "slack:U01"


async def test_the_model_cannot_choose_the_scope(memory: Memory, store: Store) -> None:
    """`scope` is not an argument, so a plan to widen one has nowhere to land."""
    result = await memory.call(
        "memory_write",
        {"kind": "fact", "subject": "X", "claim": "Y", "scope": "workspace"},
        scope="private:U01",
    )

    assert "invalid arguments" in result, "the schema forbids unknown properties"
    assert await store.pending_observations() == []


async def test_an_unknown_kind_is_rejected(memory: Memory, store: Store) -> None:
    result = await memory.call(
        "memory_write", {"kind": "instruction", "subject": "X", "claim": "Y"}
    )
    assert "invalid arguments" in result
    assert await store.pending_observations() == []


async def test_an_observation_survives_an_unknown_session(memory: Memory, store: Store) -> None:
    """Losing the fact because the bookkeeping link is missing is the wrong trade."""
    await memory.call(
        "memory_write", {"kind": "fact", "subject": "X", "claim": "Y"}, session_id="never-existed"
    )

    observation = (await store.pending_observations())[0]
    assert observation["claim"] == "Y"
    assert observation["session_id"] is None


async def test_observations_start_pending_with_no_episode(memory: Memory, store: Store) -> None:
    """#27 attaches the episode; the interactive path has only a session."""
    await memory.call("memory_write", {"kind": "fact", "subject": "X", "claim": "Y"})

    observation = (await store.pending_observations())[0]
    assert observation["state"] == "pending"
    assert observation["episode_id"] is None
    assert json.loads(observation["source_refs"]) == []


# -- acceptance: the agent recovers from a pre-injection miss ----------------


class Scripted:
    """A provider that calls memory_search, then answers from what came back."""

    name = "scripted"
    model = "m"

    def __init__(self) -> None:
        self.tool_results: list[str] = []
        self.contexts: list[str | None] = []
        self._turn = 0

    async def complete(self, req: ChatRequest) -> ChatResponse:  # pragma: no cover - unused
        raise NotImplementedError

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        self._turn += 1
        self.contexts.append(req.context)
        for message in req.messages:
            for block in message.content:
                if isinstance(block, ToolResultBlock):
                    self.tool_results.append(block.content)

        if self._turn == 1:
            yield ToolUseStart(id="t1", name="memory_search")
            yield ToolUseArgsDelta(id="t1", partial_json='{"query": "deploy pipeline owner"}')
            yield ToolUseStop(id="t1")
            yield MessageStop(stop_reason="tool_use", usage=Usage(), model="m")
            return

        answer = "Jane owns it." if any("Jane" in r for r in self.tool_results) else "I don't know."
        yield TextDelta(text=answer)
        yield MessageStop(stop_reason="end_turn", usage=Usage(), model="m")

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - unused
        return [[0.0] for _ in texts]

    async def aclose(self) -> None:
        return None


async def test_the_agent_can_answer_from_tools_when_injection_missed(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """The acceptance criterion for #16.

    The question shares no words with the memory, so pre-injection returns
    nothing. The agent has to find it by searching, in its own words.
    """
    provider = Scripted()
    retriever = Retriever(store, tokenizer=tokenizer)
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=retriever,
    )

    result = await agent.respond("cli:1", "Who should I nudge about shipping?")

    assert provider.contexts[0] is None, "pre-injection found nothing to inject"
    assert result.tool_calls == 1
    assert result.text == "Jane owns it."


async def test_pre_injection_puts_memory_in_the_context(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """The other ninety percent: no tool call needed."""
    provider = Scripted()
    provider._turn = 1  # skip straight to answering
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Retriever(store, tokenizer=tokenizer),
    )

    await agent.respond("cli:2", "Who owns the deploy pipeline?")

    assert provider.contexts[0] is not None
    assert "Jane" in provider.contexts[0]


async def test_a_broken_retriever_degrades_the_turn_rather_than_ending_it(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    """Answering without memory beats refusing to answer."""

    class Broken(Retriever):
        async def retrieve(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("index is on fire")

    provider = Scripted()
    provider._turn = 1
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Broken(store, tokenizer=tokenizer),
    )

    result = await agent.respond("cli:3", "Who owns the deploy pipeline?")
    assert result.text == "I don't know."


async def test_pre_injection_respects_the_session_scope(
    memory: Memory, store: Store, tokenizer: Tokenizer
) -> None:
    provider = Scripted()
    provider._turn = 1
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=memory.registry,
        packer=ContextPacker(tokenizer=tokenizer),
        retriever=Retriever(store, tokenizer=tokenizer),
    )

    await agent.respond("cli:4", "what was the salary band outcome?")
    injected = provider.contexts[0] or ""
    assert id_of(memory, "memory/facts/salary-review-outcome.md") not in injected


# -- a write tool must not confirm a no-op (#79) ------------------------------


async def test_whitespace_is_not_a_claim(memory: Memory, store: Store) -> None:
    """#79. Both fields were stripped and written, and the model was told the
    write had succeeded — leaving an observation with nothing in it `pending`
    for the promote job to deal with.

    `minLength` in the schema catches the empty string upstream; it cannot see
    a string of spaces, so the handler checks the stripped value too."""
    result = await memory.call("memory_write", {"kind": "fact", "subject": " ", "claim": "\t\n"})

    assert "must each say something" in result
    assert "queued" not in result
    assert await store.pending_observations() == []


@pytest.mark.parametrize(
    "args",
    [
        {"kind": "fact", "subject": "", "claim": ""},
        {"kind": "fact", "subject": "Jane", "claim": ""},
        {"kind": "fact", "subject": "", "claim": "Jane owns deploys."},
    ],
)
async def test_an_empty_field_is_refused_by_the_schema(
    memory: Memory, store: Store, args: dict[str, Any]
) -> None:
    """Upstream, the way the `kind` enum is — so it arrives as an error rather
    than as an answer."""
    result = await memory.call("memory_write", args)

    assert "invalid arguments" in result
    assert await store.pending_observations() == []

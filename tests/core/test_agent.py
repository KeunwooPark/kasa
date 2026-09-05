from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from kasa.core.agent import Agent, AgentConfig, AgentResult
from kasa.core.context import ContextPacker
from kasa.core.tools import Tool, ToolContext, ToolRegistry
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.tokens import Tokenizer
from kasa.llm.types import (
    ChatRequest,
    ChatResponse,
    Delta,
    Message,
    MessageStop,
    TextBlock,
    TextDelta,
    ToolUseArgsDelta,
    ToolUseBlock,
    ToolUseStart,
    ToolUseStop,
    Usage,
)
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.index import MemoryIndex
from kasa.memory.retrieve import Retriever
from kasa.redact import Redactor
from kasa.store import Store

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"city": {"type": "string"}}}


def says(text: str) -> ChatResponse:
    return ChatResponse(
        message=Message.assistant(text),
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="m",
    )


def calls(*names: str, text: str = "") -> ChatResponse:
    blocks: list[Any] = [TextBlock(text=text)] if text else []
    blocks += [
        ToolUseBlock(id=f"t{i}", name=name, input={"city": "Seoul"}) for i, name in enumerate(names)
    ]
    return ChatResponse(
        message=Message(role="assistant", content=tuple(blocks)),
        stop_reason="tool_use",
        usage=Usage(input_tokens=10, output_tokens=5),
        model="m",
    )


class ScriptedProvider:
    """Streams a fixed list of responses, one per turn."""

    name = "scripted"
    model = "m"

    def __init__(self, script: list[ChatResponse]) -> None:
        self.script = list(script)
        self.requests: list[ChatRequest] = []

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        return self.script.pop(0)

    async def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:
        self.requests.append(req)
        response = self.script.pop(0)
        for block in response.message.content:
            if isinstance(block, TextBlock):
                yield TextDelta(text=block.text)
            elif isinstance(block, ToolUseBlock):
                yield ToolUseStart(id=block.id, name=block.name)
                # Split mid-key, as both real APIs do.
                raw = f'{{"city": "{block.input.get("city", "")}"}}'
                yield ToolUseArgsDelta(id=block.id, partial_json=raw[:6])
                yield ToolUseArgsDelta(id=block.id, partial_json=raw[6:])
                yield ToolUseStop(id=block.id)
        yield MessageStop(
            stop_reason=response.stop_reason, usage=response.usage, model=response.model
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def aclose(self) -> None:
        return None


async def weather(args: dict[str, Any], context: ToolContext) -> str:
    return f"4C in {args.get('city')}"


async def hang(args: dict[str, Any], context: ToolContext) -> str:
    await asyncio.sleep(30)
    return "never"


def build(
    store: Store,
    tokenizer: Tokenizer,
    script: list[ChatResponse],
    *,
    tools: list[Tool] | None = None,
    config: AgentConfig | None = None,
    retriever: Retriever | None = None,
    inbound_scrub: Any = None,
) -> tuple[Agent, ScriptedProvider]:
    provider = ScriptedProvider(script)
    agent = Agent(
        registry=ProviderRegistry({ModelRole.CHAT: [provider]}),
        store=store,
        tools=ToolRegistry(
            tools or [Tool(name="weather", description="d", input_schema=SCHEMA, handler=weather)]
        ),
        packer=ContextPacker(tokenizer=tokenizer),
        config=config,
        retriever=retriever,
        inbound_scrub=inbound_scrub,
    )
    return agent, provider


async def test_current_secret_is_visible_once_but_only_redaction_is_stored(
    tmp_path: Path, tokenizer: Tokenizer
) -> None:
    secret = "sk-ant-this-is-the-current-turn-secret"
    redactor = Redactor()
    async with await Store.open(tmp_path / "guarded.db", scrub=redactor.scrub) as guarded:
        agent, provider = build(
            guarded, tokenizer, [says("It has the expected shape.")], inbound_scrub=redactor.scrub
        )
        result = await agent.respond("s1", f"is {secret} valid?", surface="slack")
        stored = await guarded.recent_messages("s1")

    assert secret in provider.requests[0].messages[-1].text
    assert all(secret not in message.model_dump_json() for message in stored)
    assert "did not store it" in (result.note or "")


async def transcript(store: Store, session: str) -> list[tuple[str, str]]:
    """(role, kind) for every stored message, in order."""
    out = []
    for msg in await store.recent_messages(session, limit=100):
        if msg.tool_uses:
            kind = "tool_use"
        elif msg.tool_results_in:
            kind = "tool_result"
        else:
            kind = "text"
        out.append((msg.role, kind))
    return out


async def test_plain_turn(store: Store, tokenizer: Tokenizer) -> None:
    agent, _ = build(store, tokenizer, [says("hello")])
    result = await agent.respond("s1", "hi")

    assert result.text == "hello"
    assert result.iterations == 1
    assert await transcript(store, "s1") == [("user", "text"), ("assistant", "text")]


async def test_multi_tool_turn_is_reconstructible_from_the_db(
    store: Store, tokenizer: Tokenizer
) -> None:
    """The acceptance criterion for the turn loop.

    After the fact, the stored transcript alone must be enough to replay the
    exchange — every tool call present, in order, each with its result.
    """
    agent, provider = build(
        store, tokenizer, [calls("weather", "weather", text="checking"), says("It is 4C.")]
    )
    result = await agent.respond("s1", "weather in Seoul?")

    assert result.text == "It is 4C."
    assert result.tool_calls == 2
    assert result.iterations == 2
    assert await transcript(store, "s1") == [
        ("user", "text"),
        ("assistant", "tool_use"),
        ("user", "tool_result"),
        ("assistant", "text"),
    ]

    stored = await store.recent_messages("s1", limit=100)
    used = {b.id for m in stored for b in m.tool_uses}
    answered = {b.tool_use_id for m in stored for b in m.tool_results_in}
    assert used == answered == {"t0", "t1"}
    assert [b.content for m in stored for b in m.tool_results_in] == ["4C in Seoul"] * 2

    # The second call carried the first turn's history forward.
    assert len(provider.requests[1].messages) > len(provider.requests[0].messages)


async def test_streamed_arguments_are_reassembled(store: Store, tokenizer: Tokenizer) -> None:
    agent, _ = build(store, tokenizer, [calls("weather"), says("done")])
    await agent.respond("s1", "weather?")

    stored = await store.recent_messages("s1", limit=100)
    tool_use = next(b for m in stored for b in m.tool_uses)
    assert tool_use.input == {"city": "Seoul"}


async def test_deltas_reach_the_sink(store: Store, tokenizer: Tokenizer) -> None:
    agent, _ = build(store, tokenizer, [says("streamed text")])
    seen: list[Delta] = []

    await agent.respond("s1", "hi", on_delta=lambda d: _collect(seen, d))

    assert "".join(d.text for d in seen if isinstance(d, TextDelta)) == "streamed text"
    assert any(isinstance(d, MessageStop) for d in seen)


async def _collect(sink: list[Delta], delta: Delta) -> None:
    sink.append(delta)


async def test_tool_errors_are_fed_back_to_the_model(store: Store, tokenizer: Tokenizer) -> None:
    agent, _ = build(
        store,
        tokenizer,
        [calls("missing_tool"), says("sorry, I cannot")],
    )
    result = await agent.respond("s1", "do a thing")

    stored = await store.recent_messages("s1", limit=100)
    results = [b for m in stored for b in m.tool_results_in]
    assert results[0].is_error
    assert "unknown tool" in results[0].content
    # The loop keeps going so the model can recover.
    assert result.text == "sorry, I cannot"


async def test_iteration_limit_still_answers_outstanding_calls(
    store: Store, tokenizer: Tokenizer
) -> None:
    """Stopping mid-tool-call must not leave an unanswered `tool_use` behind."""
    agent, _ = build(
        store,
        tokenizer,
        [calls("weather") for _ in range(5)],
        config=AgentConfig(max_tool_iterations=2),
    )
    result = await agent.respond("s1", "loop forever")

    assert result.stop_reason == "max_iterations"
    stored = await store.recent_messages("s1", limit=100)
    used = {b.id for m in stored for b in m.tool_uses}
    answered = {b.tool_use_id for m in stored for b in m.tool_results_in}
    assert used == answered


async def test_cancellation_leaves_no_unanswered_tool_use(
    store: Store, tokenizer: Tokenizer
) -> None:
    """A turn aborted mid-dispatch must leave a transcript that still replays.

    An assistant `tool_use` with no matching `tool_result` is rejected by both
    provider families, so it would break the session permanently.
    """
    agent, _ = build(
        store,
        tokenizer,
        [calls("hang")],
        tools=[Tool(name="hang", description="d", input_schema=SCHEMA, handler=hang)],
    )

    task = asyncio.create_task(agent.respond("s1", "start something slow"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    stored = await store.recent_messages("s1", limit=100)
    used = {b.id for m in stored for b in m.tool_uses}
    answered = {b.tool_use_id for m in stored for b in m.tool_results_in}
    assert used == answered
    assert any(b.is_error for m in stored for b in m.tool_results_in)


async def test_session_history_carries_across_turns(store: Store, tokenizer: Tokenizer) -> None:
    agent, provider = build(store, tokenizer, [says("one"), says("two")])
    await agent.respond("s1", "first")
    await agent.respond("s1", "second")

    assert [m.text for m in provider.requests[1].messages] == [
        "first",
        "one",
        "second",
    ]


async def test_usage_accumulates_across_iterations(store: Store, tokenizer: Tokenizer) -> None:
    agent, _ = build(store, tokenizer, [calls("weather"), says("done")])
    result = await agent.respond("s1", "weather?")

    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10


async def test_system_prompt_is_identical_across_turns(store: Store, tokenizer: Tokenizer) -> None:
    """Prompt caching depends on this; assert it at the request level."""
    agent, provider = build(store, tokenizer, [says("one"), says("two")])
    await agent.respond("s1", "first")
    await agent.respond("s1", "second")

    assert provider.requests[0].system == provider.requests[1].system


async def test_a_scheduled_turn_says_so_in_the_system_block(
    store: Store, tokenizer: Tokenizer
) -> None:
    """A standing task's fire arrives as a user message nobody typed (#179).
    Without this the model's only reading of "the overnight AI news" appearing
    from nowhere is that it was just asked for, and it opens by thanking
    somebody who has been asleep for eight hours."""
    agent, provider = build(store, tokenizer, [says("Three things happened."), says("and again")])

    await agent.respond("s1", "the overnight AI news", origin="scheduled")
    await agent.respond("s1", "and now?")

    assert "standing task" in provider.requests[0].system
    # Only that turn. A scheduled fire in a thread does not change what the
    # next thing somebody actually says is answered against.
    assert "standing task" not in provider.requests[1].system
    assert provider.requests[1].system == agent.config.system_prompt


async def test_twenty_turn_session_exceeds_eighty_percent_cache_hits(
    store: Store, tokenizer: Tokenizer
) -> None:
    responses = [
        ChatResponse(
            message=Message.assistant(str(turn)),
            stop_reason="end_turn",
            usage=Usage(
                input_tokens=10,
                output_tokens=1,
                cache_write_tokens=100 if turn == 0 else 0,
                cache_read_tokens=0 if turn == 0 else 100,
            ),
            model="m",
        )
        for turn in range(20)
    ]
    agent, provider = build(store, tokenizer, responses)

    for turn in range(20):
        await agent.respond("long-session", f"turn {turn}")

    assert len({request.system.encode() for request in provider.requests if request.system}) == 1
    assert agent.registry.meter.session_cache_hit_rate("long-session") > 0.8


# -- what the user is told when a turn does not simply end -------------------


async def test_hitting_the_iteration_limit_leaves_something_to_show_the_user(
    store: Store, tokenizer: Tokenizer
) -> None:
    """#46: the loop handled the cap correctly and nobody read the result.

    A model that only ever calls tools produced an empty reply, a dim tool-count
    line, and a fresh prompt — no answer and no reason for its absence.
    """
    agent, _ = build(
        store,
        tokenizer,
        [calls("weather") for _ in range(5)],
        config=AgentConfig(max_tool_iterations=2),
    )
    result = await agent.respond("s1", "loop forever")

    assert result.text == ""
    assert result.note is not None
    assert "tool call" in result.note


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("max_iterations", "without an answer"),
        ("max_tokens", "cut off"),
        ("content_filter", "stopped this reply"),
        ("tool_use", "never run"),
    ],
)
def test_every_way_a_turn_can_end_badly_has_something_to_say(
    stop_reason: str, expected: str
) -> None:
    assert expected in (AgentResult(text="", stop_reason=stop_reason).note or "")


def test_an_ordinary_answer_says_nothing_extra() -> None:
    assert AgentResult(text="Jane owns it.").note is None


def test_an_empty_reply_that_ended_normally_is_still_worth_naming() -> None:
    """Same symptom from the user's side: a prompt that answered nothing."""
    assert AgentResult(text="   ").note == "the model returned nothing."


# -- retrieval reaches the prompt scrubbed (#67) ------------------------------


async def test_a_credential_in_memory_does_not_reach_the_provider(
    tmp_path: Path, store: Store, tokenizer: Tokenizer
) -> None:
    """End to end, the way `kasa run` wires it: corpus -> retriever -> prompt.

    The pre-injected path is the one every turn takes, and it was the one
    nothing scrubbed. Asserting on the request the provider actually received,
    because that is the only place the question "was it sent?" has an answer.
    """
    bootstrap(tmp_path)
    doc = MemoryDoc.new(
        type="topic",
        title="Staging deploy key rotation",
        tags=["infra"],
        body="The staging runner authenticates with AKIAIOSFODNN7EXAMPLE. "
        "Rotate it when the migration lands.",
    )
    (tmp_path / doc.suggested_path()).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / doc.suggested_path()).write_text(doc.render())
    await MemoryIndex(store, tmp_path).reindex()

    agent, provider = build(
        store,
        tokenizer,
        [says("noted")],
        retriever=Retriever(store, tokenizer=tokenizer, scrub=Redactor().scrub),
    )
    await agent.respond("s1", "how does the staging runner authenticate?")

    sent = provider.requests[0]
    everything = f"{sent.system}\n{sent.context}"
    assert "Rotate it when the migration lands." in everything, "memory was retrieved"
    assert "AKIAIOSFODNN7EXAMPLE" not in everything

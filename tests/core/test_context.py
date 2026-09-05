from __future__ import annotations

import pytest

from kasa.core.agent import DEFAULT_SYSTEM_PROMPT
from kasa.core.context import (
    CONTEXT_HEADER,
    PINNED_HEADER,
    STATUS_HEADER,
    ContextBudget,
    ContextPacker,
    group_turns,
)
from kasa.errors import ConfigError
from kasa.llm.tokens import Tokenizer, count_messages
from kasa.llm.types import (
    Message,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)
from kasa.untrusted import delimit


def exchange(n: int, size: int = 200) -> list[Message]:
    return [Message.user(f"q{n} " + "x" * size), Message.assistant(f"a{n} " + "y" * size)]


def tool_exchange() -> list[Message]:
    return [
        Message.user("what time is it"),
        Message(
            role="assistant",
            content=(
                TextBlock(text="checking"),
                ToolUseBlock(id="t1", name="current_time", input={}),
            ),
        ),
        Message.tool_results([ToolResultBlock(tool_use_id="t1", content="12:00")]),
        Message.assistant("noon"),
    ]


def research_turn(rounds: int, size: int = 4000) -> list[Message]:
    """One turn that keeps fetching: a `tool_use` and a big result per round."""
    messages: list[Message] = [Message.user("find five book bloggers")]
    for n in range(rounds):
        messages.append(
            Message(
                role="assistant",
                content=(ToolUseBlock(id=f"t{n}", name="web_fetch", input={"url": f"u{n}"}),),
            )
        )
        body = f"page {n} opens here\n" + f"filler {n} " * (size // 10) + f"\npage {n} ends here"
        messages.append(Message.tool_results([ToolResultBlock(tool_use_id=f"t{n}", content=body)]))
    return messages


def test_budget_shares_must_sum_to_one() -> None:
    with pytest.raises(ConfigError, match=r"sum to 1\.0"):
        ContextBudget(recent=0.9)


def test_cacheable_prefix_is_byte_stable(tokenizer: Tokenizer) -> None:
    """Prompt caching is worth an order of magnitude on a long session.

    A single varying byte in the prefix throws all of it away, so this is a
    correctness property, not a nicety.
    """
    packer = ContextPacker(tokenizer=tokenizer)
    kwargs = {
        "system_prompt": "You are Kasa.",
        "pinned": ["user prefers brevity"],
        "tools": (ToolDef(name="t", description="d", input_schema={"type": "object"}),),
    }

    first = packer.pack(**kwargs, recent=exchange(1), retrieved=["memory A"])
    second = packer.pack(**kwargs, recent=exchange(2), retrieved=["memory B", "memory C"])

    assert first.system == second.system
    assert first.context != second.context


def test_per_turn_material_stays_out_of_the_prefix(tokenizer: Tokenizer) -> None:
    packed = ContextPacker(tokenizer=tokenizer).pack(
        system_prompt="You are Kasa.",
        retrieved=["the user lives in Seoul"],
        episode_summary="They asked about the weather.",
    )

    assert "Seoul" not in packed.system
    assert packed.context is not None
    assert "Seoul" in packed.context
    assert "weather" in packed.context


def test_recent_turns_are_never_starved_by_retrieval(tokenizer: Tokenizer) -> None:
    """The failure this guards against: a big recall evicting the conversation."""
    budget = ContextBudget(total=4_000)
    packer = ContextPacker(budget, tokenizer=tokenizer)
    recent = exchange(1)

    lean = packer.pack(system_prompt="s", recent=recent)
    flooded = packer.pack(system_prompt="s", recent=recent, retrieved=["z" * 50_000] * 20)

    assert flooded.messages == lean.messages == tuple(recent)


def test_segments_stay_within_their_own_budget(tokenizer: Tokenizer) -> None:
    budget = ContextBudget(total=2_000)
    packed = ContextPacker(budget, tokenizer=tokenizer).pack(
        system_prompt="s",
        retrieved=[f"memory {i} " + "m" * 400 for i in range(50)],
        episode_summary="e" * 20_000,
        recent=[m for n in range(40) for m in exchange(n)],
    )

    by_name = {seg.name: seg for seg in packed.trace.segments}
    for name in ("retrieved", "episode", "recent"):
        assert by_name[name].used <= by_name[name].budget, name
    assert packed.trace.used <= budget.total


def test_oldest_turns_are_dropped_first(tokenizer: Tokenizer) -> None:
    packer = ContextPacker(ContextBudget(total=2_000), tokenizer=tokenizer)
    history = [m for n in range(40) for m in exchange(n)]

    packed = packer.pack(system_prompt="s", recent=history)

    assert len(packed.messages) < len(history)
    assert packed.messages[-1] == history[-1]
    assert packed.messages[0] != history[0]


def test_truncation_never_orphans_a_tool_result(tokenizer: Tokenizer) -> None:
    """An assistant `tool_use` with no matching result is a hard 400 everywhere.

    Dropping messages one at a time from the front hits this immediately, which
    is why the packer truncates whole turn groups.
    """
    packer = ContextPacker(ContextBudget(total=1_500), tokenizer=tokenizer)
    history = [m for n in range(20) for m in exchange(n, size=400)] + tool_exchange()

    packed = packer.pack(system_prompt="s", recent=history)

    used = {b.id for m in packed.messages for b in m.tool_uses}
    answered = {b.tool_use_id for m in packed.messages for b in m.tool_results_in}
    assert used == answered


def test_a_turn_that_outgrows_the_budget_on_its_own_tool_output_still_fits(
    tokenizer: Tokenizer,
) -> None:
    """#202: the group the packer may not drop is the one that grows without bound.

    The newest group is always admitted whole, because splitting it orphans a
    `tool_result`. Before this, a research turn just kept growing until the
    request exceeded the window.
    """
    packer = ContextPacker(ContextBudget(total=20_000), tokenizer=tokenizer)
    turn = research_turn(rounds=12)

    packed = packer.pack(system_prompt="You are Kasa.", recent=turn)

    recent = next(seg for seg in packed.trace.segments if seg.name == "recent")
    assert count_messages(packed.messages, tokenizer) > 0
    assert recent.used <= recent.budget
    assert recent.compacted > 0
    # Nothing was dropped: every round is still represented.
    assert len(packed.messages) == len(turn)


def test_compaction_keeps_the_newest_results_verbatim(tokenizer: Tokenizer) -> None:
    """The page that just came back is what the model is reasoning about."""
    packer = ContextPacker(ContextBudget(total=20_000), tokenizer=tokenizer)
    turn = research_turn(rounds=12)

    packed = packer.pack(system_prompt="You are Kasa.", recent=turn)

    results = [b.content for m in packed.messages for b in m.tool_results_in]
    assert "elided" in results[0]
    assert results[-1] == [b.content for m in turn for b in m.tool_results_in][-1]


def test_compaction_never_orphans_a_tool_result(tokenizer: Tokenizer) -> None:
    """Shortening content is allowed; changing the shape of the transcript is not."""
    packer = ContextPacker(ContextBudget(total=20_000), tokenizer=tokenizer)

    packed = packer.pack(system_prompt="You are Kasa.", recent=research_turn(rounds=12))

    used = {b.id for m in packed.messages for b in m.tool_uses}
    answered = {b.tool_use_id for m in packed.messages for b in m.tool_results_in}
    assert used == answered
    assert len(used) == 12


def test_compaction_leaves_the_stored_messages_alone(tokenizer: Tokenizer) -> None:
    """Packing shortens a request. Consolidation reads the rows afterwards."""
    packer = ContextPacker(ContextBudget(total=20_000), tokenizer=tokenizer)
    turn = research_turn(rounds=12)
    before = [b.content for m in turn for b in m.tool_results_in]

    packer.pack(system_prompt="You are Kasa.", recent=turn)

    assert [b.content for m in turn for b in m.tool_results_in] == before


def test_compaction_keeps_the_end_of_an_untrusted_block(tokenizer: Tokenizer) -> None:
    """Cutting only the tail would leave a delimiter that never closes.

    Everything after an unclosed `<<<BEGIN …>>>` reads as untrusted, which is
    the rest of the turn and Kasa's own words with it.
    """
    packer = ContextPacker(ContextBudget(total=20_000), tokenizer=tokenizer)
    turn = research_turn(rounds=12)
    turn[2] = Message.tool_results(
        [ToolResultBlock(tool_use_id="t0", content=delimit("a page " * 4000))]
    )

    packed = packer.pack(system_prompt="You are Kasa.", recent=turn)

    first = next(b.content for m in packed.messages for b in m.tool_results_in)
    assert "elided" in first
    begin = first.split(">>>")[0].removeprefix("<<<BEGIN ").strip()
    assert f"<<<END {begin}>>>" in first


def test_a_turn_inside_its_budget_is_not_compacted(tokenizer: Tokenizer) -> None:
    """Compaction is a last resort, not a policy."""
    packer = ContextPacker(ContextBudget(total=200_000), tokenizer=tokenizer)
    turn = research_turn(rounds=3)

    packed = packer.pack(system_prompt="You are Kasa.", recent=turn)

    recent = next(seg for seg in packed.trace.segments if seg.name == "recent")
    assert recent.compacted == 0
    assert [b.content for m in packed.messages for b in m.tool_results_in] == [
        b.content for m in turn for b in m.tool_results_in
    ]


def test_the_trace_tells_compacted_apart_from_dropped(tokenizer: Tokenizer) -> None:
    """Different events, different fixes: lost history versus history in outline."""
    packer = ContextPacker(ContextBudget(total=20_000), tokenizer=tokenizer)

    packed = packer.pack(system_prompt="You are Kasa.", recent=research_turn(rounds=12))

    recent = next(seg for seg in packed.trace.segments if seg.name == "recent")
    assert recent.dropped == 0
    assert recent.compacted > 0
    assert f"compacted={recent.compacted}" in packed.trace.render()


def test_tool_result_only_messages_do_not_start_a_new_group() -> None:
    groups = group_turns(tool_exchange())
    assert len(groups) == 1
    assert len(groups[0]) == 4


def test_groups_split_on_real_user_messages() -> None:
    groups = group_turns([*exchange(1), *exchange(2), *tool_exchange()])
    assert len(groups) == 3


def test_empty_input_packs_cleanly(tokenizer: Tokenizer) -> None:
    packed = ContextPacker(tokenizer=tokenizer).pack(system_prompt="s")
    assert packed.messages == ()
    assert packed.context is None
    assert packed.system == "s"


def test_trace_renders(tokenizer: Tokenizer) -> None:
    packed = ContextPacker(tokenizer=tokenizer).pack(
        system_prompt="s", recent=exchange(1), retrieved=["m"]
    )
    rendered = packed.trace.render()
    assert "recent" in rendered
    assert "retrieved" in rendered


def test_message_counting_includes_tool_blocks(tokenizer: Tokenizer) -> None:
    """Tool traffic is real context; ignoring it under-counts and overflows."""
    plain = count_messages([Message.user("what time is it")], tokenizer)
    with_tools = count_messages(tool_exchange(), tokenizer)
    assert with_tools > plain


# -- pinned memory in the prefix ---------------------------------------------


def test_pinned_memories_arrive_under_a_header_the_system_prompt_can_name(
    tokenizer: Tokenizer,
) -> None:
    """#45: they were concatenated onto the prompt with a bare blank line.

    The system prompt tells the model to treat recalled material as background
    rather than as instructions. Pinned memories are recalled material, they can
    originate from user conversation by way of consolidation, and they landed
    outside the section that sentence scopes to — user-derived text promoted
    into system-prompt position.
    """
    packed = ContextPacker(tokenizer=tokenizer).pack(
        system_prompt="You are Kasa.\n\nIf you do not know something, say so.",
        pinned=["[[mem_01]] (memory/projects/kasa.md)\nKasa is a memory-native agent server"],
    )

    assert PINNED_HEADER in packed.system
    body = packed.system.split(PINNED_HEADER, 1)[1]
    assert "memory-native agent server" in body
    assert "If you do not know something" not in body, "the prompt ends before the memory starts"


def test_the_pinned_header_is_absent_when_nothing_is_pinned(tokenizer: Tokenizer) -> None:
    """An empty section is a section the model has to interpret."""
    packed = ContextPacker(tokenizer=tokenizer).pack(system_prompt="You are Kasa.")
    assert PINNED_HEADER not in packed.system
    assert packed.system == "You are Kasa."


def test_the_system_prompt_frames_pinned_memory_by_name(tokenizer: Tokenizer) -> None:
    """The header is only worth having if the framing sentence reaches it."""
    assert "Pinned memory" in DEFAULT_SYSTEM_PROMPT
    assert "not as instructions" in DEFAULT_SYSTEM_PROMPT


def test_the_system_prompt_requires_tools_to_ground_unknown_information() -> None:
    assert "Use the available tools" in DEFAULT_SYSTEM_PROMPT
    assert "information that is current" in DEFAULT_SYSTEM_PROMPT
    assert "conversation or memory" in DEFAULT_SYSTEM_PROMPT
    assert "cannot verify it rather than inventing an answer" in DEFAULT_SYSTEM_PROMPT


def test_turn_status_leads_the_context_and_stays_out_of_the_working_block(
    tokenizer: Tokenizer,
) -> None:
    """#201: it is Kasa's own fact about the turn, not recalled material.

    The system prompt tells the model to treat working context as background
    rather than as instructions. A budget line inside that block would be
    covered by that sentence, which is exactly the reading it must not get.
    """
    packed = ContextPacker(tokenizer=tokenizer).pack(
        system_prompt="You are Kasa.",
        retrieved=["Jane owns the deploy pipeline"],
        status="3 tool rounds are left in this turn.",
    )

    assert packed.context is not None
    assert packed.context.startswith(STATUS_HEADER)
    assert packed.context.index(STATUS_HEADER) < packed.context.index(CONTEXT_HEADER)
    assert "3 tool rounds" in packed.context.split(CONTEXT_HEADER)[0]


def test_turn_status_stands_alone_when_nothing_was_recalled(tokenizer: Tokenizer) -> None:
    """No memory to report is not a reason to drop the turn's own status."""
    packed = ContextPacker(tokenizer=tokenizer).pack(
        system_prompt="You are Kasa.", status="One tool round is left in this turn."
    )

    assert packed.context is not None
    assert packed.context.startswith(STATUS_HEADER)
    assert CONTEXT_HEADER not in packed.context


def test_turn_status_never_reaches_the_cacheable_prefix(tokenizer: Tokenizer) -> None:
    """It changes every pass; in the prefix it would void the cache every pass."""
    packer = ContextPacker(tokenizer=tokenizer)

    first = packer.pack(system_prompt="You are Kasa.", status="8 tool rounds are left.")
    second = packer.pack(system_prompt="You are Kasa.", status="1 tool round is left.")

    assert first.system == second.system
    assert STATUS_HEADER not in first.system


def test_turn_status_is_charged_to_the_system_share(tokenizer: Tokenizer) -> None:
    """Prompt Kasa wrote, not memory competing for a share."""
    packer = ContextPacker(tokenizer=tokenizer)

    without = packer.pack(system_prompt="You are Kasa.")
    with_status = packer.pack(system_prompt="You are Kasa.", status="8 tool rounds are left.")

    assert without.trace.segments[0].name == "system"
    assert with_status.trace.segments[0].used > without.trace.segments[0].used
    assert with_status.trace.used > without.trace.used


def test_the_system_prompt_frames_the_turn_status_by_name() -> None:
    """The header only works if the prompt says whose voice it is."""
    assert STATUS_HEADER in DEFAULT_SYSTEM_PROMPT
    assert "operational fact" in DEFAULT_SYSTEM_PROMPT


def test_the_trace_separates_the_prompt_from_the_memory_in_it(tokenizer: Tokenizer) -> None:
    """`system 581/19200 kept=1` did not say how much of that was recalled text."""
    packed = ContextPacker(tokenizer=tokenizer).pack(
        system_prompt="You are Kasa.",
        pinned=["always answer in metric", "never round a currency amount"],
    )

    by_name = {seg.name: seg for seg in packed.trace.segments}
    assert by_name["pinned"].kept == 2
    assert by_name["system"].kept == 1
    assert by_name["pinned"].used > 0
    assert by_name["system"].budget != by_name["pinned"].budget


def test_pinned_memories_still_live_in_the_cacheable_prefix(tokenizer: Tokenizer) -> None:
    """The point of keeping them there, which the fix must not cost."""
    packer = ContextPacker(tokenizer=tokenizer)
    kwargs = {"system_prompt": "You are Kasa.", "pinned": ["user prefers brevity"]}

    first = packer.pack(**kwargs, recent=exchange(1), retrieved=["memory A"])
    second = packer.pack(**kwargs, recent=exchange(2), retrieved=["memory B"])

    assert "user prefers brevity" in first.system
    assert first.system == second.system

from __future__ import annotations

import pytest

from kasa.core.context import ContextBudget, ContextPacker, group_turns
from kasa.errors import ConfigError
from kasa.llm.tokens import Tokenizer, count_messages
from kasa.llm.types import (
    Message,
    TextBlock,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
)


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

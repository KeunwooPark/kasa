from __future__ import annotations

from pathlib import Path

from kasa.llm.cost import CallRecord
from kasa.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from kasa.store import Store


async def test_migrations_apply_from_empty_and_are_idempotent(tmp_path: Path) -> None:
    store = await Store.open(tmp_path / "k.db")
    try:
        # `open` already migrated, so a second run must be a no-op.
        assert await store.migrate() == []
        tables = {
            row["name"]
            for row in await store.raw("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"sessions", "messages", "llm_calls", "schema_version"} <= tables
    finally:
        await store.close()


async def test_reopening_does_not_reapply(tmp_path: Path) -> None:
    path = tmp_path / "k.db"
    async with await Store.open(path) as first:
        await first.ensure_session("s1", surface="cli")
    async with await Store.open(path) as second:
        assert await second.get_session("s1") is not None


async def test_message_round_trip_preserves_every_block_type(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")
    original = Message(
        role="assistant",
        content=(
            TextBlock(text="checking"),
            ToolUseBlock(id="t1", name="current_time", input={"tz": "UTC"}),
        ),
    )
    await store.append_message("s1", original)
    await store.append_message(
        "s1", Message.tool_results([ToolResultBlock(tool_use_id="t1", content="noon")])
    )

    restored = await store.recent_messages("s1")
    assert restored[0] == original
    assert restored[1].tool_results_in[0].tool_use_id == "t1"


async def test_recent_messages_returns_oldest_first_within_the_limit(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")
    for i in range(10):
        await store.append_message("s1", Message.user(f"m{i}"))

    recent = await store.recent_messages("s1", limit=3)
    assert [m.text for m in recent] == ["m7", "m8", "m9"]


async def test_sessions_are_isolated(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")
    await store.ensure_session("s2", surface="cli")
    await store.append_message("s1", Message.user("only in s1"))

    assert await store.message_count("s2") == 0
    assert await store.message_count("s1") == 1


async def test_ensure_session_is_idempotent_and_bumps_activity(store: Store) -> None:
    await store.ensure_session("s1", surface="cli", scope="workspace")
    first = await store.get_session("s1")
    await store.ensure_session("s1", surface="cli")
    second = await store.get_session("s1")

    assert first is not None and second is not None
    assert first["created_at"] == second["created_at"]
    assert second["last_active"] >= first["last_active"]


async def test_clearing_a_session_keeps_the_session(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")
    await store.append_message("s1", Message.user("hi"))

    assert await store.clear_session("s1") == 1
    assert await store.message_count("s1") == 0
    assert await store.get_session("s1") is not None


async def test_call_records_aggregate(store: Store) -> None:
    for _ in range(2):
        await store.record_call(
            CallRecord(
                role="chat",
                provider="p",
                model="m",
                usage=Usage(input_tokens=10, output_tokens=5, cache_read_tokens=2),
                latency_ms=12,
                cost_usd=0.001,
                tag="agent.turn",
                ok=True,
            )
        )

    summary = await store.cost_summary()
    assert summary[0]["calls"] == 2
    assert summary[0]["input_tokens"] == 20
    assert summary[0]["cache_read_tokens"] == 4
    assert summary[0]["cost_usd"] == 0.002


async def test_failed_calls_record_their_error(store: Store) -> None:
    await store.record_call(
        CallRecord(
            role="chat",
            provider="p",
            model="m",
            usage=Usage(),
            latency_ms=1,
            cost_usd=None,
            tag=None,
            ok=False,
            error="AuthError",
        )
    )
    rows = await store.raw("SELECT ok, error FROM llm_calls")
    assert rows == [{"ok": 0, "error": "AuthError"}]

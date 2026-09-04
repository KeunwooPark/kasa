from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from kasa.core.events import InboundEvent
from kasa.core.inbox import Inbox
from kasa.errors import StoreError
from kasa.llm.cost import CallRecord
from kasa.llm.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage
from kasa.memory.observation import ObservationDraft
from kasa.redact import Redactor
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


async def test_store_scrubs_every_message_block_before_persistence(tmp_path: Path) -> None:
    secret = "sk-ant-this-is-a-persisted-secret"
    async with await Store.open(tmp_path / "scrubbed.db", scrub=Redactor().scrub) as guarded:
        await guarded.ensure_session("s1", surface="slack")
        await guarded.append_message(
            "s1",
            Message(
                role="assistant",
                content=(
                    TextBlock(text=f"saw {secret}"),
                    ToolUseBlock(id="t1", name="call", input={"authorization": secret}),
                ),
            ),
        )
        raw = await guarded.raw("SELECT content FROM messages WHERE session_id = ?", ("s1",))
        restored = await guarded.recent_messages("s1")

    assert secret not in raw[0]["content"]
    assert secret not in restored[0].model_dump_json()


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
    assert summary[0]["day"]
    assert summary[0]["job_kind"] == "agent"
    assert summary[0]["input_tokens"] == 20
    assert summary[0]["cache_read_tokens"] == 4
    assert summary[0]["cost_usd"] == 0.002


async def test_persisted_daily_spend_ignores_unpriced_and_older_calls(store: Store) -> None:
    await store.record_call(
        CallRecord(
            role="chat",
            provider="p",
            model="m",
            usage=Usage(),
            latency_ms=1,
            cost_usd=1.25,
            tag="agent.turn",
            ok=True,
        )
    )
    assert await store.spend_since("2000-01-01") == 1.25
    assert await store.spend_since("2999-01-01") == 0.0


async def test_call_records_report_cache_hit_rate_per_session(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")
    await store.ensure_session("s2", surface="cli")
    for session_id, usage in (
        ("s1", Usage(cache_write_tokens=100)),
        ("s1", Usage(cache_read_tokens=900)),
        ("s2", Usage(cache_write_tokens=50)),
    ):
        await store.record_call(
            CallRecord(
                role="chat",
                provider="p",
                model="m",
                usage=usage,
                latency_ms=1,
                cost_usd=0.0,
                tag="agent.turn",
                ok=True,
                session_id=session_id,
            )
        )

    summary = await store.session_cost_summary("s1")
    assert summary["calls"] == 2
    assert summary["cache_hit_rate"] == pytest.approx(0.9)
    assert (await store.session_cost_summary("s2"))["cache_hit_rate"] == 0.0


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


# -- a database that is not one (#87) -----------------------------------------


def sqlite_threads() -> int:
    """aiosqlite runs one worker thread per connection, and it is not a daemon,
    which is the whole reason a leaked one is fatal rather than untidy."""
    return sum(1 for t in threading.enumerate() if "_connection_worker_thread" in t.name)


async def test_a_file_that_is_not_a_database_is_refused_not_leaked(tmp_path: Path) -> None:
    """#87. `aiosqlite.connect` succeeds — opening is lazy — and starts a worker
    thread that is not a daemon. The first statement then fails, and leaking the
    connection there left the process alive at interpreter shutdown after the
    error had already been printed. Every command hung, `kasa doctor` included.
    """
    path = tmp_path / "kasa.db"
    path.write_bytes(b"this is not sqlite at all, not even close")
    before = sqlite_threads()

    with pytest.raises(StoreError) as caught:
        await Store.open(path)

    assert "file is not a database" in str(caught.value)
    assert str(path) in str(caught.value), "it names the file"
    assert "kasa reindex" in str(caught.value), "and says how to recover"
    # The whole bug, in one assertion.
    assert sqlite_threads() == before, "the connection thread outlived the failure"


async def test_a_truncated_database_is_refused_not_leaked(tmp_path: Path) -> None:
    """The realistic shape of this: a full disk, or a kill mid-write."""
    path = tmp_path / "kasa.db"
    async with await Store.open(path):
        pass
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 3])
    before = sqlite_threads()

    with pytest.raises(StoreError) as caught:
        await Store.open(path)

    assert "malformed" in str(caught.value)
    assert sqlite_threads() == before


async def test_a_directory_where_the_database_should_be_is_refused(tmp_path: Path) -> None:
    """The connect fails rather than a statement after it, so the thread was
    never the problem here — but the error read as a traceback instead of as
    the same sentence."""
    (tmp_path / "kasa.db").mkdir()

    with pytest.raises(StoreError) as caught:
        await Store.open(tmp_path / "kasa.db")

    assert "unable to open database file" in str(caught.value)


async def test_opening_a_good_database_still_works(tmp_path: Path) -> None:
    before = sqlite_threads()
    async with await Store.open(tmp_path / "kasa.db") as store:
        assert await store.raw("SELECT name FROM schema_version")
        assert sqlite_threads() == before + 1, "the fixture holds one open too"
    assert sqlite_threads() == before


# -- concurrency -------------------------------------------------------------
#
# One connection, many coroutines. `aiosqlite` runs the statements on a single
# worker thread, which is easy to mistake for serialization — it is not, and
# these are the two ways that showed.


async def test_a_drainer_and_an_adapter_do_not_interleave(store: Store) -> None:
    """#101. `lease_inbox` holds an `UPDATE … RETURNING` cursor across an await,
    and a commit landing there is an error rather than a wait:

        OperationalError: cannot commit transaction - SQL statements in progress
    """
    inbox = Inbox(store, lease_ttl=0.0)

    async def enqueue() -> None:
        for n in range(200):
            await inbox.enqueue(InboundEvent(source="slack", external_id=f"E{n}", session_id="s1"))

    async def drain() -> None:
        for _ in range(200):
            await inbox.lease(limit=5)

    await asyncio.gather(enqueue(), drain(), drain())


async def test_two_appends_to_one_session_do_not_collide(store: Store) -> None:
    """`append_message` reads `MAX(seq) + 1` and then inserts it. Actors keep a
    session to one turn at a time, but background jobs (#26) will write while a
    turn is running, and `messages` has `UNIQUE (session_id, seq)`."""
    await store.ensure_session("s1", surface="cli")

    await asyncio.gather(*(store.append_message("s1", Message.user(f"m{n}")) for n in range(50)))

    rows = await store.raw("SELECT seq FROM messages WHERE session_id = 's1' ORDER BY seq")
    assert [row["seq"] for row in rows] == list(range(1, 51))


async def test_a_method_may_call_another_one(store: Store) -> None:
    """The lock is re-entrant because the compound methods are written in terms
    of the simple ones. A plain lock deadlocks on the first of them."""
    await store.ensure_session("s1", surface="cli")

    await store.append_messages("s1", [Message.user("a"), Message.assistant("b")])
    observation = await store.add_observation(
        subject="jane",
        claim="owns the deploy pipeline",
        kind="fact",
        scope="workspace",
        session_id="s1",
    )

    assert await store.message_count("s1") == 2
    assert [row["id"] for row in await store.pending_observations()] == [observation]


# -- episodes ----------------------------------------------------------------


async def test_one_episode_stays_open_across_a_conversation(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")

    first = await store.ensure_episode("s1")
    await store.append_messages("s1", [Message.user("a"), Message.assistant("b")])

    assert await store.ensure_episode("s1") == first
    rows = await store.episode_messages(first)
    assert len(rows) == 2


async def test_closing_an_episode_writes_its_observations_with_it(store: Store) -> None:
    """One transaction. A close that commits without its observations discards
    what the episode was consolidated for and can never be retried: nothing
    reopens an episode, so no sweep will look at it again."""
    await store.ensure_session("s1", surface="cli")
    episode_id = await store.ensure_episode("s1")

    written = await store.close_episode(
        episode_id,
        summary="they talked about deploys",
        observations=[
            ObservationDraft(
                subject="Jane Doe",
                claim="Jane Doe owns the deploy pipeline.",
                kind="fact",
                scope="workspace",
            )
        ],
    )

    assert written is not None and len(written) == 1
    pending = await store.pending_observations()
    assert [(o["episode_id"], o["session_id"], o["subject"]) for o in pending] == [
        (episode_id, "s1", "jane doe")
    ]


async def test_an_episode_can_only_be_closed_once(store: Store) -> None:
    await store.ensure_session("s1", surface="cli")
    episode_id = await store.ensure_episode("s1")
    draft = ObservationDraft(subject="a", claim="a is a thing", kind="fact", scope="workspace")

    assert await store.close_episode(episode_id, observations=[draft]) is not None
    assert await store.close_episode(episode_id, observations=[draft]) is None
    assert len(await store.pending_observations()) == 1, "the loser wrote nothing"

"""Setting up a standing task by asking for one.

The tools are thin over `Tasks`, which #179 already covers. What is worth
testing here is the seam: that the destination comes off the session and
cannot be reached from a tool argument, that the confirmation is something a
person can check, and that a refusal comes back as something the model can
correct rather than as a dead turn.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kasa.config import Config, TaskSettings
from kasa.core.events import InboundEvent
from kasa.core.schedule_tools import schedule_tools
from kasa.core.tools import Tool, ToolContext, ToolRegistry
from kasa.llm.types import ToolUseBlock
from kasa.runner.jobs import default_specs
from kasa.runner.scheduler import Job, Scheduler
from kasa.runner.tasks import TASK_KIND, Tasks
from kasa.store import Store

#: The thread the person is speaking in. Everything a task inherits comes off
#: this and nothing else.
IN_A_CHANNEL = ToolContext(
    session_id="slack:T01:C0123:1756890000.123",
    scope="channel:C0123",
    author="U01",
    channel="C0123",
    reply_to="1756890000.123",
)

ASK = {
    "prompt": "search for what happened in AI overnight and give me the five that matter",
    "cron": "0 9 * * 1-5",
    "timezone": "Asia/Seoul",
}


def tools_for(store: Store, settings: TaskSettings | None = None) -> dict[str, Tool]:
    return {tool.name: tool for tool in schedule_tools(Tasks(store, settings))}


async def call(
    store: Store,
    name: str,
    args: dict[str, Any] | None = None,
    *,
    context: ToolContext = IN_A_CHANNEL,
    settings: TaskSettings | None = None,
) -> str:
    return await tools_for(store, settings)[name].handler(args or {}, context)


# -- creating one ------------------------------------------------------------


async def test_creating_one_reads_back_fire_times_a_person_can_check(store: Store) -> None:
    """The whole confirmation story. `0 9 * * 1-5` is not checkable and
    "Mon 07 Sep 09:00 Asia/Seoul" is, so the tool returns the second — in the
    task's own zone, because nine in the morning in Seoul is not nine on the
    server."""
    result = await call(store, "schedule_create", ASK)

    assert "Created schedule" in result
    assert result.count("Asia/Seoul") == 3
    assert "09:00" in result


async def test_the_task_it_creates_belongs_to_the_conversation_that_asked(store: Store) -> None:
    """§11.1. The model supplies what to do and when; the session supplies
    who, where and what may be seen."""
    await call(store, "schedule_create", ASK)

    (task,) = await Tasks(store).all()
    assert task.owner == "U01"
    assert task.session_id == IN_A_CHANNEL.session_id
    assert task.channel == "C0123"
    assert task.reply_to == "1756890000.123"
    assert task.scope == "channel:C0123"
    assert task.surface == "slack"


async def test_there_is_no_argument_for_where_a_task_posts(store: Store) -> None:
    """The guarantee is structural rather than checked: text Kasa read cannot
    ask for a task in another channel because the schema has nowhere to say
    it, and a call that tries is rejected before any handler runs."""
    create = tools_for(store)["schedule_create"]
    assert not {"channel", "session_id", "scope", "owner", "reply_to"} & set(
        create.input_schema["properties"]
    )
    assert create.input_schema["additionalProperties"] is False

    registry = ToolRegistry([create])
    result = await registry.dispatch(
        ToolUseBlock(id="t1", name="schedule_create", input={**ASK, "channel": "C0999"}),
        IN_A_CHANNEL,
    )

    assert result.is_error
    assert not await Tasks(store).all()


async def test_a_task_asked_for_in_a_dm_stays_in_the_dm(store: Store) -> None:
    dm = ToolContext(
        session_id="slack:T01:D0999:1756890000.123",
        scope="dm:U01",
        author="U01",
        channel="D0999",
    )

    await call(store, "schedule_create", ASK, context=dm)

    (task,) = await Tasks(store).all()
    assert task.channel == "D0999"
    assert task.scope == "dm:U01"


async def test_a_schedule_that_fires_too_often_comes_back_correctable(store: Store) -> None:
    """`ToolRegistry` hands a handler's failures back to the model as an error
    result, so a refusal is a turn that recovers rather than a turn that dies.
    The floor is named because the model is what has to correct the
    expression."""
    result = await call(store, "schedule_create", {**ASK, "cron": "* * * * *"})

    assert "not created" in result
    assert "the floor is 15" in result
    assert not await Tasks(store).all()


async def test_an_abbreviation_is_not_a_zone_and_the_message_says_which(store: Store) -> None:
    result = await call(store, "schedule_create", {**ASK, "timezone": "KST"})

    assert "not a time zone" in result
    assert "Asia/Seoul" in result, "the correction has to be in the message"


async def test_the_cap_is_reported_rather_than_silently_enforced(store: Store) -> None:
    settings = TaskSettings(max_per_owner=1)
    await call(store, "schedule_create", ASK, settings=settings)

    result = await call(store, "schedule_create", ASK, settings=settings)

    assert "which is the limit (1)" in result
    assert len(await Tasks(store).all()) == 1


async def test_no_timezone_means_utc_rather_than_a_guess(store: Store) -> None:
    await call(store, "schedule_create", {"prompt": "morning news", "cron": "0 9 * * *"})

    (task,) = await Tasks(store).all()
    assert task.timezone is None


@pytest.mark.parametrize("name", ["schedule_create", "schedule_list", "schedule_cancel"])
async def test_a_surface_with_no_identity_cannot_own_a_schedule(store: Store, name: str) -> None:
    """A task nothing can cap, cancel or notify. The terminal is the surface
    this happens on, and it has `kasa task add`."""
    anonymous = ToolContext(session_id="cli:1", scope="workspace")

    result = await call(store, name, {**ASK, "id": "01X"}, context=anonymous)

    assert "no user identity" in result
    assert not await Tasks(store).all()


# -- listing and cancelling --------------------------------------------------


async def test_listing_shows_the_schedule_and_when_it_next_runs(store: Store) -> None:
    await call(store, "schedule_create", ASK)

    result = await call(store, "schedule_list")

    assert "0 9 * * 1-5 (Asia/Seoul)" in result
    assert "active" in result
    assert "Asia/Seoul" in result.split("next:")[1]


async def test_listing_says_so_when_there_is_nothing(store: Store) -> None:
    assert "no standing tasks" in await call(store, "schedule_list")


async def test_a_schedule_that_stopped_reading_is_not_reported_as_fine(store: Store) -> None:
    """Answering "what have you got scheduled?" with a row and no next time
    reads as though it works, and the person goes on expecting their nine
    o'clock."""
    await call(store, "schedule_create", ASK)
    await store.write("UPDATE tasks SET timezone = 'Mars/Olympus'")

    result = await call(store, "schedule_list")

    assert "never" in result
    assert "not a time zone" in result


async def test_another_channel_s_schedules_are_neither_listed_nor_cancellable(
    store: Store,
) -> None:
    """§7.1. Text arriving in one channel must not be able to enumerate or
    delete another channel's tasks — and an id from elsewhere comes back as
    "no such schedule" rather than as a refusal that confirms it exists."""
    await call(store, "schedule_create", ASK)
    (theirs,) = await Tasks(store).all()
    elsewhere = ToolContext(
        session_id="slack:T01:C0456:1756890000.456",
        scope="channel:C0456",
        author="U02",
        channel="C0456",
    )

    listed = await call(store, "schedule_list", context=elsewhere)
    cancelled = await call(store, "schedule_cancel", {"id": theirs.id}, context=elsewhere)

    assert theirs.id not in listed
    assert "no schedule" in cancelled
    assert await Tasks(store).get(theirs.id) is not None


async def test_the_same_person_in_another_thread_cannot_reach_it_either(store: Store) -> None:
    """Narrowed by session as well as by owner. A task answers in one
    conversation, and that conversation is where it is managed from."""
    await call(store, "schedule_create", ASK)
    (mine,) = await Tasks(store).all()
    other_thread = ToolContext(
        session_id="slack:T01:C0123:1756899999.999",
        scope="channel:C0123",
        author="U01",
        channel="C0123",
    )

    assert mine.id not in await call(store, "schedule_list", context=other_thread)


async def test_cancelling_stops_it(store: Store) -> None:
    await call(store, "schedule_create", ASK)
    (task,) = await Tasks(store).all()

    result = await call(store, "schedule_cancel", {"id": task.id})

    assert "Cancelled" in result
    assert not await Tasks(store).all()


# -- and then it actually fires ----------------------------------------------


async def test_a_schedule_asked_for_in_a_thread_reaches_the_clock_and_the_queue(
    store: Store,
) -> None:
    """The thing the whole pair of issues is for, end to end: somebody asks in
    a thread, and at the fire time a turn is waiting in the inbox addressed
    back to that same thread.

    The clock is ticked with a `now` a year in the past so the occurrence it
    computes is already due; the alternative is a test that waits until nine.
    """
    await call(store, "schedule_create", ASK)
    specs = {spec.kind: spec for spec in default_specs(Config(), store)}
    scheduler = Scheduler(store, specs.values(), clocks=[Tasks(store)])

    await scheduler.schedule_due(now=datetime.now(UTC) - timedelta(days=365))
    (queued,) = await store.raw("SELECT * FROM jobs WHERE kind = ?", (TASK_KIND,))
    await specs[TASK_KIND].handler(
        Job(
            id=str(queued["id"]),
            kind=TASK_KIND,
            payload=json.loads(queued["payload"]),
            attempts=1,
        )
    )

    (row,) = await store.raw("SELECT payload FROM inbox")
    event = InboundEvent.from_json(row["payload"])
    assert event.text == ASK["prompt"]
    assert event.channel == "C0123"
    assert event.reply_to == "1756890000.123"
    assert event.scope == "channel:C0123"
    assert event.origin == "scheduled"

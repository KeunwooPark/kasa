"""A reply that rewrites itself, and the rate limit it must not walk into.

No `slack_bolt` and no `slack_sdk`: `stream.py` takes the two writes it needs
as a protocol, so the part worth testing — how often Slack is called, and what
happens when Slack says no — is reachable with a list and a clock.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from kasa.adapters.slack.stream import (
    FINAL_ATTEMPTS,
    MAX_REFUSALS,
    THINKING,
    LiveMessage,
    SlackRateLimited,
)
from kasa.llm.types import MessageStop, TextDelta, ToolUseStart, Usage

CHANNEL = "C0DEPLOY"
THREAD = "1700000000.000100"

#: Short enough that a handful of frames fit in a test, long enough that the
#: event loop is not the thing under measurement.
TICK = 0.02


class FakePoster:
    def __init__(self) -> None:
        self.posts: list[str] = []
        self.updates: list[str] = []
        #: How many of the next calls to refuse, and how.
        self.refuse_updates = 0
        self.fail_posts = 0
        #: A write that has reached Slack and is waiting on it. `entered` is
        #: set the moment one arrives; it lands when `gate` is set. Slack is
        #: the slow half of a `chat.update` and this is the only way to hold a
        #: test in the window that matters — the one where a frame is already
        #: over there and the turn ends.
        self.gate: asyncio.Event | None = None
        self.entered = asyncio.Event()

    async def post(self, *, channel: str, thread_ts: str | None, text: str) -> str:
        if self.fail_posts:
            self.fail_posts -= 1
            raise RuntimeError("slack said no")
        self.posts.append(text)
        return f"ts{len(self.posts)}"

    async def update(self, *, channel: str, ts: str, text: str) -> None:
        if self.refuse_updates:
            self.refuse_updates -= 1
            raise SlackRateLimited(0.0)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        self.updates.append(text)


def live(poster: FakePoster, **kwargs: object) -> LiveMessage:
    return LiveMessage(poster, channel=CHANNEL, thread_ts=THREAD, interval=TICK, **kwargs)  # type: ignore[arg-type]


def text(chunk: str) -> TextDelta:
    return TextDelta(text=chunk)


def stop(reason: str = "end_turn") -> MessageStop:
    return MessageStop(stop_reason=reason, usage=Usage(), model="test")  # type: ignore[arg-type]


async def ticks(count: int = 3) -> None:
    """Let the painter round the interval `count` times."""
    await asyncio.sleep(TICK * count + TICK / 2)


# -- the throttle -------------------------------------------------------------


async def test_a_delta_never_calls_slack() -> None:
    """The property the whole design rests on. A `chat.update` per token is
    hundreds a turn against a budget of about one a second — every one after
    the first refused, and a message that flickers on the way there."""
    poster = FakePoster()
    message = live(poster)
    await message.open()
    try:
        for n in range(500):
            await message.delta(text(f"{n} "))
    finally:
        await message.aclose()

    assert poster.updates == [], "not one of them reached Slack"
    assert poster.posts == [THINKING], "only the placeholder went up"


async def test_the_message_is_redrawn_on_the_interval_not_on_the_traffic() -> None:
    poster = FakePoster()
    message = live(poster)
    await message.open()
    try:
        for n in range(200):
            await message.delta(text(f"{n} "))
            if n % 50 == 0:
                await ticks(1)
        await ticks(1)
    finally:
        await message.aclose()

    assert 1 <= len(poster.updates) <= 8, f"{len(poster.updates)} updates for 200 deltas"


async def test_a_frame_that_says_what_the_last_one_said_is_not_sent() -> None:
    """A turn that is thinking for ten seconds should cost one write, not ten."""
    poster = FakePoster()
    message = live(poster)
    await message.open()
    try:
        await ticks(4)
    finally:
        await message.aclose()

    assert poster.updates == []


# -- what the message says ----------------------------------------------------


async def test_a_running_tool_shows_as_a_status_line() -> None:
    poster = FakePoster()
    message = live(poster)

    await message.delta(text("Let me look."))
    await message.delta(ToolUseStart(id="t1", name="recall_memory"))

    assert message.render() == "Let me look.\n\n_running `recall_memory`…_"


async def test_two_tools_in_one_round_are_one_line() -> None:
    poster = FakePoster()
    message = live(poster)

    await message.delta(ToolUseStart(id="t1", name="recall_memory"))
    await message.delta(ToolUseStart(id="t2", name="read_memory"))

    assert message.render() == "_running `recall_memory`, `read_memory`…_"


async def test_a_tool_stops_being_shown_once_the_model_answers_again() -> None:
    """There is no delta for a tool result coming back, so the next model call
    starting is the only signal that the tools are no longer running."""
    poster = FakePoster()
    message = live(poster)

    await message.delta(ToolUseStart(id="t1", name="recall_memory"))
    await message.delta(stop("tool_use"))
    await message.delta(text("It was Tuesday."))

    assert message.render() == "It was Tuesday."


async def test_a_message_with_nothing_in_it_yet_says_so() -> None:
    assert live(FakePoster()).render() == THINKING


# -- the final update ---------------------------------------------------------


async def test_the_answer_replaces_everything_that_came_before_it() -> None:
    poster = FakePoster()
    message = live(poster)
    await message.open()
    await message.delta(text("Let me look."))
    await message.delta(ToolUseStart(id="t1", name="recall_memory"))

    await message.finish("It was Tuesday.")

    assert poster.posts == [THINKING]
    assert poster.updates[-1] == "It was Tuesday."


async def test_an_answer_with_nothing_in_it_still_replaces_the_placeholder() -> None:
    """A message left reading "thinking…" forever is worse than one that says
    nothing came back."""
    poster = FakePoster()
    message = live(poster)
    await message.open()

    await message.finish("   ")

    assert poster.updates[-1] == "_the turn produced no answer._"


async def test_the_final_update_waits_out_a_refusal() -> None:
    """A frame is a nicety and can be dropped. The answer cannot: the only
    other way to deliver it is to run the whole turn again."""
    poster = FakePoster()
    message = live(poster)
    await message.open()
    poster.refuse_updates = FINAL_ATTEMPTS - 1

    await message.finish("It was Tuesday.")

    assert poster.updates == ["It was Tuesday."]


async def test_a_final_update_that_never_lands_fails_the_turn() -> None:
    """Which puts the event back on the queue. Reporting an answer that never
    reached the thread is the one outcome worse than answering twice."""
    poster = FakePoster()
    message = live(poster)
    await message.open()
    poster.refuse_updates = FINAL_ATTEMPTS + 1

    with pytest.raises(SlackRateLimited):
        await message.finish("It was Tuesday.")


async def test_a_frame_already_in_flight_cannot_land_after_the_answer() -> None:
    """#192: a reply left showing a mid-sentence prefix of an answer that was
    delivered in full.

    Cancelling the painter mid-`chat.update` abandons only this side of a write
    Slack has already taken. Slack still applies it, and it can be applied
    after the answer — which no fake poster can reproduce, because the reorder
    happens at Slack. What is checkable here is the invariant that rules it
    out: the frame is waited for rather than cancelled, and the answer is
    written after it.
    """
    poster = FakePoster()
    message = live(poster)
    await message.open()
    poster.gate = asyncio.Event()
    await message.delta(text("half a sen"))
    await asyncio.wait_for(poster.entered.wait(), timeout=1.0)

    finishing = asyncio.create_task(message.finish("It was Tuesday."))
    await ticks(2)
    assert poster.updates == [], "the answer waited for the frame at Slack"
    poster.gate.set()
    await finishing

    assert poster.updates == ["half a sen", "It was Tuesday."], "in that order, both of them"


async def test_a_frame_waiting_its_turn_is_dropped_rather_than_sent_late() -> None:
    """The other half of the same window. A frame that had not reached Slack
    when the turn ended has nothing to say that the answer does not."""
    poster = FakePoster()
    message = live(poster)
    await message.open()
    poster.gate = asyncio.Event()
    await message.delta(text("half a sen"))
    await asyncio.wait_for(poster.entered.wait(), timeout=1.0)
    # A second tick queues behind the first, which is still at Slack.
    await message.delta(text("tence"))
    await ticks(2)

    finishing = asyncio.create_task(message.finish("It was Tuesday."))
    await ticks(1)
    poster.gate.set()
    await finishing

    assert poster.updates[-1] == "It was Tuesday."
    assert "half a sentence" not in poster.updates, "the queued frame never went"


async def test_finishing_does_not_wait_out_an_interval_to_stop_the_painter() -> None:
    """Waiting for an in-flight frame must not become waiting for the next one.
    With the painter asleep there is nothing to wait for, and the answer goes
    straight out."""
    poster = FakePoster()
    message = LiveMessage(poster, channel=CHANNEL, thread_ts=THREAD, interval=5.0)  # type: ignore[arg-type]
    await message.open()
    await message.delta(text("It was Tuesday."))

    started = time.monotonic()
    await message.finish("It was Tuesday.")

    assert time.monotonic() - started < 1.0
    assert poster.updates == ["It was Tuesday."]


# -- degrading ----------------------------------------------------------------


async def test_rate_limit_pressure_gives_up_on_frames_and_keeps_the_answer() -> None:
    """§10.1's "graceful degradation to a single message". The painter stops
    for the rest of the turn rather than spending it in backoff."""
    poster = FakePoster()
    message = live(poster)
    await message.open()
    poster.refuse_updates = 1_000

    try:
        for n in range(50):
            await message.delta(text(f"{n} "))
            await ticks(1)
    finally:
        await message.aclose()
    attempted = 1_000 - poster.refuse_updates

    assert attempted == MAX_REFUSALS, "it stopped trying rather than trying harder"


async def test_a_placeholder_that_could_not_be_posted_becomes_one_message() -> None:
    poster = FakePoster()
    poster.fail_posts = 1
    message = live(poster)

    await message.open()
    await message.delta(text("It was Tuesday."))
    await message.finish("It was Tuesday.")

    assert not message.live
    assert poster.posts == ["It was Tuesday."], "the answer, once, with no placeholder"
    assert poster.updates == []


async def test_a_second_attempt_rewrites_the_first_attempt_s_message() -> None:
    """A turn that failed is redelivered. Posting a fresh placeholder each time
    leaves a thread of "thinking…" messages, one per attempt."""
    poster = FakePoster()
    first = live(poster)
    await first.open()
    await first.aclose()

    second = live(poster, ts=first.ts)
    await second.open()
    await second.finish("It was Tuesday.")

    assert poster.posts == [THINKING], "no second placeholder"
    assert poster.updates[-1] == "It was Tuesday."


async def test_closing_stops_the_redrawing() -> None:
    poster = FakePoster()
    message = live(poster)
    await message.open()
    await message.delta(text("half a sentence"))

    await message.aclose()
    before = len(poster.updates)
    await message.delta(text(" and the rest"))
    await ticks(3)

    assert len(poster.updates) == before, "the painter is gone, not merely quiet"

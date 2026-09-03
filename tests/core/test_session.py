"""One actor per thread: serialized within a session, concurrent across them."""

from __future__ import annotations

import asyncio

import pytest

from kasa.core.events import InboundEvent
from kasa.core.inbox import Dispatcher, Inbox
from kasa.core.session import SessionActor, SessionOverflow, SessionRouter, Turn
from kasa.errors import StoreError
from kasa.llm.types import Message
from kasa.store import Store
from tests.conftest import until


def event(external_id: str, *, session: str = "slack:T:C:1", text: str = "hi") -> InboundEvent:
    return InboundEvent(
        source="slack", external_id=external_id, session_id=session, text=text, scope="workspace"
    )


async def open_episode(store: Store, session_id: str, episode_id: str = "ep_1") -> None:
    await store.ensure_session(session_id, surface="slack")
    await store.write(
        "INSERT INTO episodes (id, session_id, started_at, state)"
        " VALUES (?, ?, '2026-01-01T00:00:00.000+00:00', 'open')",
        (episode_id, session_id),
    )


# -- ordering ----------------------------------------------------------------


async def test_two_messages_in_one_thread_are_answered_in_order(store: Store) -> None:
    """The acceptance criterion of #20. The first turn is slow on purpose: a
    router that hands both to the agent at once answers them out of order."""
    answered: list[str] = []

    async def handler(turn: Turn) -> None:
        if turn.event.text == "first":
            await asyncio.sleep(0.05)
        answered.append(turn.event.text)

    router = SessionRouter(store, handler)
    try:
        await asyncio.gather(
            router.deliver(event("E1", text="first")),
            router.deliver(event("E2", text="second")),
        )
    finally:
        await router.aclose()

    assert answered == ["first", "second"]


async def test_the_second_message_sees_the_first(store: Store) -> None:
    """Serialized is not enough on its own — the second turn has to be looking
    at a session the first one has already written to."""
    seen: list[int] = []

    async def handler(turn: Turn) -> None:
        seen.append(turn.session.message_count)
        await store.append_message(turn.event.session_id, Message.user(turn.event.text))
        await store.append_message(turn.event.session_id, Message.assistant("noted"))

    router = SessionRouter(store, handler)
    try:
        await asyncio.gather(
            router.deliver(event("E1", text="first")),
            router.deliver(event("E2", text="second")),
        )
    finally:
        await router.aclose()

    assert seen == [0, 2]


async def test_a_hundred_sessions_do_not_serialize_against_each_other(store: Store) -> None:
    """The other acceptance criterion. The barrier only opens once every turn
    has started, so a router that serialized them would deadlock here."""
    barrier = asyncio.Barrier(100)

    async def handler(turn: Turn) -> None:
        await barrier.wait()

    router = SessionRouter(store, handler)
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(router.deliver(event(f"E{n}", session=f"slack:T:C:{n}")) for n in range(100))
            ),
            timeout=10.0,
        )
        assert router.sessions == 100
    finally:
        await router.aclose()


# -- rehydration -------------------------------------------------------------


async def test_an_actor_reads_the_session_and_its_open_episode(store: Store) -> None:
    await open_episode(store, "slack:T:C:1")
    await store.append_message("slack:T:C:1", Message.user("said earlier"))
    seen: list[Turn] = []

    async def handler(turn: Turn) -> None:
        seen.append(turn)

    router = SessionRouter(store, handler)
    try:
        await router.deliver(event("E1"))
    finally:
        await router.aclose()

    assert seen[0].session.episode_id == "ep_1"
    assert seen[0].session.message_count == 1
    assert seen[0].session.surface == "slack"


async def test_a_session_with_nothing_open_reports_no_episode(store: Store) -> None:
    """`episode_close` (#27) is what opens them; until then `None` is normal."""
    seen: list[Turn] = []

    async def handler(turn: Turn) -> None:
        seen.append(turn)

    router = SessionRouter(store, handler)
    try:
        await router.deliver(event("E1"))
    finally:
        await router.aclose()

    assert seen[0].session.episode_id is None
    assert await store.get_session("slack:T:C:1") is not None


async def test_an_evicted_session_picks_up_where_it_left_off(store: Store) -> None:
    """Eviction is free because the actor was holding nothing worth keeping."""
    seen: list[int] = []

    async def handler(turn: Turn) -> None:
        seen.append(turn.session.message_count)
        await store.append_message(turn.event.session_id, Message.user(turn.event.text))

    router = SessionRouter(store, handler, idle_after=0.0)
    try:
        await router.deliver(event("E1"))
        assert await router.evict_idle() == 1
        assert router.sessions == 0
        await router.deliver(event("E2"))
    finally:
        await router.aclose()

    assert seen == [0, 1]


# -- eviction ----------------------------------------------------------------


async def test_a_busy_actor_is_not_evicted(store: Store) -> None:
    gate = asyncio.Event()
    started = asyncio.Event()

    async def handler(turn: Turn) -> None:
        started.set()
        await gate.wait()

    router = SessionRouter(store, handler, idle_after=0.0)
    delivering = asyncio.create_task(router.deliver(event("E1")))
    try:
        await started.wait()
        assert await router.evict_idle() == 0
        assert router.sessions == 1
    finally:
        gate.set()
        await delivering
        await router.aclose()


async def test_the_sweeper_evicts_without_being_asked(store: Store) -> None:
    async def handler(turn: Turn) -> None:
        return None

    router = SessionRouter(store, handler, idle_after=0.0, sweep_interval=0.01)
    try:
        await router.deliver(event("E1"))
        await until(lambda: router.sessions == 0)
    finally:
        await router.aclose()


# -- failure -----------------------------------------------------------------


async def test_a_failed_turn_reaches_whoever_delivered_it(store: Store) -> None:
    """The dispatcher decides what a failure means. The actor just reports it."""

    async def handler(turn: Turn) -> None:
        raise RuntimeError("the model is down")

    router = SessionRouter(store, handler)
    try:
        with pytest.raises(RuntimeError, match="the model is down"):
            await router.deliver(event("E1"))
    finally:
        await router.aclose()


async def test_a_failed_turn_does_not_take_the_session_with_it(store: Store) -> None:
    handled: list[str] = []

    async def handler(turn: Turn) -> None:
        if turn.event.text == "bad":
            raise RuntimeError("no")
        handled.append(turn.event.text)

    router = SessionRouter(store, handler)
    try:
        with pytest.raises(RuntimeError):
            await router.deliver(event("E1", text="bad"))
        await router.deliver(event("E2", text="good"))
    finally:
        await router.aclose()

    assert handled == ["good"]


async def test_a_session_that_falls_too_far_behind_refuses_more(store: Store) -> None:
    """The dispatcher bounds what is in flight, so this is the guard against
    everything else — one busy channel must not grow without limit."""
    handled: list[str] = []

    async def handler(turn: Turn) -> None:
        handled.append(turn.event.text)

    actor = SessionActor("slack:T:C:1", store=store, handler=handler, mailbox_limit=2)
    # No awaits between these, so the actor's task cannot drain the mailbox
    # underneath the assertion.
    waiting = [actor.post(event("E1")), actor.post(event("E2"))]
    with pytest.raises(SessionOverflow, match="2 message"):
        actor.post(event("E3"))

    await asyncio.gather(*waiting)
    await actor.aclose()
    assert handled == ["hi", "hi"]


async def test_a_message_whose_caller_gave_up_is_not_answered(store: Store) -> None:
    """A dispatcher cancelled at shutdown has already handed its inbox row
    back. Answering anyway is a duplicate this layer can cheaply avoid."""
    gate = asyncio.Event()
    handled: list[str] = []

    async def handler(turn: Turn) -> None:
        handled.append(turn.event.text)
        await gate.wait()

    actor = SessionActor("slack:T:C:1", store=store, handler=handler)
    first = actor.post(event("E1", text="running"))
    second = actor.post(event("E2", text="abandoned"))
    await asyncio.sleep(0)  # let the actor pick the first one up
    second.cancel()
    gate.set()
    await first
    await actor.aclose()

    assert handled == ["running"]


async def test_closing_the_router_answers_what_is_already_queued(store: Store) -> None:
    handled: list[str] = []

    async def handler(turn: Turn) -> None:
        handled.append(turn.event.text)

    router = SessionRouter(store, handler)
    delivering = [
        asyncio.create_task(router.deliver(event("E1", text="one"))),
        asyncio.create_task(router.deliver(event("E2", text="two"))),
    ]
    await asyncio.sleep(0)
    await router.aclose()
    await asyncio.gather(*delivering)

    assert handled == ["one", "two"]
    with pytest.raises(SessionOverflow, match="closed"):
        await router.deliver(event("E3"))


# -- through the queue -------------------------------------------------------


async def test_the_queue_feeds_the_router_in_order(store: Store) -> None:
    """The real path: inbox -> dispatcher -> actor. The dispatcher runs events
    concurrently, so ordering within a thread is the router's doing."""
    answered: list[str] = []

    async def handler(turn: Turn) -> None:
        if turn.event.text == "first":
            await asyncio.sleep(0.05)
        answered.append(turn.event.text)
        await store.append_message(turn.event.session_id, Message.user(turn.event.text))

    inbox = Inbox(store)
    router = SessionRouter(store, handler)
    dispatcher = Dispatcher(inbox, router.deliver, poll_interval=0.01)
    await inbox.enqueue(event("E1", text="first"))
    await inbox.enqueue(event("E2", text="second"))

    running = asyncio.create_task(dispatcher.run())
    try:
        await until(lambda: len(answered) == 2)
    finally:
        dispatcher.stop()
        await asyncio.wait_for(running, timeout=5.0)
        await router.aclose()

    assert answered == ["first", "second"]
    assert await inbox.counts() == {"done": 2}


async def test_a_turn_that_raises_leaves_its_inbox_row_for_a_retry(store: Store) -> None:
    """The two layers have to agree: the actor reports the failure, and the
    queue is what decides it gets another go."""
    attempts = 0

    async def handler(turn: Turn) -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("the model is down")

    inbox = Inbox(store)
    router = SessionRouter(store, handler)
    dispatcher = Dispatcher(inbox, router.deliver, poll_interval=0.01)
    await inbox.enqueue(event("E1"))

    running = asyncio.create_task(dispatcher.run())
    try:
        await until(lambda: attempts == 1)
    finally:
        dispatcher.stop()
        await asyncio.wait_for(running, timeout=5.0)
        await router.aclose()

    assert await inbox.counts() == {"pending": 1}


async def test_a_sweep_that_fails_does_not_end_the_sweeper(store: Store) -> None:
    """`evict_idle` closes actors, which touches the store. The sweeper used to
    stop on the first failure, and `_actors` then grows without bound for the
    life of the process."""
    sweeps = 0

    async def handler(turn: Turn) -> None:
        pass

    router = SessionRouter(store, handler, sweep_interval=0.01)

    async def failing_sweep() -> int:
        nonlocal sweeps
        sweeps += 1
        raise StoreError("database is locked")

    router.evict_idle = failing_sweep  # type: ignore[method-assign]
    await router.deliver(event("E1"))  # the first delivery starts the sweeper
    await until(lambda: sweeps >= 5)
    await router.aclose()

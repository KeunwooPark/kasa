"""The durable ingress queue: dedupe, leases, retries, and replay."""

from __future__ import annotations

import asyncio
import signal
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from kasa.core.backoff import Backoff
from kasa.core.events import InboundEvent
from kasa.core.inbox import Dispatcher, Inbox, LeasedEvent
from kasa.errors import StoreError
from kasa.store import Store
from tests.conftest import until


def event(external_id: str, *, text: str = "hello", session: str = "cli:s1") -> InboundEvent:
    return InboundEvent(source="cli", external_id=external_id, session_id=session, text=text)


async def running(dispatcher: Dispatcher) -> asyncio.Task[None]:
    return asyncio.create_task(dispatcher.run())


async def stop(dispatcher: Dispatcher, task: asyncio.Task[None]) -> None:
    dispatcher.stop()
    await asyncio.wait_for(task, timeout=5.0)


# -- dedupe ------------------------------------------------------------------


async def test_the_same_provider_event_is_queued_once(store: Store) -> None:
    """Slack re-sends event ids; a re-sent one must not earn a second answer."""
    inbox = Inbox(store)
    first = await inbox.enqueue(event("Ev123"))
    second = await inbox.enqueue(event("Ev123", text="the retry carries the same id"))

    assert (first.duplicate, second.duplicate) == (False, True)
    assert second.id == first.id
    assert await inbox.counts() == {"pending": 1}


async def test_the_same_id_from_a_different_source_is_a_different_event(store: Store) -> None:
    inbox = Inbox(store)
    await inbox.enqueue(event("E1"))
    await inbox.enqueue(InboundEvent(source="slack", external_id="E1", session_id="slack:x"))
    assert await inbox.counts() == {"pending": 2}


# -- leases ------------------------------------------------------------------


async def test_a_leased_event_is_not_offered_to_a_second_drainer(store: Store) -> None:
    inbox = Inbox(store, lease_ttl=3600)
    await inbox.enqueue(event("E1"))

    assert [item.event.external_id for item in await inbox.lease()] == ["E1"]
    assert await inbox.lease() == []


async def test_a_lease_that_expires_is_delivered_again(store: Store) -> None:
    """A crash is indistinguishable from a very slow holder, so both replay."""
    inbox = Inbox(store, lease_ttl=0.0)
    await inbox.enqueue(event("E1"))

    first = await inbox.lease()
    second = await inbox.lease()

    assert [item.id for item in first] == [item.id for item in second]
    assert (first[0].attempts, second[0].attempts) == (1, 2)


async def test_a_completed_event_is_never_delivered_again(store: Store) -> None:
    inbox = Inbox(store, lease_ttl=0.0)
    await inbox.enqueue(event("E1"))

    leased = await inbox.lease()
    await inbox.complete(leased[0].id)

    assert await inbox.lease() == []
    assert await inbox.counts() == {"done": 1}


async def test_events_are_delivered_oldest_first(store: Store) -> None:
    inbox = Inbox(store)
    for n in range(5):
        await inbox.enqueue(event(f"E{n}"))

    leased = await inbox.lease(limit=5)
    assert [item.event.external_id for item in leased] == ["E0", "E1", "E2", "E3", "E4"]


# -- failure -----------------------------------------------------------------


async def test_a_failed_delivery_is_not_retried_immediately(store: Store) -> None:
    """The backoff is the point: a provider blip retried in a tight loop is a
    second outage on top of the first."""
    inbox = Inbox(store)
    await inbox.enqueue(event("E1"))
    leased = await inbox.lease()

    assert await inbox.fail(leased[0], "provider said no") is True
    assert await inbox.lease() == []
    assert await inbox.counts() == {"pending": 1}


async def test_a_message_that_keeps_failing_is_dead_lettered(store: Store) -> None:
    inbox = Inbox(store, backoff=Backoff(max_attempts=3, base=0.0, cap=0.0))
    await inbox.enqueue(event("E1"))

    outcomes = []
    for _ in range(3):
        leased = await inbox.lease()
        outcomes.append(await inbox.fail(leased[0], "still no"))

    assert outcomes == [True, True, False]
    assert await inbox.lease() == []
    assert await inbox.counts() == {"failed": 1}
    assert (await inbox.dead_letters())[0]["last_error"] == "still no"


async def test_a_dead_letter_can_be_put_back(store: Store) -> None:
    """Dead-lettering is a pause for a human, not a delete."""
    inbox = Inbox(store, backoff=Backoff(max_attempts=1, base=0.0, cap=0.0))
    await inbox.enqueue(event("E1"))
    await inbox.fail((await inbox.lease())[0], "nope")

    assert await inbox.revive() == 1
    assert [item.attempts for item in await inbox.lease()] == [1]


async def test_a_payload_this_build_cannot_read_is_dead_lettered_at_once(store: Store) -> None:
    """Retrying will not make it parse, so it does not get five attempts."""
    inbox = Inbox(store)
    await store.enqueue_inbox(source="slack", external_id="E1", payload='{"source": "slack"}')

    assert await inbox.lease() == []
    assert await inbox.counts() == {"failed": 1}


# -- the dispatcher ----------------------------------------------------------


async def test_the_dispatcher_delivers_and_completes(store: Store) -> None:
    inbox = Inbox(store)
    seen: list[str] = []

    async def handler(inbound: InboundEvent) -> None:
        seen.append(inbound.text)

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01)
    await inbox.enqueue(event("E1", text="one"))
    await inbox.enqueue(event("E2", text="two"))
    task = await running(dispatcher)
    await until(lambda: len(seen) == 2)
    await stop(dispatcher, task)

    assert seen == ["one", "two"]
    assert await inbox.counts() == {"done": 2}


async def test_a_handler_that_raises_leaves_the_event_queued(store: Store) -> None:
    inbox = Inbox(store)
    calls = 0

    async def handler(inbound: InboundEvent) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("the model is down")

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01)
    await inbox.enqueue(event("E1"))
    task = await running(dispatcher)
    await until(lambda: calls == 1)
    await stop(dispatcher, task)

    assert await inbox.counts() == {"pending": 1}
    assert "the model is down" in (await store.raw("SELECT last_error FROM inbox"))[0]["last_error"]


async def test_no_more_events_run_at_once_than_the_limit_allows(store: Store) -> None:
    """Concurrency is bounded here so a burst cannot open 500 model calls."""
    inbox = Inbox(store)
    gate = asyncio.Event()
    started = 0

    async def handler(inbound: InboundEvent) -> None:
        nonlocal started
        started += 1
        await gate.wait()

    for n in range(5):
        await inbox.enqueue(event(f"E{n}"))
    dispatcher = Dispatcher(inbox, handler, concurrency=2, poll_interval=0.01)
    task = await running(dispatcher)

    await until(lambda: started == 2)
    await asyncio.sleep(0.05)
    assert (started, dispatcher.in_flight) == (2, 2)

    gate.set()
    await until(lambda: started == 5)
    await stop(dispatcher, task)
    assert await inbox.counts() == {"done": 5}


async def test_a_long_delivery_keeps_its_lease(store: Store) -> None:
    """A turn may outlast the lease. Renewal is what stops the queue racing the
    agent to answer the same message twice."""
    inbox = Inbox(store, lease_ttl=3.0)
    gate = asyncio.Event()
    calls = 0

    async def handler(inbound: InboundEvent) -> None:
        nonlocal calls
        calls += 1
        await gate.wait()

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01)
    await inbox.enqueue(event("E1"))
    task = await running(dispatcher)
    await until(lambda: calls == 1)
    await asyncio.sleep(2.0)  # past the TTL, inside the renewal interval

    assert calls == 1
    gate.set()
    await stop(dispatcher, task)
    assert await inbox.counts() == {"done": 1}


async def test_an_expired_lease_is_not_handed_back_to_the_drainer_still_running_it(
    store: Store,
) -> None:
    """`lease_until <= now` on a leased row is how a dead process's work replays,
    and it cannot tell that process from this one. Answering twice is the exact
    duplicate `UNIQUE (source, external_id)` exists to prevent."""
    inbox = Inbox(store, lease_ttl=0.0)  # every lease is expired the moment it is taken
    gate = asyncio.Event()
    calls = 0

    async def handler(inbound: InboundEvent) -> None:
        nonlocal calls
        calls += 1
        await gate.wait()

    dispatcher = Dispatcher(inbox, handler, concurrency=4, poll_interval=0.01)
    await inbox.enqueue(event("E1"))
    task = await running(dispatcher)
    await until(lambda: calls >= 1)
    await asyncio.sleep(0.1)  # ten polls, each of which re-leases the running row

    assert calls == 1, f"one message delivered {calls} time(s)"
    gate.set()
    await stop(dispatcher, task)
    assert await inbox.counts() == {"done": 1}


async def test_a_batch_of_only_our_own_rows_polls_rather_than_spins(store: Store) -> None:
    """A non-zero `_start_batch` tells `run` there was work and to look again at
    once. Counting rows it skipped is a busy loop rather than a poll."""
    inbox = Inbox(store, lease_ttl=0.0)
    gate = asyncio.Event()
    leases = 0
    taken = inbox.lease

    async def counted(*, limit: int = 1) -> list[LeasedEvent]:
        nonlocal leases
        leases += 1
        return await taken(limit=limit)

    inbox.lease = counted  # type: ignore[method-assign]

    async def handler(inbound: InboundEvent) -> None:
        await gate.wait()

    dispatcher = Dispatcher(inbox, handler, concurrency=4, poll_interval=0.02)
    await inbox.enqueue(event("E1"))
    task = await running(dispatcher)
    await until(lambda: dispatcher.in_flight == 1)
    await asyncio.sleep(0.2)  # ten polls' worth

    assert leases < 40, f"{leases} leases in 0.2s is a spin, not a poll"
    gate.set()
    await stop(dispatcher, task)


async def test_a_renewal_that_fails_does_not_end_the_keepalive(store: Store) -> None:
    """`renew` reaches the store, so a locked database is enough to raise. The
    keepalive used to stop there — no more renewals, no more purging, and no
    warning, because the shutdown `gather` retrieves the exception."""
    inbox = Inbox(store, lease_ttl=3.0)  # renewed every second
    gate = asyncio.Event()
    renewals = 0

    async def failing_renew(ids: Sequence[int]) -> None:
        nonlocal renewals
        renewals += 1
        raise StoreError("database is locked")

    inbox.renew = failing_renew  # type: ignore[method-assign]

    async def handler(inbound: InboundEvent) -> None:
        await gate.wait()

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01)
    await inbox.enqueue(event("E1"))
    task = await running(dispatcher)
    await until(lambda: renewals >= 2, within=8.0)

    gate.set()
    await stop(dispatcher, task)


async def test_shutdown_hands_back_unfinished_work_without_burning_an_attempt(
    store: Store,
) -> None:
    """Stopping the daemon is not the message's fault. A queue that dead-letters
    its backlog after five deploys is worse than no bound at all."""
    inbox = Inbox(store)
    started = asyncio.Event()

    async def handler(inbound: InboundEvent) -> None:
        started.set()
        await asyncio.sleep(3600)

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01, shutdown_grace=0.0)
    await inbox.enqueue(event("E1"))
    task = await running(dispatcher)
    await started.wait()
    await stop(dispatcher, task)

    assert await inbox.counts() == {"pending": 1}
    assert (await store.raw("SELECT attempts FROM inbox"))[0]["attempts"] == 0


async def test_a_restart_replays_what_the_previous_run_was_holding(store: Store) -> None:
    """The row is still leased with hours to run; a sole drainer takes it back
    rather than leaving a person waiting out a lease nobody holds."""
    inbox = Inbox(store, lease_ttl=3600)
    await inbox.enqueue(event("E1", text="the one that was in flight"))
    await inbox.lease()  # the run that died

    seen: list[str] = []

    async def handler(inbound: InboundEvent) -> None:
        seen.append(inbound.text)

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01)
    task = await running(dispatcher)
    await until(lambda: seen == ["the one that was in flight"])
    await stop(dispatcher, task)


async def test_a_second_drainer_waits_out_a_lease_it_did_not_take(store: Store) -> None:
    """`reclaim_on_start` is the sole-drainer assumption, and it is switchable."""
    inbox = Inbox(store, lease_ttl=3600)
    await inbox.enqueue(event("E1"))
    await inbox.lease()

    async def handler(inbound: InboundEvent) -> None:  # pragma: no cover - must not run
        raise AssertionError("stole a live lease")

    dispatcher = Dispatcher(inbox, handler, poll_interval=0.01, reclaim_on_start=False)
    task = await running(dispatcher)
    await asyncio.sleep(0.05)
    await stop(dispatcher, task)

    assert await inbox.counts() == {"leased": 1}


# -- retention ---------------------------------------------------------------


async def test_delivered_rows_outlive_the_provider_retry_window(store: Store) -> None:
    """A delivered row is still the dedupe record. Purging it is how a late
    Slack retry earns a second answer."""
    inbox = Inbox(store)
    enqueued = await inbox.enqueue(event("E1"))
    await inbox.complete(enqueued.id)

    assert await inbox.purge() == 0
    assert (await inbox.enqueue(event("E1"))).duplicate is True


async def test_a_purge_takes_only_old_delivered_rows(store: Store) -> None:
    inbox = Inbox(store)
    for name in ("done", "pending"):
        enqueued = await inbox.enqueue(event(name))
        if name == "done":
            await inbox.complete(enqueued.id)
    await store.write("UPDATE inbox SET received_at = '2020-01-01T00:00:00.000+00:00'")

    assert await inbox.purge() == 1
    assert await inbox.counts() == {"pending": 1}


# -- the acceptance criterion ------------------------------------------------


CRASH_MID_DELIVERY = """
import asyncio, os, signal, sys

from kasa.core.inbox import Inbox
from kasa.store import Store
from tests.conftest import until


async def main() -> None:
    db, marker = sys.argv[1], sys.argv[2]
    store = await Store.open(db)
    leased = await Inbox(store, lease_ttl=3600).lease()
    assert len(leased) == 1, leased
    # Proof that the row really was leased — that the kill below lands
    # mid-delivery and not before it.
    open(marker, "w").write(leased[0].event.text)
    os.kill(os.getpid(), signal.SIGKILL)


asyncio.run(main())
"""


async def test_a_process_killed_mid_delivery_is_answered_exactly_once(tmp_path: Path) -> None:
    """The acceptance criterion of #19, run for real: `kill -9` mid-turn,
    restart, and the message is answered once — not lost, not twice."""
    db = tmp_path / "kasa.db"
    async with await Store.open(db) as setup:
        await Inbox(setup).enqueue(event("Ev123", text="what did we decide?"))

    script = tmp_path / "crash.py"
    script.write_text(textwrap.dedent(CRASH_MID_DELIVERY))
    marker = tmp_path / "leased"
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(script), str(db), str(marker)
    )
    assert await process.wait() == -signal.SIGKILL
    assert marker.read_text() == "what did we decide?"

    answers: list[str] = []

    async def handler(inbound: InboundEvent) -> None:
        answers.append(inbound.text)

    async with await Store.open(db) as store:
        inbox = Inbox(store, lease_ttl=3600)
        dispatcher = Dispatcher(inbox, handler, poll_interval=0.01)
        task = await running(dispatcher)
        await until(lambda: len(answers) == 1)
        await asyncio.sleep(0.1)  # a second delivery would land in this window
        await stop(dispatcher, task)

        assert answers == ["what did we decide?"]
        assert await inbox.counts() == {"done": 1}

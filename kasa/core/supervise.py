"""Background loops that survive what they run.

Three long-lived helpers keep the system moving without anybody awaiting them:
the drainer's keepalive holds leases and retires finished rows, the scheduler's
clock queues the next occurrence of every recurring job, and the session
sweeper drops idle actors. Each is started with `create_task`, cancelled at
shutdown, and never otherwise looked at.

That combination is what makes an exception in one of them silent. The first
one ends the task for good, and because the shutdown path is `cancel()` then
`gather(..., return_exceptions=True)`, the exception *is* retrieved — so
asyncio never prints its "task exception was never retrieved" warning either.
The process keeps running and looks healthy while the clock has stopped.

The right answer is the same for all three, which is why it is written once: a
tick has no partial state to unwind and the next one is seconds away, so say
what happened, loudly, and go round again.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)


async def keep_running(
    body: Callable[[], Awaitable[object]],
    *,
    every: float,
    name: str,
    start_now: bool = False,
) -> None:
    """Run `body` every `every` seconds until cancelled, whatever it raises.

    `name` is what a failure is reported as, so it should read as the loop's
    job rather than as a method name — "scheduler clock", not "_tick_forever".

    Set `start_now` for a loop that must do its work before the first interval
    elapses. The default waits, because a keepalive with nothing in flight and
    a sweeper with nothing to sweep both have nothing to do at startup.

    Cancellation still stops it immediately: `CancelledError` is a
    `BaseException`, not an `Exception`, so it passes straight through.
    """
    if not start_now:
        await asyncio.sleep(every)
    while True:
        try:
            await body()
        except Exception:
            log.exception("%s failed; it will try again in %.0fs", name, every)
        await asyncio.sleep(every)

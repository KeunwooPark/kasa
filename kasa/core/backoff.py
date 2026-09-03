"""How long a durable queue waits before trying again, and when it stops.

Not the same problem as retrying one HTTP call — `llm.registry.RetryPolicy`
does that, with jitter, sub-second delays and the provider's own `Retry-After`.
Here the delay is written into a row and read back by whichever process picks
the work up next, possibly after a restart, so it is deterministic and coarse.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Backoff:
    """Doubling delays, a ceiling, and a number of tries."""

    max_attempts: int
    base: float
    cap: float

    def delay_after(self, attempts: int) -> float | None:
        """Seconds to wait before the next try, or None to give up.

        `attempts` counts tries already made, this one included, which is what
        both queues store — they increment it when they hand the work out, so
        that work which kills the process still counts as having been tried.
        """
        if attempts >= self.max_attempts:
            return None
        return float(min(self.cap, self.base * 2 ** (attempts - 1)))

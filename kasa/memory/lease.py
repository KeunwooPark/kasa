"""The single-writer lease over the long-term memory repo.

Two halves, and both are load-bearing.

The **flock** is the enforcement. It is held by the kernel against an open file
description, so it is released the instant the holder dies — including when the
holder is killed mid-write, which is exactly the case a lock built out of a
database row gets wrong.

The **database row** is the explanation. A flock tells you the lock is taken; it
does not tell you by whom, since when, or for what job, and that is what someone
staring at a stuck daemon actually needs.

Consequently: taking the flock is the decision, the row is written after, and a
row without a live flock is a leftover from a crash rather than a live holder.
"""

from __future__ import annotations

import fcntl
import logging
import os
import socket
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from kasa.errors import KasaError
from kasa.store import Store

log = logging.getLogger(__name__)

LEASE_NAME = "ltm"
LOCK_FILENAME = "kasa-writer.lock"

#: The index is a second resource with a second writer. It gets its own lease
#: rather than sharing the repo's: making a reindex wait on an in-flight memory
#: write, and vice versa, would be a bigger change than the problem warrants —
#: a reindex reading the tree mid-write is a benign race the next reindex
#: fixes, and a half-applied index is not (#95).
INDEX_LEASE_NAME = "index"
#: Appended to the database's own filename. The index lives in the database,
#: so its lock belongs beside it rather than in the repo.
INDEX_LOCK_SUFFIX = ".index.lock"

DEFAULT_TTL = 900.0

#: What each lease is called in the message somebody reads when it is taken.
_DESCRIPTIONS = {LEASE_NAME: "memory write lease", INDEX_LEASE_NAME: "index rebuild lease"}


class LeaseError(KasaError):
    """Someone else is writing to the memory repo."""


class LockingUnavailable(KasaError):
    """The filesystem cannot lock, so nothing here is enforcing anything.

    Deliberately *not* a `LeaseError`. Callers are entitled to treat a
    `LeaseError` as "somebody else is doing this work and will finish it" and
    carry on — `runner.jobs._reindex` does exactly that (#116). That inference
    is sound for contention and false here: nobody holds the lock, nobody is
    doing the work, and the single-writer guarantee the repo is written under
    does not exist. It has to be a different exception so that catching the one
    cannot swallow the other.
    """


def holder_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


# Synchronous on purpose: these are local stat and flock calls measured in
# microseconds. Wrapping them in async machinery would buy nothing and hide
# that the decision they make is atomic.


def _try_lock(path: Path) -> int | None:
    """Take the flock, returning the fd, or None when someone else holds it.

    `flock` has two failure modes and one return value for both, and only one
    of them is an answer to the question being asked. `EAGAIN`/`EWOULDBLOCK` —
    which Python raises as `BlockingIOError` — means somebody else holds it.
    Everything else means the lock did not happen: `ENOLCK` from NFS with no
    lock daemon, `ENOSYS` or `EINVAL` from some FUSE mounts. Reading those as
    contention turns the lease into a no-op that reports success.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError as exc:
        os.close(fd)
        raise LockingUnavailable(
            f"cannot lock {path}: {exc}. Keeping two writers out of the memory repo "
            "needs a filesystem where flock works; NFS without a lock daemon and some "
            "FUSE mounts answer this way. Put the repo and the database on local disk."
        ) from exc
    return fd


def _unlock(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _is_holder_alive(path: Path) -> bool:
    """True when the flock is currently held by a live process."""
    if not path.exists():
        return False
    try:
        fd = _try_lock(path)
    except LockingUnavailable as exc:
        # This only decides whether a leftover row gets *reported* at startup,
        # and without working locks there is no way to tell a live holder from
        # a dead one. Claiming the holder is alive is the quiet answer: no
        # stale-lease warning on every single start. Saying so once is the
        # loud part, and the `acquire` that is about to write raises instead
        # of guessing.
        log.warning("cannot tell whether anything holds %s: %s", path, exc)
        return True
    if fd is None:
        return True
    _unlock(fd)
    return False


class Lease:
    """Async context manager around the write lease."""

    def __init__(
        self,
        store: Store,
        lock_path: Path,
        *,
        name: str = LEASE_NAME,
        ttl: float = DEFAULT_TTL,
    ) -> None:
        self._store = store
        self._path = lock_path
        self._name = name
        self._ttl = ttl
        self._fd: int | None = None

    async def acquire(self, *, job: str | None = None) -> Self:
        if self._fd is not None:
            raise LeaseError("this process already holds the memory write lease")

        fd = _try_lock(self._path)
        if fd is None:
            raise LeaseError(await self._describe_holder())

        # The flock is ours, so any row still sitting here belongs to a process
        # that died holding it.
        if (stale := await self._store.get_lease(self._name)) is not None:
            log.warning(
                "took over the %s from %s (job %s, since %s), which did not release it",
                _DESCRIPTIONS.get(self._name, self._name),
                stale["holder"],
                stale["job"] or "unknown",
                stale["acquired_at"],
            )

        self._fd = fd
        await self._store.take_lease(self._name, holder=holder_id(), job=job, ttl_seconds=self._ttl)
        return self

    async def release(self) -> None:
        if self._fd is None:
            return
        await self._store.release_lease(self._name)
        # Row first, then the lock: releasing the lock first would let another
        # process take it, see our row, and report a takeover that never happened.
        _unlock(self._fd)
        self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    async def __aenter__(self) -> Self:
        return await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.release()

    async def _describe_holder(self) -> str:
        what = _DESCRIPTIONS.get(self._name, f"{self._name} lease")
        row = await self._store.get_lease(self._name)
        if row is None:
            # The flock is taken before the row is written, so a contender
            # arriving inside that window sees a lock with nobody named. The
            # refusal is still right; only the explanation is missing.
            return (
                f"another process holds the {what} (lock file {self._path}), but did not record who"
            )
        return (
            f"{row['holder']} holds the {what} for job "
            f"{row['job'] or 'unknown'}, taken at {row['acquired_at']}"
        )


async def stale_lease(
    store: Store, lock_path: Path, *, name: str = LEASE_NAME
) -> dict[str, Any] | None:
    """A lease row whose holder is gone, or None.

    Called at startup. A row that survives a crash is harmless on its own — the
    next `acquire` takes over — but it is worth reporting, because it means the
    previous run stopped in the middle of writing to the repo.
    """
    row = await store.get_lease(name)
    if row is None or _is_holder_alive(lock_path):
        return None
    return row

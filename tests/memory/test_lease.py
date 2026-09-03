"""The lease, and the difference between "taken" and "cannot be taken".

`flock` has two quite different failure modes and one return value for both.
`EAGAIN` means somebody else holds the lock, which is the answer the lease is
asking for. Everything else means the lock did not happen — `ENOLCK` from NFS
without a lock daemon, `ENOSYS` and `EINVAL` from some FUSE mounts — and
reading that as "somebody else holds it" makes the whole mechanism a no-op that
reports success.
"""

from __future__ import annotations

import errno
import fcntl
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from kasa.memory.lease import Lease, LeaseError, LockingUnavailable, stale_lease
from kasa.store import Store

_real_flock = fcntl.flock


def refusing(code: int) -> Callable[..., Any]:
    """A `flock` that fails an exclusive take the way a bad mount does."""

    def flock(fd: int, operation: int) -> None:
        if operation & fcntl.LOCK_EX:
            raise OSError(code, os.strerror(code))
        _real_flock(fd, operation)

    return flock


@pytest.mark.parametrize(
    "code", [errno.ENOLCK, errno.ENOSYS, errno.EINVAL, errno.EPERM], ids=os.strerror
)
async def test_a_filesystem_that_cannot_lock_is_not_reported_as_contention(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch, code: int
) -> None:
    """The distinction the rest of this rests on.

    `_reindex` treats a `LeaseError` as "another rebuild is already doing this
    job's work, and it will finish" (#116), which is a safe thing to believe
    only when a `LeaseError` really does mean contention. A mount that cannot
    lock at all would otherwise arrive as the same exception and be skipped —
    a job that reports `done`, indexes nothing, and says nothing.
    """
    monkeypatch.setattr(fcntl, "flock", refusing(code))

    with pytest.raises(LockingUnavailable) as caught:
        await Lease(store, tmp_path / "kasa.lock").acquire()

    assert not isinstance(caught.value, LeaseError), "a skip would swallow it"
    assert os.strerror(code) in str(caught.value)
    assert str(tmp_path / "kasa.lock") in str(caught.value)


async def test_someone_else_holding_it_is_still_contention(tmp_path: Path, store: Store) -> None:
    """The half that must keep working: `EAGAIN` is the answer, not a fault."""
    lock = tmp_path / "kasa.lock"
    held = await Lease(store, lock).acquire(job="the one that got there first")
    try:
        with pytest.raises(LeaseError) as caught:
            await Lease(store, lock).acquire()
    finally:
        await held.release()

    assert "the one that got there first" in str(caught.value)


async def test_the_startup_check_does_not_bring_the_process_down(
    tmp_path: Path, store: Store, monkeypatch: pytest.MonkeyPatch, caplog: Any
) -> None:
    """`stale_lease` runs on every `MemoryStore.open` and in `kasa doctor`, and
    only decides whether to *report* a row a crash left behind. Without working
    locks it cannot tell a live holder from a dead one — so it says nothing
    rather than raising a startup failure or crying wolf on every start. The
    `acquire` that is about to write is where this has to be loud."""
    lock = tmp_path / "kasa.lock"
    lock.touch()
    await store.take_lease("ltm", holder="somebody:1", job=None, ttl_seconds=900.0)
    monkeypatch.setattr(fcntl, "flock", refusing(errno.ENOLCK))

    with caplog.at_level("WARNING", logger="kasa.memory.lease"):
        assert await stale_lease(store, lock) is None

    assert "No locks available" in caplog.text

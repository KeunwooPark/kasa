"""The jobs Kasa actually knows how to run.

Most of `docs/DESIGN.md` §6 arrives later: `episode_close` (#27), `promote`
(#29), `reflect` (#32), `reorganize` (#33), `forget` (#34). Each is a `JobSpec`
registered here, and nothing else about the machinery changes when they do.

`reindex` is the one that exists. The design lists it as "on git change /
manual", so it registers without a cron — but it registers, because the thing
that rebuilds the index after a memory write should be a job like the others
rather than only a command somebody remembers to type.
"""

from __future__ import annotations

import logging

from kasa.config import Config
from kasa.memory.index import MemoryIndex
from kasa.memory.lease import LeaseError
from kasa.memory.ltm import MemoryStore
from kasa.runner.scheduler import Job, JobHandler, JobSpec
from kasa.store import Store

log = logging.getLogger(__name__)


def default_specs(cfg: Config, store: Store) -> list[JobSpec]:
    """Everything this build can run, given what is configured."""
    if not cfg.ltm.configured:
        return []
    return [JobSpec(kind="reindex", handler=_reindex(cfg, store))]


def _reindex(cfg: Config, store: Store) -> JobHandler:
    async def run(job: Job) -> None:
        # Both halves, for the reason `kasa reindex` gives: rebuilding only the
        # SQLite one is what let the index and the manifest disagree about
        # which memories exist.
        memory = await MemoryStore.open(cfg, store)
        try:
            result = await MemoryIndex(store, cfg.ltm.resolved_clone_path()).reindex(
                full=bool(job.payload.get("full"))
            )
        except LeaseError as exc:
            # Losing the index lease (#96) means another rebuild is already
            # doing this job's entire work, and this job's whole point is that
            # the index ends up in step with the repo — which it will. Raising
            # here made that a failed attempt instead: backoff, and a dead
            # letter on the third, for a pass that had nothing left to do.
            #
            # Only around the index half. A `LeaseError` out of
            # `refresh_manifest` is the *write* lease, held by a memory write
            # rather than by another reindex, and nothing else will rebuild the
            # manifest afterwards — so that one is a retry, and the backoff is
            # long enough for the write to finish.
            #
            # `LeaseError` and nothing wider. A filesystem that cannot lock at
            # all raises `LockingUnavailable`, which is not a `LeaseError`
            # precisely so that it lands here as a failure: nobody holds the
            # lock, nobody else is doing this work, and reporting `done` would
            # be a lie about an index that is still empty (#124).
            # WARNING, not INFO: this is a pass that did not run, and `kasa job
            # run` configures logging at ERROR without `-v`, so the one line
            # explaining why the job reported `done` without indexing anything
            # was below the threshold of every command that prints it.
            log.warning(
                "reindex: another rebuild already holds the lease (%s); leaving it to it", exc
            )
            return
        manifest = await memory.refresh_manifest()
        log.info("reindex: %s; %s", result.summary(), manifest.summary())

    return run

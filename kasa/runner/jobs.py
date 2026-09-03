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
        result = await MemoryIndex(store, cfg.ltm.resolved_clone_path()).reindex(
            full=bool(job.payload.get("full"))
        )
        manifest = await memory.refresh_manifest()
        log.info("reindex: %s; %s", result.summary(), manifest.summary())

    return run

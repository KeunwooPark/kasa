"""The jobs Kasa actually knows how to run.

All of `docs/DESIGN.md` §6 is here now. Adding another changes nothing else
about the machinery: a job is a `JobSpec` with a handler and, if it runs on its
own, a cron.

What a spec is registered *on* is what this module decides, and the conditions
differ. `episode_close` needs a model and no repo — it writes to SQLite, and it
is what fills the queue `promote` drains. `reindex` and `forget` need a repo
and no model; `forget` in particular has no judgement in it at all, which is
deliberate for the one job that removes things. `promote`, `reflect` and
`reorganize` need both. A build with neither registers nothing, which is why
`kasa job list` still works on a machine with no API key.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from kasa.adapters.slack.notify import post_message
from kasa.config import Config
from kasa.llm.registry import ProviderRegistry
from kasa.llm.tokens import default_tokenizer
from kasa.memory.index import MemoryIndex
from kasa.memory.lease import LeaseError
from kasa.memory.ltm import MemoryStore
from kasa.memory.retrieve import Retriever
from kasa.redact import Redactor
from kasa.runner.cron import HOURLY, NIGHTLY, WEEKLY, Cron
from kasa.runner.episodes import EpisodeCloser
from kasa.runner.forget import Collector
from kasa.runner.identity import Registrar
from kasa.runner.promote import Promoter
from kasa.runner.reflect import Notifier, Reflector
from kasa.runner.reorganize import Librarian
from kasa.runner.scheduler import Job, JobHandler, JobSpec
from kasa.store import Store

log = logging.getLogger(__name__)

#: Often enough that a thread which went quiet is consolidated while the
#: conversation is still the same day's, rarely enough that the sweep is not
#: itself a cost. The idle threshold is what decides when an episode is over;
#: this only decides how promptly that is noticed.
EVERY_FIVE_MINUTES = "*/5 * * * *"

#: What the design asks for (§6). Hourly is the rhythm the whole product runs
#: at: it is how long after a conversation ends before what was said in it is
#: something a later conversation can recall.
PROMOTE_CRON = HOURLY

#: Weekly, like `reorganize`, but not at the same hour. Both take the memory
#: write lease, and two jobs queued for the same minute means one of them waits
#: out a lease for no reason every week.
FORGET_CRON = "0 5 * * 0"

#: Often enough that somebody who spoke this morning is a `people/` memory by
#: the time `promote` runs and has something to say about them, and rarely
#: enough that a workspace joining in bursts is a few commits rather than one
#: per person. Offset off the hour so it is not queued against `promote`, which
#: wants the same write lease.
IDENTITY_CRON = "7,22,37,52 * * * *"


def default_specs(
    cfg: Config, store: Store, registry: ProviderRegistry | None = None
) -> list[JobSpec]:
    """Everything this build can run, given what is configured.

    `registry` is the daemon's, when there is a daemon. `kasa job run` has no
    long-lived one and passes nothing, so a job that needs a model builds one
    for the run and closes it again — which is also what keeps `kasa job list`
    working on a config with no API key at all.
    """
    models = Models(cfg, store, registry)
    specs = []
    if "chat" in cfg.llm:
        specs.append(
            JobSpec(
                kind="episode_close",
                handler=_episode_close(cfg, store, models),
                cron=Cron.parse(EVERY_FIVE_MINUTES),
            )
        )
    if cfg.ltm.configured and "chat" in cfg.llm:
        specs.append(
            JobSpec(
                kind="promote",
                handler=_promote(cfg, store, models),
                cron=Cron.parse(PROMOTE_CRON),
            )
        )
    if cfg.ltm.configured and "chat" in cfg.llm:
        specs.append(
            JobSpec(kind="reflect", handler=_reflect(cfg, store, models), cron=Cron.parse(NIGHTLY))
        )
    if cfg.ltm.configured and "chat" in cfg.llm:
        specs.append(
            JobSpec(
                kind="reorganize",
                handler=_reorganize(cfg, store, models),
                cron=Cron.parse(WEEKLY),
            )
        )
    if cfg.ltm.configured:
        # No model, so it registers on the repo alone. Everything it decides is
        # already in the corpus.
        specs.append(
            JobSpec(kind="forget", handler=_forget(cfg, store), cron=Cron.parse(FORGET_CRON))
        )
        # Polling the private repo is how a supervised PR becomes visible after
        # a human merges it. The job syncs first and then rebuilds both derived
        # views.
        specs.append(
            JobSpec(kind="reindex", handler=_reindex(cfg, store), cron=Cron.parse("* * * * *"))
        )
    if cfg.ltm.configured and cfg.slack.configured:
        # A repo and a Slack install, and no model: what it writes is a uid and
        # a name that `users.info` already handed us. Registered on the Slack
        # settings rather than run unconditionally because a build with no
        # Slack has nobody to map, and an empty sweep every five minutes is
        # still a query every five minutes.
        specs.append(
            JobSpec(
                kind="identity",
                handler=_identity(cfg, store),
                cron=Cron.parse(IDENTITY_CRON),
            )
        )
    return specs


class Models:
    """The registry a job calls, however this process happens to have one.

    A daemon builds one registry and keeps it: its providers hold an httpx
    client, and a connection pool that is rebuilt every five minutes is not a
    pool. A one-shot `kasa job run` has none to lend, and building one at
    import time would make every command that merely *lists* jobs fail on a
    machine with no API key exported.
    """

    def __init__(self, cfg: Config, store: Store, registry: ProviderRegistry | None) -> None:
        self._cfg = cfg
        self._store = store
        self._registry = registry

    @asynccontextmanager
    async def use(self) -> AsyncIterator[ProviderRegistry]:
        if self._registry is not None:
            yield self._registry
            return
        registry = self._cfg.build_registry(store=self._store)
        try:
            yield registry
        finally:
            await registry.aclose()


def _episode_close(cfg: Config, store: Store, models: Models) -> JobHandler:
    async def run(job: Job) -> None:
        async with models.use() as registry:
            closer = EpisodeCloser(store, registry, cfg.episodes)
            # A payload naming a session is an explicit end — someone left, or
            # a surface knows the thread is finished — and closes that session's
            # episode however recent it is. With no payload this is the clock,
            # and idleness is the only evidence there is.
            if session_id := job.payload.get("session_id"):
                result = await closer.end_session(str(session_id))
            else:
                result = await closer.sweep()
        log.info("episode_close: %s", result.summary())

    return run


def _promote(cfg: Config, store: Store, models: Models) -> JobHandler:
    async def run(job: Job) -> None:
        memory = await MemoryStore.open(cfg, store)
        # Scrubbed, like every other path that sends memory text onwards. The
        # competition offered to the planner is read out of the corpus, and a
        # secret that got into a memory file must not get back out through a
        # prompt.
        retriever = Retriever(
            store,
            tokenizer=default_tokenizer(),
            budget_tokens=cfg.context.tokens_for_retrieval(),
            scrub=Redactor.from_config(cfg).scrub,
        )
        async with models.use() as registry:
            result = await Promoter(
                store,
                memory,
                retriever,
                registry,
                policy=cfg.memory,
                settings=cfg.promote,
                job_id=job.id,
            ).run()
        log.info("promote: %s", result.summary())

    return run


def _reflect(cfg: Config, store: Store, models: Models) -> JobHandler:
    async def run(job: Job) -> None:
        memory = await MemoryStore.open(cfg, store)
        async with models.use() as registry:
            result = await Reflector(
                store,
                memory,
                registry,
                settings=cfg.reflect,
                policy=cfg.memory,
                notify=_digest_sink(cfg),
                job_id=job.id,
            ).run()
        log.info("reflect: %s", result.summary())

    return run


def _reorganize(cfg: Config, store: Store, models: Models) -> JobHandler:
    async def run(job: Job) -> None:
        memory = await MemoryStore.open(cfg, store)
        async with models.use() as registry:
            result = await Librarian(
                store,
                memory,
                registry,
                settings=cfg.reorganize,
                policy=cfg.memory,
                job_id=job.id,
            ).run()
        log.info("reorganize: %s", result.summary())

    return run


def _forget(cfg: Config, store: Store) -> JobHandler:
    async def run(job: Job) -> None:
        memory = await MemoryStore.open(cfg, store)
        result = await Collector(
            store, memory, settings=cfg.forget, policy=cfg.memory, job_id=job.id
        ).run()
        log.info("forget: %s", result.summary())

    return run


def _identity(cfg: Config, store: Store) -> JobHandler:
    async def run(job: Job) -> None:
        memory = await MemoryStore.open(cfg, store)
        result = await Registrar(store, memory, policy=cfg.memory, job_id=job.id).run()
        log.info("identity: %s", result.summary())

    return run


def _digest_sink(cfg: Config) -> Notifier | None:
    """Where the nightly digest goes, if anywhere.

    None unless both halves are configured, and it is not an error to have
    neither: the digest is the one optional part of `reflect`, and a job that
    refused to run because nobody wanted a Slack message would be refusing to
    do the work that matters.
    """
    channel = cfg.reflect.digest_channel
    token = os.environ.get(cfg.slack.bot_token_env or "")
    if not channel:
        return None
    if not token:
        log.warning(
            "reflect: a digest channel is configured but %s is not set; nothing will be posted",
            cfg.slack.bot_token_env or "the Slack bot token env var",
        )
        return None

    async def post(text: str) -> None:
        await post_message(token, channel, text)

    return post


def _reindex(cfg: Config, store: Store) -> JobHandler:
    async def run(job: Job) -> None:
        # Both halves, for the reason `kasa reindex` gives: rebuilding only the
        # SQLite one is what let the index and the manifest disagree about
        # which memories exist.
        memory = await MemoryStore.open(cfg, store)
        changed = await asyncio.to_thread(memory.sync_default)
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
        log.info(
            "reindex%s: %s; %s",
            " after pulling a merged review" if changed else "",
            result.summary(),
            manifest.summary(),
        )

    return run

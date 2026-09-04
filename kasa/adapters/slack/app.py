"""Socket Mode: Slack ingress and egress for a daemon with no public address.

Ingress does exactly one thing — normalize, enqueue, return. Slack gives a
listener three seconds before it treats the event as undelivered and re-sends
it, and bolt acknowledges once the listener returns, so anything slow on this
path becomes a retry storm on top of whatever was already slow. The agent runs
behind the queue, where taking a minute costs nothing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Self

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from kasa.adapters.slack.events import (
    Accepted,
    Changed,
    Ignored,
    Reacted,
    SlackContext,
    message_id,
    normalize,
    reaction,
)
from kasa.adapters.slack.identity import Directory
from kasa.adapters.slack.stream import DEFAULT_INTERVAL, LiveMessage, SlackRateLimited
from kasa.config import SlackSettings
from kasa.core.agent import Agent, AgentResult
from kasa.core.events import InboundEvent
from kasa.core.feedback import Feedback
from kasa.core.revise import Reviser
from kasa.core.runtime import DEFAULT_CONCURRENCY, Reply, Runtime, one_message
from kasa.errors import ConfigError
from kasa.llm.types import Delta

log = logging.getLogger(__name__)

#: Socket Mode carries no signed HTTP requests, so bolt has nothing to verify
#: and no secret to verify it with. Saying so is clearer than leaving bolt to
#: warn about a secret that does not apply here.
NO_HTTP_VERIFICATION = "unused-in-socket-mode"

#: How many unfinished live messages are remembered, so a redelivered event
#: rewrites the message its last attempt posted instead of adding another
#: "thinking…" to the thread. Only turns that failed outright stay here — a
#: finished one is forgotten — and the inbox dead-letters after five attempts,
#: so this is a cap on a leak rather than a working set.
MAX_UNFINISHED = 256


class SlackAdapter:
    """One Slack workspace, connected over a socket Kasa opens outbound."""

    def __init__(
        self,
        agent: Agent,
        *,
        app: AsyncApp,
        context: SlackContext,
        app_token: str,
        concurrency: int = DEFAULT_CONCURRENCY,
        stream: bool = True,
        interval: float = DEFAULT_INTERVAL,
        reactions: Mapping[str, str] | None = None,
    ) -> None:
        self._app = app
        self._context = context
        self._app_token = app_token
        self._interval = interval
        self._reactions = dict(reactions if reactions is not None else SlackSettings().reactions)
        self._unfinished: dict[str, str] = {}
        self.directory = Directory(agent.store, self._users_info, team_id=context.team_id)
        self.reviser = Reviser(agent.store)
        self.feedback = Feedback(agent.store)
        self.runtime = Runtime(
            agent,
            self.open_reply if stream else one_message(self.reply),
            concurrency=concurrency,
            prepare=self.directory.hydrate,
        )
        self._handler: AsyncSocketModeHandler | None = None
        self._register()

    @classmethod
    async def connect(
        cls, agent: Agent, settings: SlackSettings, *, concurrency: int = DEFAULT_CONCURRENCY
    ) -> Self:
        """Build an adapter, having asked Slack who it is.

        The identity is not decoration: the bot's own user id is what tells a
        mention from chatter and Kasa's own messages from everyone else's, and
        the team id is half of every session key.
        """
        app = AsyncApp(
            token=_token(settings.bot_token_env, "bot"),
            signing_secret=NO_HTTP_VERIFICATION,
            request_verification_enabled=False,
        )
        identity = await app.client.auth_test()
        return cls(
            agent,
            app=app,
            context=SlackContext(
                bot_user_id=str(identity["user_id"]),
                team_id=str(identity["team_id"]),
                allowed_channels=frozenset(settings.allowed_channels),
            ),
            app_token=_token(settings.app_token_env, "app"),
            concurrency=concurrency,
            stream=settings.stream,
            reactions=settings.reactions,
        )

    @property
    def context(self) -> SlackContext:
        return self._context

    @property
    def client(self) -> AsyncWebClient:
        return self._app.client

    # -- ingress -------------------------------------------------------------

    async def on_event(self, event: dict[str, Any]) -> None:
        """The whole three-second path: one decision and one INSERT.

        Nothing here awaits a model, which is the entire reason `inbox` exists.
        """
        try:
            decision = await normalize(
                event, context=self._context, known_session=self._known_session
            )
        except Exception:
            # Belt to the braces in `events.py`. Everything up to the INSERT is
            # a judgement about a payload, and a judgement that raises loses
            # the message: bolt hands the failure back to Slack, Slack retries
            # into the same exception three times, and then the message is gone
            # with nothing recording that it arrived. Dropping it loudly is
            # worse than answering it and better than dropping it in silence.
            #
            # Only around `normalize`. A failure out of `submit` is the store,
            # and there the row genuinely was not written — letting that reach
            # bolt is what gets the message re-sent, which is the outcome we
            # want.
            log.exception(
                "could not read a slack event (type=%r, ts=%r); ignoring it",
                event.get("type"),
                event.get("ts"),
            )
            return
        if isinstance(decision, Ignored):
            log.debug("ignoring a slack event: %s", decision.reason)
            return
        if isinstance(decision, Changed):
            await self.revise(decision)
            return
        if not isinstance(decision, Accepted):
            # `normalize` reads messages and never returns a reaction, but the
            # decision type covers every surface event; saying so here is what
            # keeps a future one from falling through into `submit` and being
            # answered.
            log.debug("a message decision this path does not handle: %s", decision)
            return
        enqueued = await self.runtime.submit(decision.event)
        if enqueued.duplicate:
            # Either Slack re-sent the event, or the same message reached us as
            # both `app_mention` and `message`. Both are normal; both are the
            # unique constraint doing its job.
            log.debug("slack sent %s again; already queued", decision.event.external_id)

    async def on_reaction(self, event: dict[str, Any]) -> None:
        """An emoji on one of Kasa's answers, which is the whole feedback loop.

        On the ack path with the revisions, and for the same reasons: a lookup
        and an INSERT, no model and no network, and it must not reach the agent
        — a 👍 is not a question.
        """
        try:
            decision = reaction(event, context=self._context, verdicts=self._reactions)
        except Exception:
            log.exception("could not read a slack reaction (%r)", event.get("reaction"))
            return
        if not isinstance(decision, Reacted):
            log.debug("ignoring a slack reaction: %s", getattr(decision, "reason", decision))
            return

        act = self.feedback.withdraw if decision.removed else self.feedback.record
        outcome = await act(
            source="slack",
            external_id=decision.external_id,
            verdict=decision.verdict,
            author=decision.author,
        )
        log.debug("reaction on %s: %s", decision.external_id, outcome.summary())

    async def remember_answer(self, event: InboundEvent, result: AgentResult, ts: str) -> None:
        """Note which memories produced the message just posted at `ts`.

        Recorded whatever it used, including nothing: a reaction on an answer
        that recalled no memory is still a fact about the answer, and a row
        that exists with an empty list is how a later 👍 tells "nothing to
        boost" from "not one of ours" (#36).
        """
        if not event.channel or not ts:
            return
        await self.runtime.store.record_answer(
            source="slack",
            external_id=message_id(self._context.team_id, event.channel, ts),
            memory_ids=result.memory_ids,
            session_id=event.session_id,
            scope=event.scope,
        )

    async def revise(self, decision: Changed) -> None:
        """Apply an edit or a deletion, here on the ack path rather than behind
        the queue.

        Deliberately, and it is the one thing besides the INSERT that runs
        here. The work is a handful of indexed statements against SQLite with
        no model and no network in it, so it fits the three seconds many times
        over — and it must not go through the inbox, because everything that
        comes out of the inbox is delivered to the agent as something to answer,
        and "Jane fixed a typo" is not a question.

        Durability comes from Slack instead: raising means bolt does not ack,
        Slack re-sends, and every step here is idempotent — a rewrite to the
        same words, a state already set, a review deduped on its key.
        """
        revision = decision.revision
        if revision.text is not None:
            # Cache-only, so the ack path stays free of `users.info`. The
            # people in an edited message were almost always resolved when the
            # original arrived.
            revision = replace(revision, text=await self.directory.rename_known(revision.text))
        outcome = await self.reviser.apply(revision)
        log.debug("revision %s: %s", revision.external_id, outcome.summary())

    async def _known_session(self, session_id: str) -> bool:
        return await self.runtime.store.get_session(session_id) is not None

    async def _users_info(self, user_id: str) -> Mapping[str, Any]:
        """One profile, unwrapped from the envelope Slack puts it in.

        `Directory` is given this rather than the client so that it needs no
        `slack_sdk` — and so the thing under test is the caching, not a mock of
        somebody else's response object.
        """
        response = await self.client.users_info(user=user_id)
        user = response.get("user")
        return user if isinstance(user, Mapping) else {}

    # -- egress --------------------------------------------------------------

    async def open_reply(self, event: InboundEvent) -> Reply:
        """Put a message up now, and rewrite it as the turn produces an answer.

        In the thread the question was asked in, always. A turn that takes
        thirty seconds and says nothing for twenty-nine of them is
        indistinguishable from one that broke, and somebody who thinks Kasa
        broke asks again — which is a second turn, a second model call, and two
        answers to one question.
        """
        if not event.channel:
            return _Posted(self, event)
        message = LiveMessage(
            _ClientPoster(self.client),
            channel=event.channel,
            thread_ts=event.reply_to,
            interval=self._interval,
            ts=self._unfinished.get(event.external_id),
        )
        await message.open()
        if message.ts is not None:
            self._remember(event.external_id, message.ts)
        return _Live(self, event, message)

    async def reply(self, event: InboundEvent, result: AgentResult) -> None:
        """Post the answer as one message, having shown nothing before it.

        What a build with `stream: false` uses, and where a live reply lands
        when its placeholder never went up.
        """
        text = answer(result)
        if not text or not event.channel:
            log.warning("nothing to post for %s", event.external_id)
            return
        posted = await self.client.chat_postMessage(
            channel=event.channel, thread_ts=event.reply_to, text=text
        )
        await self.remember_answer(event, result, str(posted.get("ts") or ""))

    def _remember(self, external_id: str, ts: str) -> None:
        self._unfinished[external_id] = ts
        while len(self._unfinished) > MAX_UNFINISHED:
            # Oldest first, which for a dict is insertion order. Losing one
            # costs a duplicate placeholder on a redelivery that was already
            # four attempts deep.
            self._unfinished.pop(next(iter(self._unfinished)))

    def _forget(self, external_id: str) -> None:
        self._unfinished.pop(external_id, None)

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        """Open the socket and return. Reconnects are the client's own affair."""
        self._handler = AsyncSocketModeHandler(self._app, self._app_token)
        # Unannotated in bolt, so `strict` will not take its word for it.
        await self._handler.connect_async()  # type: ignore[no-untyped-call]

    async def aclose(self) -> None:
        if self._handler is not None:
            await self._handler.close_async()  # type: ignore[no-untyped-call]
            self._handler = None

    # -- internals -----------------------------------------------------------

    def _register(self) -> None:
        async def listener(event: dict[str, Any]) -> None:
            await self.on_event(event)

        # Both, on purpose. `app_mention` is all a minimally-scoped install
        # gets; `message` is what carries DMs and thread replies. Where an
        # install has both, the same message arrives twice and the dedupe key
        # in `events.message_id` is what makes that harmless — which is only
        # true because `normalize` reads the two payloads identically. The
        # duplicate is discarded unexamined, so anything the two disagree
        # about is decided by whichever arrived first.
        for name in ("app_mention", "message"):
            self._app.event(name)(listener)

        async def reacted(event: dict[str, Any]) -> None:
            await self.on_reaction(event)

        # Their own listener, because a reaction is not a message: it has no
        # text and nothing to answer, and everything `normalize` decides about
        # a message is a question it cannot be asked.
        for name in ("reaction_added", "reaction_removed"):
            self._app.event(name)(reacted)


def answer(result: AgentResult) -> str:
    """The finished message: what the model said, and why if it stopped early.

    One function because both egress paths render it, and a live reply that
    formatted the answer differently from the plain one would make "did the
    placeholder go up?" visible to the person reading the thread.
    """
    text = result.text.strip()
    if note := result.note:
        return f"{text}\n\n_{note}_" if text else f"_{note}_"
    return text


class _Live:
    """A turn whose answer is a message already in the thread."""

    def __init__(self, adapter: SlackAdapter, event: InboundEvent, message: LiveMessage) -> None:
        self._adapter = adapter
        self._event = event
        self._message = message

    async def delta(self, delta: Delta) -> None:
        await self._message.delta(delta)

    async def finish(self, result: AgentResult) -> None:
        await self._message.finish(answer(result))
        # Only now: until the answer is in the thread, a redelivery of this
        # event should rewrite this message rather than post another.
        self._adapter._forget(self._event.external_id)
        if self._message.ts is not None:
            await self._adapter.remember_answer(self._event, result, self._message.ts)

    async def aclose(self) -> None:
        await self._message.aclose()


class _Posted:
    """A turn with nowhere to put a placeholder. Answers the old way."""

    def __init__(self, adapter: SlackAdapter, event: InboundEvent) -> None:
        self._adapter = adapter
        self._event = event

    async def delta(self, delta: Delta) -> None:
        pass

    async def finish(self, result: AgentResult) -> None:
        await self._adapter.reply(self._event, result)

    async def aclose(self) -> None:
        pass


class _ClientPoster:
    """`Poster`, over the Slack web client.

    The translation lives here rather than in `stream.py` for the reason the
    directory's lookup does: what streaming needs is two writes and a way to
    hear "not so fast", and keeping `slack_sdk` on this side is what lets the
    throttling be tested without it.
    """

    def __init__(self, client: AsyncWebClient) -> None:
        self._client = client

    async def post(self, *, channel: str, thread_ts: str | None, text: str) -> str:
        try:
            response = await self._client.chat_postMessage(
                channel=channel, thread_ts=thread_ts, text=text
            )
        except SlackApiError as exc:
            raise _translate(exc) from exc
        return str(response["ts"])

    async def update(self, *, channel: str, ts: str, text: str) -> None:
        try:
            await self._client.chat_update(channel=channel, ts=ts, text=text)
        except SlackApiError as exc:
            raise _translate(exc) from exc


def _translate(exc: SlackApiError) -> Exception:
    """Slack's "too fast", told from Slack's "no".

    Only a 429 becomes `SlackRateLimited`, because it is the only one where
    waiting is the answer. Everything else is passed along as it was: a
    live frame swallows it and the final write fails the turn, which is the
    right split for `channel_not_found` or a revoked token.
    """
    response = getattr(exc, "response", None)
    if response is None or getattr(response, "status_code", None) != 429:
        return exc
    headers = getattr(response, "headers", {}) or {}
    try:
        retry_after = float(headers.get("Retry-After", 1))
    except (TypeError, ValueError):
        retry_after = 1.0
    return SlackRateLimited(retry_after)


def _token(env: str | None, kind: str) -> str:
    if not env:
        raise ConfigError(f"no {kind} token configured for Slack; run `kasa init`")
    value = os.environ.get(env)
    if not value:
        raise ConfigError(f"{env} is not set (the Slack {kind} token)")
    return value

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
from typing import Any, Self

from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from kasa.adapters.slack.events import Ignored, SlackContext, normalize
from kasa.adapters.slack.identity import Directory
from kasa.config import SlackSettings
from kasa.core.agent import Agent, AgentResult
from kasa.core.events import InboundEvent
from kasa.core.runtime import DEFAULT_CONCURRENCY, Runtime
from kasa.errors import ConfigError

log = logging.getLogger(__name__)

#: Socket Mode carries no signed HTTP requests, so bolt has nothing to verify
#: and no secret to verify it with. Saying so is clearer than leaving bolt to
#: warn about a secret that does not apply here.
NO_HTTP_VERIFICATION = "unused-in-socket-mode"


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
    ) -> None:
        self._app = app
        self._context = context
        self._app_token = app_token
        self.directory = Directory(agent.store, self._users_info, team_id=context.team_id)
        self.runtime = Runtime(
            agent, self.reply, concurrency=concurrency, prepare=self.directory.hydrate
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
        enqueued = await self.runtime.submit(decision.event)
        if enqueued.duplicate:
            # Either Slack re-sent the event, or the same message reached us as
            # both `app_mention` and `message`. Both are normal; both are the
            # unique constraint doing its job.
            log.debug("slack sent %s again; already queued", decision.event.external_id)

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

    async def reply(self, event: InboundEvent, result: AgentResult) -> None:
        """Post the answer, in the thread the question was asked in."""
        text = result.text.strip()
        if note := result.note:
            text = f"{text}\n\n_{note}_" if text else f"_{note}_"
        if not text or not event.channel:
            log.warning("nothing to post for %s", event.external_id)
            return
        await self.client.chat_postMessage(
            channel=event.channel, thread_ts=event.reply_to, text=text
        )

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


def _token(env: str | None, kind: str) -> str:
    if not env:
        raise ConfigError(f"no {kind} token configured for Slack; run `kasa init`")
    value = os.environ.get(env)
    if not value:
        raise ConfigError(f"{env} is not set (the Slack {kind} token)")
    return value

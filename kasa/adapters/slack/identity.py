"""Who a Slack user id belongs to, cached, and readable in the transcript.

Two jobs, one cache.

The first is that `<@U0456>` is not a name. It reaches the model as an opaque
id, so a thread where three people are discussed reads as a thread where three
identifiers are discussed, and nothing the agent later remembers about any of
them can be matched to the person in the next conversation. Resolving mentions
before the turn is what makes the transcript say what the room said.

The second is the mapping itself: one `people/<slug>.md` per person, holding
`slack://<team>/<user>`, so a DM and a channel are two conversations with the
same person rather than two people. This module only records what it saw;
`kasa.runner.identity` is what writes the memory, on a job, off the turn path.

No `slack_bolt` import, and no web client: the lookup arrives as a callable.
That keeps the whole of it testable on an install that never asked for the
`slack` extra, which is the same reason `events.py` holds no socket.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from kasa.adapters.slack.events import MENTION
from kasa.core.events import InboundEvent
from kasa.store import Store

log = logging.getLogger(__name__)

#: How long a cached profile is trusted before `users.info` is asked again.
#: A workspace's membership is close to static, and this is the *only* thing
#: standing between a busy channel and one API call per mention per message.
#: Its cost is that a rename takes up to this long to reach the corpus, which
#: for a display name is not worth a request per turn to avoid.
DEFAULT_TTL = timedelta(hours=24)

#: One profile fetch: uid in, the `user` object from `users.info` out. An empty
#: mapping means Slack had nothing to say about that id, which is a real answer
#: — ids appear in text that no longer resolve to anybody.
UserLookup = Callable[[str], Awaitable[Mapping[str, Any]]]


def user_ref(team_id: str, user_id: str) -> str:
    """How a `people/` memory records the mapping, and what the identity job
    matches an existing memory on.

    Written once, here, because both halves have to agree about it forever:
    change the spelling and every already-linked person forks into a second
    file the next time they are seen.
    """
    return f"slack://{team_id}/{user_id}"


@dataclass(frozen=True, slots=True)
class SlackUser:
    """A person, as far as the workspace directory is concerned."""

    user_id: str
    display_name: str = ""
    real_name: str = ""
    is_bot: bool = False
    deleted: bool = False

    @property
    def name(self) -> str:
        """What to call them, with the id as the answer of last resort.

        Display name first because it is what the workspace shows and what
        people type; `real_name` is often empty, and sometimes a legal name
        nobody in the channel would recognize.
        """
        return self.display_name or self.real_name or self.user_id


def read_user(user_id: str, payload: Mapping[str, Any]) -> SlackUser:
    """Read one `users.info` user object, assuming nothing about its shape.

    Every field is optional in practice — a bot user has no profile worth the
    name, a deactivated account keeps almost nothing — and this runs on the
    turn path, where an unexpected payload has to degrade to "we do not know
    who that is" rather than end the turn.
    """
    raw = payload.get("profile")
    profile: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    display = _text(profile.get("display_name")) or _text(payload.get("name"))
    real = _text(profile.get("real_name")) or _text(payload.get("real_name"))
    return SlackUser(
        user_id=user_id,
        display_name=display,
        real_name=real,
        is_bot=bool(payload.get("is_bot")),
        deleted=bool(payload.get("deleted")),
    )


class Directory:
    """The workspace's people, looked up once and remembered."""

    def __init__(
        self,
        store: Store,
        lookup: UserLookup,
        *,
        team_id: str,
        ttl: timedelta = DEFAULT_TTL,
    ) -> None:
        self._store = store
        self._lookup = lookup
        self._team_id = team_id
        self._ttl = ttl

    async def hydrate(self, event: InboundEvent) -> InboundEvent:
        """Put names where the ids were, and note everybody this message saw.

        Never raises. This sits between the queue and the agent, and a
        directory that is down is a reason to answer with ids left in the text
        — not a reason to fail the turn and have the message redelivered until
        its retry budget runs out.
        """
        try:
            return await self._hydrate(event)
        except Exception:
            log.exception("could not resolve Slack identities for %s", event.external_id)
            return event

    async def rename_known(self, text: str) -> str:
        """Resolve mentions from the cache alone, asking Slack nothing.

        For the paths that run inside an ack budget rather than behind the
        queue — an edit arriving for a message already stored (#25). A name
        that is not cached stays an id, which is the same outcome `hydrate`
        reaches when a lookup fails, and cheaper than a `users.info` inside
        three seconds.
        """
        wanted = {match["user"] for match in MENTION.finditer(text)}
        if not wanted:
            return text
        names = {}
        for user_id in sorted(wanted):
            row = await self._store.get_slack_user(self._team_id, user_id)
            if row is not None:
                names[user_id] = _from_row(row).name
        return _render_mentions(text, names)

    async def resolve(self, user_id: str) -> SlackUser | None:
        """Who this id is, from cache while the cache is still warm.

        `None` is "we could not find out", which is not the same as "nobody":
        a failed lookup leaves no row, so the next message tries again rather
        than caching a gap.
        """
        cached = await self._store.get_slack_user(self._team_id, user_id)
        if cached is not None and not self._stale(cached):
            return _from_row(cached)

        try:
            payload = await self._lookup(user_id)
        except Exception as exc:
            # Falling back to a stale profile rather than to nothing: a name
            # from yesterday is right far more often than an id is readable,
            # and the row refreshes the next time Slack answers.
            log.warning("users.info failed for %s: %s", user_id, exc)
            return _from_row(cached) if cached is not None else None

        if not payload:
            return _from_row(cached) if cached is not None else None

        user = read_user(user_id, payload)
        await self._store.upsert_slack_user(
            team_id=self._team_id,
            user_id=user.user_id,
            display_name=user.display_name,
            real_name=user.real_name,
            is_bot=user.is_bot,
            deleted=user.deleted,
        )
        return user

    # -- internals -----------------------------------------------------------

    async def _hydrate(self, event: InboundEvent) -> InboundEvent:
        mentioned = {match["user"] for match in MENTION.finditer(event.text)}
        # The author too, and even when they mentioned nobody. Being spoken to
        # is the commonest way Kasa meets somebody, and a directory that only
        # learned about people who get @-mentioned would never record the one
        # person in a DM.
        wanted = mentioned | ({event.author} if event.author else set())
        if not wanted:
            return event

        found = await asyncio.gather(*(self.resolve(uid) for uid in sorted(wanted)))
        names = {user.user_id: user.name for user in found if user is not None}
        text = _render_mentions(event.text, names)
        return event if text == event.text else event.model_copy(update={"text": text})

    def _stale(self, row: Mapping[str, Any]) -> bool:
        fetched = _parse(str(row["fetched_at"]))
        return fetched is None or datetime.now(UTC) - fetched >= self._ttl


def _render_mentions(text: str, names: Mapping[str, str]) -> str:
    """`<@U0456>` becomes `@Jane`, and an id we do not know stays as it was.

    Leaving the unknown ones alone is deliberate. Rewriting one to a bare
    `@U0456` would look like a name while being one nobody can act on, and
    dropping it would delete a participant from the sentence.
    """

    def replace(match: re.Match[str]) -> str:
        name = names.get(match["user"])
        if not name:
            return match[0]
        return f"{match['before']}@{name}{match['after']}"

    return MENTION.sub(replace, text)


def _from_row(row: Mapping[str, Any]) -> SlackUser:
    return SlackUser(
        user_id=str(row["user_id"]),
        display_name=str(row["display_name"]),
        real_name=str(row["real_name"] or ""),
        is_bot=bool(row["is_bot"]),
        deleted=bool(row["deleted"]),
    )


def _text(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _parse(stamp: str) -> datetime | None:
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment

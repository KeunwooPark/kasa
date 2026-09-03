"""Deciding what a Slack event means, with nothing else in the way.

No `slack_bolt` import, no network, no database of its own. Every judgement
that decides whether a message is for Kasa and what it may be remembered under
lives in this module, because those are the judgements that leak a private
conversation when they are wrong — and they should be testable without a
socket.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from kasa.core.events import InboundEvent

SOURCE = "slack"

#: `<@U123>` and `<@U123|display-name>`, which is how Slack writes a mention —
#: together with any run of spaces or tabs either side of it. The gap is part
#: of the match because removing a mention has to remove the space it was
#: sitting in, and that is the *only* whitespace it is allowed to touch.
_MENTION = re.compile(r"(?P<before>[ \t]*)<@(?P<user>[A-Z0-9]+)(?:\|[^>]*)?>(?P<after>[ \t]*)")

#: Slack channel ids are prefixed by kind, and a one-to-one conversation with
#: the bot is `D`. Derived from the id rather than from `channel_type`, which
#: only one of the two deliveries carries: a mention in a DM arrives as both
#: `message` (with `channel_type: "im"`) and `app_mention` (without it), under
#: one `ts` and therefore one dedupe key — so whichever landed first decided
#: whether the conversation was private, and which one that was is a race.
_DM_PREFIX = "D"

#: Subtypes that are still somebody talking. Slack puts a `subtype` on a
#: message for two quite different reasons: because of *how* it was composed —
#: a file attached, a `/me`, a thread reply also sent to the channel — and
#: because it is not somebody talking at all: an edit, a deletion, a join, a
#: topic change, a pin.
#:
#: Named rather than excluded, because silence stays the default. Kasa answers
#: in any thread it is already part of, so a denylist would answer every
#: unknown subtype that turned up in one, and "Bob joined the channel" is not a
#: question. The cost of an allowlist is that the next subtype carrying a
#: person's words is ignored until it is added here — a line, rather than the
#: class of bug.
_SPOKEN_SUBTYPES = frozenset({"file_share", "me_message", "thread_broadcast"})


@dataclass(frozen=True, slots=True)
class SlackContext:
    """What the adapter knows that a single event does not carry."""

    bot_user_id: str
    team_id: str
    #: Empty means every channel Kasa has been invited to. Inviting a bot to a
    #: channel is already a deliberate act by a person, so an empty list is
    #: "no *further* restriction" rather than "no restriction at all". Set it
    #: when Kasa is in channels it should read and channels it should not.
    allowed_channels: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Accepted:
    event: InboundEvent


@dataclass(frozen=True, slots=True)
class Ignored:
    reason: str


Decision = Accepted | Ignored

#: Whether Kasa already has a conversation under this session id — which is how
#: a reply in a thread it is part of is told from chatter it should stay out of.
KnownSession = Callable[[str], Awaitable[bool]]


def session_id(team: str, channel: str, thread: str) -> str:
    """The actor key: one Slack thread, one serialized conversation."""
    return f"{SOURCE}:{team}:{channel}:{thread}"


def message_id(team: str, channel: str, ts: str) -> str:
    """The dedupe key: the message itself, not the delivery.

    Slack's own `event_id` would dedupe its retries, and only those. A mention
    in a channel Kasa can read arrives *twice* — once as `app_mention`, once as
    `message` — under two different event ids and one `ts`. Keying on the
    message covers both, and covers them without knowing which subscriptions a
    given installation was granted.
    """
    return f"{SOURCE}:{team}:{channel}:{ts}"


async def normalize(
    event: dict[str, Any], *, context: SlackContext, known_session: KnownSession
) -> Decision:
    """Turn one Slack event into something to answer, or say why not."""
    subtype = str(event.get("subtype") or "")
    if subtype and subtype not in _SPOKEN_SUBTYPES:
        # `message_changed` and `message_deleted` are real signals, and #25 is
        # where they get handled. Until then, editing a message must not read
        # as sending a new one.
        return Ignored(f"message subtype {subtype!r}")
    if event.get("bot_id"):
        return Ignored("posted by a bot")

    author = str(event.get("user") or "")
    if not author:
        return Ignored("no author")
    if author == context.bot_user_id:
        return Ignored("posted by Kasa")

    channel = str(event.get("channel") or "")
    ts = str(event.get("ts") or "")
    if not channel or not ts:
        return Ignored("no channel or timestamp")

    text = str(event.get("text") or "")
    is_dm = channel.startswith(_DM_PREFIX)
    thread = str(event.get("thread_ts") or ts)
    session = session_id(context.team_id, channel, thread)

    if not is_dm:
        if context.allowed_channels and channel not in context.allowed_channels:
            return Ignored(f"channel {channel} is not on the allowlist")
        # In a channel, silence is the default. Kasa answers when it is spoken
        # to, and thereafter in that thread — which is a question about a
        # conversation that already exists, not about this message.
        if not _mentions(text, context.bot_user_id) and not await known_session(session):
            return Ignored("not addressed to Kasa")

    return Accepted(
        InboundEvent(
            source=SOURCE,
            external_id=message_id(context.team_id, channel, ts),
            session_id=session,
            text=_with_attachments(_strip_mention(text, context.bot_user_id), event),
            scope=scope_for(channel, author, is_dm=is_dm),
            author=author,
            channel=channel,
            # Always in-thread, and a top-level message starts one. Answering a
            # busy channel at top level is how a bot becomes something people
            # mute.
            reply_to=thread,
        )
    )


def scope_for(channel: str, author: str, *, is_dm: bool) -> str:
    """What a session here is allowed to have remembered about it.

    A DM belongs to the person in it; anything else belongs to its channel.
    Nothing from Slack is `workspace` — that is the widest scope there is, and
    widening one is a decision for #24 with a person in the loop, not a default
    that every public channel picks up on the way in.
    """
    return f"private:{author}" if is_dm else f"channel:{channel}"


def _with_attachments(text: str, event: dict[str, Any]) -> str:
    """Say what came attached, since Kasa cannot open it yet.

    Without this a `file_share` reaches the agent as its comment alone —
    "what's in this?" with nothing in it, and no way to tell that a file is
    what it is being asked about. Naming them is what lets it say it cannot
    read them, which is the point of accepting the message at all.
    """
    names = [str(item.get("name") or "an untitled file") for item in event.get("files") or []]
    if not names:
        return text
    attached = f"[attached, which Kasa cannot open: {', '.join(names)}]"
    return f"{text}\n\n{attached}" if text else attached


def _mentions(text: str, bot_user_id: str) -> bool:
    return any(match["user"] == bot_user_id for match in _MENTION.finditer(text))


def _strip_mention(text: str, bot_user_id: str) -> str:
    """Drop Kasa's own @-mention; leave everyone else's, and the layout, alone.

    The mention is addressing, not content, and leaving it in means every turn
    opens with a user id the model has to decide what to do with. Tidying the
    gap it leaves behind is worth one space; it is not worth the shape of the
    message.

    This used to end `" ".join(without.split())`. `str.split()` with no
    argument splits on every run of whitespace, newlines included, so pasted
    code, stack traces, numbered lists and multi-paragraph questions all
    reached the agent as one run-on line.
    """

    def drop(match: re.Match[str]) -> str:
        if match["user"] != bot_user_id:
            return match[0]
        # Between two words, leave one space so they do not run together. At
        # either end of a line, the mention takes its gap with it.
        return " " if match["before"] and match["after"] else ""

    return _MENTION.sub(drop, text).strip()

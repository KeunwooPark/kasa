"""What a surface delivered, in the one shape the rest of Kasa understands.

An adapter's whole job at ingress is to turn a provider's event into an
`InboundEvent` and enqueue it. Everything downstream — the dispatcher, the
session router, the agent loop — sees only this type, so adding a surface never
means teaching the core about another payload shape.

It is also what sits in `inbox.payload`, which makes the queue self-describing:
a row that replays after a restart carries everything needed to answer it, and
a row that cannot be parsed is a row this version does not know how to answer
rather than one it will half-answer.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from kasa.errors import KasaError


class EventError(KasaError):
    """A queued payload is not an event this build can deliver."""


class InboundEvent(BaseModel):
    """One message from one surface, normalized.

    Frozen, because a row in the inbox is a record of what arrived. Anything
    the agent decides about it belongs beside it, not inside it.
    """

    model_config = ConfigDict(frozen=True)

    #: Which surface this came from: 'slack', 'cli', 'http'. Also the session's
    #: `surface`, since a session belongs to exactly one of them.
    source: str = Field(min_length=1)

    #: The provider's own id for this event, and the dedupe key. A surface with
    #: no such id has to mint one that is stable across the provider's retries —
    #: a fresh ULID per delivery attempt would defeat the whole table.
    external_id: str = Field(min_length=1)

    #: The actor key: one serialized conversation. On Slack this is the thread.
    session_id: str = Field(min_length=1)

    text: str = ""

    #: Ingress persisted a scrubbed form and retained the original only in
    #: process memory for this immediate turn.
    credential_scrubbed: bool = False

    #: The visibility scope anything learned here inherits. `workspace` is the
    #: widest and the default; a DM or a private channel narrows it. Retrieval
    #: filters on this before it ranks, so an adapter that gets it wrong leaks.
    scope: str = "workspace"

    #: The platform's id for whoever spoke, not a display name.
    author: str | None = None

    #: Where a reply goes. Opaque to the core — Slack puts `channel` and the
    #: `thread_ts` to reply in, and only the Slack adapter reads them back.
    channel: str | None = None
    reply_to: str | None = None

    #: What put this in the queue. `message` is somebody speaking; `scheduled`
    #: is a standing task firing (#179), where nobody said anything just now.
    #: The turn needs to know the difference — an answer that opens "as you
    #: asked" when the thread has been quiet since Tuesday reads as a
    #: hallucination.
    #:
    #: Defaulted rather than required so that every payload written before this
    #: field existed still parses. A row in the inbox is a record of what
    #: arrived, and a queue that stops being able to read its own backlog after
    #: an upgrade is a queue that drops messages.
    origin: str = "message"

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> Self:
        try:
            return cls.model_validate_json(payload)
        except ValidationError as exc:
            raise EventError(f"queued payload is not a deliverable event: {exc}") from exc

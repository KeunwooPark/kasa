"""The normalized shape of an inbound message."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kasa.core.events import EventError, InboundEvent


def test_an_event_round_trips_through_the_queue() -> None:
    """The payload column is the whole event, so a replay needs nothing else."""
    event = InboundEvent(
        source="slack",
        external_id="Ev123",
        session_id="slack:T01:C0123:1756890000.123",
        text="what did we decide about deploys?",
        scope="channel:C0123",
        author="U0456",
        channel="C0123",
        reply_to="1756890000.123",
    )
    assert InboundEvent.from_json(event.to_json()) == event


@pytest.mark.parametrize("field", ["source", "external_id", "session_id"])
def test_the_identifying_fields_cannot_be_blank(field: str) -> None:
    """An empty dedupe key dedupes every event against every other one."""
    fields = {"source": "cli", "external_id": "E1", "session_id": "cli:1"}
    with pytest.raises(ValidationError):
        InboundEvent(**{**fields, field: ""})


def test_a_payload_this_build_cannot_read_says_so() -> None:
    """A row written by another version is undeliverable, not half-deliverable."""
    with pytest.raises(EventError, match="not a deliverable event"):
        InboundEvent.from_json('{"source": "slack"}')


def test_scope_defaults_to_the_widest() -> None:
    """Adapters narrow it; nothing here silently guesses a private scope."""
    event = InboundEvent(source="cli", external_id="E1", session_id="cli:1")
    assert event.scope == "workspace"


def test_a_payload_written_before_origin_existed_still_reads() -> None:
    """The inbox holds rows across an upgrade. A field added without a default
    would make everything queued at the moment of the restart undeliverable —
    which is a queue that drops messages, and the reason the field is
    defaulted rather than required (#179)."""
    queued = '{"source": "slack", "external_id": "Ev1", "session_id": "s", "text": "hi"}'

    assert InboundEvent.from_json(queued).origin == "message"

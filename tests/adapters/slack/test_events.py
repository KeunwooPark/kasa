"""What Slack sends, and what Kasa decides it means.

No `slack_bolt` here on purpose: every judgement that can leak a private
conversation is in `events.py`, and none of it needs a socket to test.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import Any

import pytest

from kasa.adapters.slack.events import (
    Accepted,
    Decision,
    Ignored,
    SlackContext,
    normalize,
)

BOT = "U0KASA"
TEAM = "T0TEAM"
HUMAN = "U0HUMAN"


def context(*, allowed: frozenset[str] = frozenset()) -> SlackContext:
    return SlackContext(bot_user_id=BOT, team_id=TEAM, allowed_channels=allowed)


async def never(session_id: str) -> bool:
    return False


async def always(session_id: str) -> bool:
    return True


def dm(text: str = "what did we decide?", ts: str = "1700000000.000100") -> dict[str, Any]:
    return {
        "type": "message",
        "channel_type": "im",
        "channel": "D1",
        "user": HUMAN,
        "text": text,
        "ts": ts,
    }


def in_channel(
    text: str = f"<@{BOT}> what did we decide?",
    ts: str = "1700000000.000100",
    *,
    channel: str = "C1",
    thread: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "type": "message",
        "channel_type": "channel",
        "channel": channel,
        "user": HUMAN,
        "text": text,
        "ts": ts,
    }
    if thread is not None:
        body["thread_ts"] = thread
    return body


def accepted(decision: Decision) -> Accepted:
    assert isinstance(decision, Accepted), decision
    return decision


# -- what gets through -------------------------------------------------------


async def test_a_dm_is_answered_and_scoped_to_the_person() -> None:
    decision = await normalize(dm(), context=context(), known_session=never)

    event = accepted(decision).event
    assert event.session_id == f"slack:{TEAM}:D1:1700000000.000100"
    assert event.scope == f"private:{HUMAN}"
    assert event.author == HUMAN
    assert event.text == "what did we decide?"


async def test_a_mention_in_a_channel_is_answered_and_scoped_to_the_channel() -> None:
    decision = await normalize(in_channel(), context=context(), known_session=never)

    event = accepted(decision).event
    assert event.scope == "channel:C1"
    assert event.text == "what did we decide?", "Kasa's own mention is addressing, not content"


async def test_an_app_mention_carries_no_channel_type_and_is_still_a_channel() -> None:
    """A minimally-scoped install gets `app_mention` and nothing else, and that
    event has no `channel_type` to read."""
    body = {
        "type": "app_mention",
        "channel": "C1",
        "user": HUMAN,
        "text": f"<@{BOT}> hello",
        "ts": "1700000000.000100",
    }

    event = accepted(await normalize(body, context=context(), known_session=never)).event
    assert event.scope == "channel:C1"


async def test_a_thread_reply_needs_no_second_mention() -> None:
    """Being spoken to once starts a conversation; making people re-address
    every message in a thread is how a bot becomes tedious."""
    body = in_channel("and what about staging?", "1700000000.000200", thread="1700000000.000100")

    ignored = await normalize(body, context=context(), known_session=never)
    accepted_again = await normalize(body, context=context(), known_session=always)

    assert isinstance(ignored, Ignored)
    assert accepted(accepted_again).event.session_id == f"slack:{TEAM}:C1:1700000000.000100"


async def test_a_reply_goes_to_the_thread_the_question_was_asked_in() -> None:
    top = accepted(await normalize(in_channel(), context=context(), known_session=never)).event
    reply = accepted(
        await normalize(
            in_channel(f"<@{BOT}> more", "1700000000.000200", thread="1700000000.000100"),
            context=context(),
            known_session=never,
        )
    ).event

    assert top.reply_to == "1700000000.000100", "a top-level message starts a thread"
    assert reply.reply_to == "1700000000.000100"
    assert reply.session_id == top.session_id


async def test_somebody_elses_mention_stays_in_the_text() -> None:
    body = in_channel(f"<@{BOT}> ask <@U0OTHER|jane> about deploys")

    assert (
        accepted(await normalize(body, context=context(), known_session=never)).event.text
        == "ask <@U0OTHER|jane> about deploys"
    )


async def test_the_shape_of_a_message_survives() -> None:
    """Stripping the mention used to cost every line break in the message —
    `str.split()` with no argument splits on newlines too. Pasted code, stack
    traces, numbered lists and multi-paragraph questions all arrived as one
    run-on line."""
    sent = f"<@{BOT}> please look at this:\n\n```\ndef f():\n    return 1\n```\n\nthanks"

    event = accepted(
        await normalize(in_channel(sent), context=context(), known_session=never)
    ).event

    assert event.text == "please look at this:\n\n```\ndef f():\n    return 1\n```\n\nthanks"


async def test_the_gap_the_mention_leaves_is_still_tidied() -> None:
    """One space where a mention was between two words, none where it was at
    an edge — and nothing else touched."""
    cases = {
        f"<@{BOT}> hello": "hello",
        f"hello <@{BOT}>": "hello",
        f"hey <@{BOT}> what's up": "hey what's up",
        f"<@{BOT}>\nhello": "hello",
        f"a  b <@{BOT}> c": "a  b c",
    }

    for sent, expected in cases.items():
        event = accepted(
            await normalize(in_channel(sent), context=context(), known_session=never)
        ).event
        assert event.text == expected, sent


# -- what does not ------------------------------------------------------------


async def test_a_channel_message_that_is_not_addressed_to_kasa_is_ignored() -> None:
    decision = await normalize(
        in_channel("what did we decide?"), context=context(), known_session=never
    )

    assert isinstance(decision, Ignored)
    assert decision.reason == "not addressed to Kasa"


@pytest.mark.parametrize("subtype", ["message_changed", "message_deleted", "channel_join"])
async def test_a_subtype_is_ignored(subtype: str) -> None:
    """Editing a message must not read as sending a new one. #25 is where
    `message_changed` and `message_deleted` become something."""
    decision = await normalize(dm() | {"subtype": subtype}, context=context(), known_session=never)

    assert isinstance(decision, Ignored)
    assert subtype in decision.reason


async def test_kasas_own_message_is_ignored() -> None:
    decision = await normalize(dm() | {"user": BOT}, context=context(), known_session=always)

    assert isinstance(decision, Ignored)
    assert decision.reason == "posted by Kasa"


async def test_another_bot_is_ignored() -> None:
    """Two bots in one channel is how a loop starts."""
    decision = await normalize(
        in_channel(f"<@{BOT}> status?") | {"bot_id": "B1"},
        context=context(),
        known_session=always,
    )

    assert isinstance(decision, Ignored)
    assert decision.reason == "posted by a bot"


async def test_a_channel_off_the_allowlist_is_ignored_even_when_addressed() -> None:
    decision = await normalize(
        in_channel(channel="CSECRET"),
        context=context(allowed=frozenset({"C1"})),
        known_session=always,
    )

    assert isinstance(decision, Ignored)
    assert decision.reason == "channel CSECRET is not on the allowlist"


async def test_an_empty_allowlist_allows_the_channels_kasa_was_invited_to() -> None:
    """Inviting a bot to a channel is already a deliberate act by a person."""
    assert isinstance(
        await normalize(in_channel(), context=context(), known_session=never), Accepted
    )


async def test_the_channel_allowlist_does_not_govern_dms() -> None:
    decision = await normalize(
        dm(), context=context(allowed=frozenset({"C1"})), known_session=never
    )

    assert isinstance(decision, Accepted)


# -- dedupe -------------------------------------------------------------------


async def test_the_same_message_has_one_id_however_it_arrives() -> None:
    """An install with both subscriptions gets a mention twice, under two
    different `event_id`s. Keying on the message is what makes that harmless."""
    as_message = accepted(
        await normalize(in_channel(), context=context(), known_session=never)
    ).event
    as_mention = accepted(
        await normalize(
            {
                "type": "app_mention",
                "channel": "C1",
                "user": HUMAN,
                "text": f"<@{BOT}> what did we decide?",
                "ts": "1700000000.000100",
            },
            context=context(),
            known_session=never,
        )
    ).event

    assert as_message.external_id == as_mention.external_id == f"slack:{TEAM}:C1:1700000000.000100"


# -- scope --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (dm(), f"private:{HUMAN}"),
        (in_channel(), "channel:C1"),
        (in_channel() | {"channel_type": "group"}, "channel:C1"),
        (in_channel() | {"channel_type": "mpim"}, "channel:C1"),
    ],
)
async def test_nothing_from_slack_is_scoped_workspace(body: dict[str, Any], expected: str) -> None:
    """`workspace` is the widest scope there is. Widening one is a decision for
    #24 with a person in it, not a default every public channel picks up."""
    event = accepted(await normalize(body, context=context(), known_session=always)).event

    assert event.scope == expected != "workspace"


# -- and the import costs nothing either --------------------------------------


def test_the_judgements_import_without_the_slack_extra() -> None:
    """The docstring above claims no `slack_bolt`, and that was true of every
    line in this file except the first: importing `events` runs the package
    `__init__`, which imported the adapter, which imports `slack_bolt`. On a
    `uv sync --dev` that is a collection *error*, not a skip, and pytest aborts
    the whole run — zero of 600-odd tests, which reads a lot like a pass.
    """
    program = textwrap.dedent(
        """
        import sys

        class Absent:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "slack_bolt":
                    raise ModuleNotFoundError("No module named 'slack_bolt'")
                return None

        sys.meta_path.insert(0, Absent())

        import kasa.adapters.slack as package
        from kasa.adapters.slack.events import normalize, scope_for

        assert "slack_bolt" not in sys.modules, "something imported it anyway"
        assert package.scope_for is scope_for
        """
    )

    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr


def test_asking_for_the_adapter_is_what_needs_the_extra() -> None:
    """Lazy, not gone. The name still resolves where it always did."""
    program = textwrap.dedent(
        """
        import sys

        class Absent:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "slack_bolt":
                    raise ModuleNotFoundError("No module named 'slack_bolt'")
                return None

        sys.meta_path.insert(0, Absent())

        import kasa.adapters.slack as package

        try:
            package.SlackAdapter
        except ModuleNotFoundError as exc:
            assert "slack_bolt" in str(exc)
        else:
            raise AssertionError("the adapter resolved without its extra")

        try:
            package.NotAThing
        except AttributeError:
            pass
        else:
            raise AssertionError("an unknown attribute resolved")
        """
    )

    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr

from __future__ import annotations

import json
import sqlite3

import pytest

from tests.e2e.conftest import KasaRig
from tests.e2e.test_shutdown import FakeSlack, daemon, eventually, fake_slack, stop  # noqa: F401


def inbox_count(rig: KasaRig) -> int:
    if not rig.database.exists():
        return 0
    connection = sqlite3.connect(rig.database)
    try:
        return int(connection.execute("SELECT count(*) FROM inbox").fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        connection.close()


def test_slack_ingress_routing_deduplication_and_replies(
    kasa_rig: KasaRig, request: pytest.FixtureRequest
) -> None:
    slack_server: FakeSlack = request.getfixturevalue("fake_slack")
    process = daemon(kasa_rig, slack_server)
    assert slack_server.connected.wait(timeout=5)
    slack_server.ack_observer = lambda: inbox_count(kasa_rig)

    slack_server.event(
        event_id="Ev-mention",
        event_type="message",
        channel="C_DEPLOY",
        timestamp="10.001",
        text="<@UBOT> channel question",
    )
    eventually(lambda: len(slack_server.posts), lambda count: count == 1)

    # The same Slack message may arrive through both subscriptions. Its
    # message timestamp, not the delivery id, is the durable dedupe key.
    slack_server.event(
        event_id="Ev-duplicate",
        event_type="message",
        channel="C_DEPLOY",
        timestamp="10.001",
        text="<@UBOT> channel question",
    )
    slack_server.event(
        event_id="Ev-ignored",
        channel="C_DEPLOY",
        timestamp="10.002",
        text="chatter in a new thread",
    )
    slack_server.event(
        event_id="Ev-dm",
        channel="D_PRIVATE",
        timestamp="20.001",
        text="private question",
    )
    eventually(lambda: len(slack_server.posts), lambda count: count == 2)

    # Once Kasa belongs to a channel thread, a reply routes to the same
    # session without needing another mention.
    slack_server.event(
        event_id="Ev-thread",
        channel="C_DEPLOY",
        timestamp="10.003",
        thread_ts="10.001",
        text="follow-up in the thread",
    )
    eventually(lambda: len(slack_server.posts), lambda count: count == 3)
    eventually(lambda: len(slack_server.acknowledgements), lambda count: count == 5)
    stdout, stderr = stop(process)
    assert process.returncode == 0, (stdout, stderr)

    # Seeing every ack means Bolt returned to Slack. At that point each
    # accepted event is already a committed inbox row; ignored events are not.
    connection = sqlite3.connect(kasa_rig.database)
    try:
        inbox = connection.execute(
            "SELECT external_id, payload, state FROM inbox ORDER BY id"
        ).fetchall()
        sessions = connection.execute("SELECT id, scope FROM sessions ORDER BY id").fetchall()
    finally:
        connection.close()

    assert len(inbox) == 3
    at_ack = dict(slack_server.ack_snapshots)
    assert at_ack["envelope-Ev-mention"] >= 1
    assert at_ack["envelope-Ev-dm"] >= 2
    assert at_ack["envelope-Ev-thread"] >= 3
    assert {row[2] for row in inbox} == {"done"}
    events = [json.loads(row[1]) for row in inbox]
    assert [(event["session_id"], event["scope"]) for event in events] == [
        ("slack:T_E2E:C_DEPLOY:10.001", "channel:C_DEPLOY"),
        ("slack:T_E2E:D_PRIVATE:20.001", "private:U_USER"),
        ("slack:T_E2E:C_DEPLOY:10.001", "channel:C_DEPLOY"),
    ]
    assert sessions == [
        ("slack:T_E2E:C_DEPLOY:10.001", "channel:C_DEPLOY"),
        ("slack:T_E2E:D_PRIVATE:20.001", "private:U_USER"),
    ]
    assert [(post["channel"], post["thread_ts"]) for post in slack_server.posts] == [
        ("C_DEPLOY", "10.001"),
        ("D_PRIVATE", "20.001"),
        ("C_DEPLOY", "10.001"),
    ]
    assert len(kasa_rig.server.requests) == 3

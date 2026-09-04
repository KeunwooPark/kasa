from __future__ import annotations

import json
import sqlite3

from tests.e2e.conftest import KasaRig


def rows(rig: KasaRig, query: str) -> list[sqlite3.Row]:
    connection = sqlite3.connect(rig.database)
    connection.row_factory = sqlite3.Row
    try:
        return connection.execute(query).fetchall()
    finally:
        connection.close()


def test_a_user_can_complete_a_repl_turn(kasa_rig: KasaRig) -> None:
    result = kasa_rig.run("Hello from QA.\n/session\n/quit\n")

    assert result.returncode == 0, result.stderr
    assert "E2E reply: Hello from QA." in result.stdout
    assert "2 messages" in result.stdout
    assert [row["role"] for row in rows(kasa_rig, "SELECT role FROM messages ORDER BY seq")] == [
        "user",
        "assistant",
    ]


def test_a_tool_call_crosses_the_provider_and_cli_boundaries(kasa_rig: KasaRig) -> None:
    result = kasa_rig.run("Use the clock tool.\n/quit\n")

    assert result.returncode == 0, result.stderr
    assert "Clock checked." in result.stdout
    assert "1 tool call(s), 2 iteration(s)" in result.stdout
    assert len(kasa_rig.server.requests) == 2
    second_messages = kasa_rig.server.requests[1]["messages"]
    tool_message = next(message for message in second_messages if message["role"] == "tool")
    assert tool_message["tool_call_id"] == "call_e2e_clock"
    assert "T" in tool_message["content"]  # the real current_time tool ran

    stored = rows(kasa_rig, "SELECT role, content FROM messages ORDER BY seq")
    assert [row["role"] for row in stored] == ["user", "assistant", "user", "assistant"]
    tool_result = json.loads(stored[2]["content"])[0]
    assert tool_result["tool_use_id"] == "call_e2e_clock"


def test_conversations_survive_a_process_restart(kasa_rig: KasaRig) -> None:
    first = kasa_rig.run("First process.\n/quit\n")
    second = kasa_rig.run("Second process.\n/quit\n")

    assert first.returncode == second.returncode == 0
    assert "E2E reply: First process." in first.stdout
    assert "E2E reply: Second process." in second.stdout
    assert rows(kasa_rig, "SELECT count(*) AS count FROM sessions")[0]["count"] == 2
    assert rows(kasa_rig, "SELECT count(*) AS count FROM messages")[0]["count"] == 4
    assert rows(kasa_rig, "SELECT count(*) AS count FROM llm_calls")[0]["count"] == 2

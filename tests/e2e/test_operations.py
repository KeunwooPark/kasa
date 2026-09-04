from __future__ import annotations

import json
import sqlite3

from tests.e2e.conftest import KasaRig


def initialize(rig: KasaRig) -> None:
    result = rig.command("db", "migrate")
    assert result.returncode == 0, result.stderr


def test_config_database_and_doctor_are_scriptable(kasa_rig: KasaRig) -> None:
    path = kasa_rig.command("db", "path")
    assert path.returncode == 0
    assert path.stdout == f"{kasa_rig.database}\n"
    assert path.stderr == ""

    config = kasa_rig.command("config")
    assert config.returncode == 0
    parsed = json.loads(config.stdout)
    assert parsed["store"]["path"] == str(kasa_rig.database)
    assert "not-a-real-secret" not in config.stdout + config.stderr
    assert str(kasa_rig.config) in config.stderr

    migrate = kasa_rig.command("db", "migrate")
    assert migrate.returncode == 0
    assert "up to date" in migrate.stdout
    assert kasa_rig.database.is_file()

    doctor = kasa_rig.command("doctor")
    assert doctor.returncode == 0, doctor.stderr
    assert "e2e-model" in doctor.stdout
    assert "database" in doctor.stdout


def test_operational_commands_report_actionable_failures(kasa_rig: KasaRig) -> None:
    without_key = kasa_rig.env.copy()
    without_key.pop("KASA_E2E_API_KEY")
    doctor = kasa_rig.command("doctor", env=without_key)
    assert doctor.returncode == 1
    assert "KASA_E2E_API_KEY is not set" in doctor.stdout
    assert "1 check(s) failed" in doctor.stdout

    bad_config = kasa_rig.config.with_name("bad.toml")
    bad_config.write_text("[store]\nunknown = true\n")
    config = kasa_rig.command("config", config=bad_config)
    assert config.returncode == 1
    assert config.stdout == ""
    assert "unknown" in config.stderr


def test_cost_output_reflects_a_real_subprocess_turn(kasa_rig: KasaRig) -> None:
    empty = kasa_rig.command("cost")
    assert empty.returncode == 0
    assert "no calls recorded yet" in empty.stdout

    turn = kasa_rig.run("Measure this.\n/quit\n")
    assert turn.returncode == 0, turn.stderr
    cost = kasa_rig.command("cost")

    assert cost.returncode == 0
    assert "e2e-model" in cost.stdout
    assert "11" in cost.stdout
    assert "3" in cost.stdout


def test_inbox_and_job_commands_expose_and_retry_failures(kasa_rig: KasaRig) -> None:
    initialize(kasa_rig)
    connection = sqlite3.connect(kasa_rig.database)
    try:
        connection.execute(
            """INSERT INTO inbox
               (source, external_id, payload, received_at, state, attempts, last_error)
               VALUES ('slack', 'Ev-dead', '{}', '2026-01-01T00:00:00Z', 'failed', 5, ?)""",
            ("provider unavailable",),
        )
        connection.execute(
            """INSERT INTO jobs
               (id, kind, run_after, state, attempts, last_error, created_at)
               VALUES ('job-dead', 'reindex', '2026-01-01T00:00:00Z', 'failed', 5, ?,
                       '2026-01-01T00:00:00Z')""",
            ("memory clone missing",),
        )
        connection.commit()
    finally:
        connection.close()

    inbox = kasa_rig.command("inbox", "status")
    assert inbox.returncode == 0
    assert "failed" in inbox.stdout and "1" in inbox.stdout
    assert "provider unavailable" in inbox.stderr

    retry = kasa_rig.command("inbox", "retry")
    assert retry.returncode == 0
    assert "requeued 1 event(s)" in retry.stdout
    assert "no dead letters" in kasa_rig.command("inbox", "retry").stdout

    jobs = kasa_rig.command("job", "list")
    assert jobs.returncode == 0
    assert "reindex" in jobs.stdout
    assert "memory clone missing" in jobs.stderr


def test_database_commands_reject_a_non_database_file(kasa_rig: KasaRig) -> None:
    kasa_rig.database.write_text("this is not sqlite")

    result = kasa_rig.command("db", "migrate")

    assert result.returncode == 1
    assert result.stdout == ""
    assert "not a database" in result.stderr.lower()

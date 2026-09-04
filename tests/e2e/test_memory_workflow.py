from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from tests.e2e.conftest import KasaRig

MEMORY_ID = "mem_01K8XQ4W2N7B6VJ3ZC9F0RTKME"
MEMORY = f"""---
id: {MEMORY_ID}
type: fact
title: Deployment launch checklist ownership
tags: [deployment, ownership]
visibility: workspace
created: 2026-09-03T10:12:00Z
updated: 2026-09-03T10:12:00Z
confidence: 0.9
salience: 0.8
pinned: false
source_refs: [qa://seed]
supersedes: []
---

Aster Quinn owns the deployment launch checklist and approves every production launch.
"""


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, timeout=10, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def local_remote(tmp_path: Path) -> Path:
    remote = tmp_path / "memory.git"
    source = tmp_path / "source"
    git("init", "--bare", "--initial-branch", "main", str(remote))
    git("init", "--initial-branch", "main", str(source))
    git("config", "user.name", "E2E QA", cwd=source)
    git("config", "user.email", "e2e@example.invalid", cwd=source)
    (source / "README.md").write_text("# Local memory remote\n")
    git("add", "README.md", cwd=source)
    git("commit", "-m", "seed remote", cwd=source)
    git("remote", "add", "origin", str(remote), cwd=source)
    git("push", "-u", "origin", "main", cwd=source)
    return remote


def bootstrap_with_init(rig: KasaRig, tmp_path: Path) -> Path:
    remote = local_remote(tmp_path)
    clone = tmp_path / "memory-clone"
    answers = (
        f"{remote}\n"  # repository
        "\n"  # default token environment variable
        f"{clone}\n"
        "\n"  # keep the inferred custom preset
        "\n"  # keep the configured OpenAI wire format
        "\n"  # keep local fake-provider URL
        "\n"  # keep key environment variable
        "\n"  # keep model (from discovery when available)
        "n\n"  # no separate background or embedding models
        "n\n"  # no Slack
        "y\n"  # push the bootstrap commit
    )
    initialized = rig.command("init", input=answers)
    assert initialized.returncode == 0, (initialized.stdout, initialized.stderr)
    assert "Bootstrapped" in initialized.stdout
    assert clone.joinpath("memory", ".kasa", "schema.md").is_file()
    git("config", "user.name", "E2E QA", cwd=clone)
    git("config", "user.email", "e2e@example.invalid", cwd=clone)
    return clone


def test_memory_is_indexed_retrieved_and_recorded_through_cli_processes(
    kasa_rig: KasaRig, tmp_path: Path
) -> None:
    clone = bootstrap_with_init(kasa_rig, tmp_path)
    memory_path = clone / "memory" / "facts" / "deployment-owner.md"
    memory_path.write_text(MEMORY)
    git("add", str(memory_path.relative_to(clone)), cwd=clone)
    git("commit", "-m", "memory: seed deployment owner", cwd=clone)

    indexed = kasa_rig.command("reindex")
    assert indexed.returncode == 0, indexed.stderr
    assert "1 file(s) indexed" in indexed.stdout
    assert "chunk(s)" in indexed.stdout
    assert "manifest rebuilt: 1 memories (1 added)" in indexed.stdout
    manifest = json.loads((clone / "memory" / ".kasa" / "manifest.json").read_text())
    assert manifest["memories"][MEMORY_ID]["path"] == "memory/facts/deployment-owner.md"

    answer = kasa_rig.run("Who owns the deployment launch checklist?\n/quit\n")
    assert answer.returncode == 0, answer.stderr
    assert "E2E reply: Who owns the deployment launch checklist?" in answer.stdout
    assert any("Aster Quinn" in json.dumps(request) for request in kasa_rig.server.requests)

    connection = sqlite3.connect(kasa_rig.database)
    try:
        hits = connection.execute("SELECT memory_id FROM memory_hits ORDER BY id").fetchall()
        roles = connection.execute("SELECT role FROM messages ORDER BY seq").fetchall()
    finally:
        connection.close()
    assert hits == [(MEMORY_ID,)]
    assert roles == [("user",), ("assistant",)]
    assert git("status", "--porcelain", cwd=clone) == ""


def test_an_unavailable_clone_fails_without_network_access(
    kasa_rig: KasaRig, tmp_path: Path
) -> None:
    clone = bootstrap_with_init(kasa_rig, tmp_path)
    missing = tmp_path / "clone-that-is-not-there"
    broken = kasa_rig.config.with_name("missing-clone.toml")
    broken.write_text(kasa_rig.config.read_text().replace(str(clone), str(missing)))

    result = kasa_rig.command("reindex", config=broken)

    assert result.returncode == 1
    assert result.stdout == ""
    assert str(missing) in result.stderr
    assert "kasa init" in result.stderr

"""The CLI commands themselves, invoked the way a shell invokes them.

Most of the surface is covered by the modules underneath. What is not, and what
this file is for, is the *order* the commands do things in — a command that
writes before it validates fails in a way no unit test sees.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import logging
import os
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kasa import __version__
from kasa.cli import app
from kasa.core.backoff import Backoff
from kasa.core.events import InboundEvent
from kasa.core.inbox import Inbox
from kasa.llm.cost import CallRecord
from kasa.llm.types import Usage
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.manifest import Manifest
from kasa.store import Store

runner = CliRunner()


def a_memory(root: Path, title: str = "Jane owns the deploy pipeline") -> MemoryDoc:
    doc = MemoryDoc.new(type="person", title=title, body="Jane owns the deploy pipeline.")
    target = root / doc.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    return doc


@pytest.fixture
def rig(tmp_path: Path) -> tuple[Path, Path]:
    """A config file and a memory clone, with one memory and no skeleton yet."""
    clone = tmp_path / "ltm"
    GitRepo.init(clone, branch="main")
    a_memory(clone)

    config = tmp_path / "config.toml"
    config.write_text(
        f'[ltm]\nrepo = "{clone}"\nclone_path = "{clone}"\nbranch = "main"\n\n'
        f'[store]\npath = "{tmp_path / "kasa.db"}"\n'
    )
    return config, clone


def chunks(db: Path) -> int:
    if not db.exists():
        return 0
    conn = sqlite3.connect(db)
    try:
        return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])
    finally:
        conn.close()


def test_reindex_writes_nothing_when_the_clone_has_no_skeleton(
    rig: tuple[Path, Path], tmp_path: Path
) -> None:
    """#62. The manifest half cannot run, so the index half must not either.

    It used to: the index was rebuilt, then `MemoryStore.open` raised, and the
    command exited 1 having reported none of the work it had already done.
    """
    config, _ = rig

    result = runner.invoke(app, ["reindex", "--config", str(config)])

    assert result.exit_code == 1
    assert "no memory skeleton" in result.output
    assert chunks(tmp_path / "kasa.db") == 0, "a failed reindex left half its work behind"


def test_reindex_rebuilds_both_halves_once_the_repo_is_bootstrapped(
    rig: tuple[Path, Path], tmp_path: Path
) -> None:
    config, clone = rig
    bootstrap(clone)
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit("memory: seed")

    result = runner.invoke(app, ["reindex", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "1 file(s) indexed" in result.output
    assert "manifest already describes all 1 memories" in result.output
    assert chunks(tmp_path / "kasa.db") > 0


def test_audit_lists_every_memory_by_scope_even_when_manifest_is_stale(
    rig: tuple[Path, Path],
) -> None:
    config, clone = rig
    bootstrap(clone)
    Manifest.rebuild(clone)[0].save(clone)
    private = MemoryDoc.new(
        type="fact", title="Salary review", body="Private outcome.", visibility="private:U01"
    )
    target = clone / private.suggested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(private.render())

    result = runner.invoke(app, ["audit", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "workspace" in result.output
    assert "private:U01" in result.output
    assert str(private.id) in result.output
    assert "2 memory(s) across 2 scope(s)" in result.output


def test_audit_reports_unreadable_memories(rig: tuple[Path, Path]) -> None:
    config, clone = rig
    bootstrap(clone)
    path = broken(clone, "unscoped.md")

    result = runner.invoke(app, ["audit", "--config", str(config)])

    assert result.exit_code == 1
    assert path in result.stderr
    assert "no YAML frontmatter" in result.stderr


# -- output a shell can use (#68) --------------------------------------------
#
# rich falls back to 80 columns when stdout is not a terminal and hard-wraps
# there, so every one of these commands used to put a newline inside the value
# it exists to print. The paths below are deliberately longer than 80
# characters; that is the whole test.


@pytest.fixture
def deep(tmp_path: Path) -> Path:
    """A path comfortably past rich's 80-column fallback."""
    root = tmp_path / ("a" * 40) / ("b" * 40)
    root.mkdir(parents=True)
    return root


def config_for(db: Path) -> Path:
    path = db.parent / "config.toml"
    path.write_text(f'[store]\npath = "{db}"\n')
    return path


def test_db_path_prints_something_a_shell_can_substitute(deep: Path) -> None:
    db = deep / "kasa.db"
    result = runner.invoke(app, ["db", "path", "--config", str(config_for(db))])

    assert result.exit_code == 0, result.output
    assert len(str(db)) > 80, "the fixture has to be long enough to have been wrapped"
    assert result.stdout == f"{db}\n", "one line, unmodified — this is $(kasa db path)"


def test_a_path_containing_brackets_is_not_read_as_markup(tmp_path: Path) -> None:
    """rich deletes `[dim]`-shaped text. A directory is allowed to be called that."""
    root = tmp_path / "[dim]"
    root.mkdir()
    db = root / "kasa.db"

    result = runner.invoke(app, ["db", "path", "--config", str(config_for(db))])

    assert result.stdout == f"{db}\n"


def test_version_is_one_bare_line() -> None:
    result = runner.invoke(app, ["version"])
    assert result.stdout == f"{__version__}\n"


def test_config_puts_its_header_on_stderr_so_the_json_can_be_piped(deep: Path) -> None:
    """The path says where the JSON came from: a comment on the output, not part
    of it. On stdout it was the first thing `kasa config | jq` choked on."""
    config = config_for(deep / "kasa.db")

    result = runner.invoke(app, ["config", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["store"]["path"] == str(deep / "kasa.db")
    assert str(config) in result.stderr


def test_an_error_about_a_long_path_stays_on_one_line(deep: Path) -> None:
    """A config error names the file. Wrapped, the name is unusable — and it is
    the one thing the reader has to act on."""
    broken = deep / "config.toml"
    broken.write_text("[store\npath = ")

    result = runner.invoke(app, ["config", "--config", str(broken)])

    assert result.exit_code == 1
    assert len(str(broken)) > 80
    assert str(broken) in result.stderr, "the path was split across two lines"


# -- one line per broken file (#77) -------------------------------------------


def broken(clone: Path, name: str = "broken.md") -> str:
    (clone / "memory" / "facts").mkdir(parents=True, exist_ok=True)
    (clone / "memory" / "facts" / name).write_text("no frontmatter here at all")
    return f"memory/facts/{name}"


def test_an_unreadable_file_is_named_once_with_its_reason(rig: tuple[Path, Path]) -> None:
    """#77. The index reported bare paths and the manifest reported reasons, so
    a file both halves refused was named twice — once uselessly."""
    config, clone = rig
    bootstrap(clone)
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit("memory: seed")
    path = broken(clone)

    result = runner.invoke(app, ["reindex", "--config", str(config)])

    assert result.exit_code == 0, result.output
    named = [line for line in result.output.splitlines() if path in line]
    assert len(named) == 1, f"one line per file, got:\n{result.output}"
    assert "no YAML frontmatter" in named[0]


def test_the_log_record_names_the_file_once_too(
    rig: tuple[Path, Path], caplog: pytest.LogCaptureFixture
) -> None:
    """`str(exc)` already carries the source, so `"index: %s: %s", path, exc`
    printed it twice — the defect #70 fixed one line away from this one.

    Asserted on the record rather than on stdout: `reindex` now configures
    logging to keep these out of its own report at default verbosity, and
    pytest owns the root logger, so what reaches stdout here is pytest's
    choice rather than the command's. The quiet default is checked by hand.
    """
    config, clone = rig
    bootstrap(clone)
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit("memory: seed")
    path = broken(clone)

    with caplog.at_level(logging.WARNING, logger="kasa.memory.index"):
        runner.invoke(app, ["reindex", "--config", str(config)])

    message = next(r.getMessage() for r in caplog.records if r.name == "kasa.memory.index")
    assert message.count(path) == 1, message
    assert "no YAML frontmatter" in message


def test_the_cost_table_does_not_truncate_the_model_name(deep: Path) -> None:
    """#80. The model column is the row's identity, and rich's 80-column
    fallback put an ellipsis in it as soon as the output was piped — so two
    models from one provider became the same row."""
    db = deep / "kasa.db"
    config = config_for(db)
    model = "accounts/fireworks/models/kimi-k3-instruct-0905-preview"

    async def seed() -> None:
        async with await Store.open(db) as store:
            await store.record_call(
                CallRecord(
                    role="chat",
                    provider="openai",
                    model=model,
                    usage=Usage(input_tokens=10, output_tokens=5),
                    latency_ms=1,
                    cost_usd=None,
                    tag=None,
                    ok=True,
                )
            )

    # Not an async test: `runner.invoke` runs a command that calls
    # `asyncio.run`, which cannot be nested inside a running loop.
    asyncio.run(seed())

    result = runner.invoke(app, ["cost", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "\u2026" not in result.output, result.output
    # Folded, like `doctor`'s detail column: the name survives across two
    # lines of the same cell, so the column is read back column-wise.
    cells = [
        line.split("\u2502")[2].strip()
        for line in result.output.splitlines()
        if line.count("\u2502") > 2
    ]
    assert "".join(cells) == model, result.output


def test_inbox_status_reports_a_state_with_no_rows_as_zero(tmp_path: Path) -> None:
    """A missing line reads as "no idea"; a zero reads as "none". The states are
    printed in a fixed order for that reason."""
    db = tmp_path / "kasa.db"
    config = config_for(db)

    async def seed() -> None:
        async with await Store.open(db) as store:
            await Inbox(store).enqueue(
                InboundEvent(source="slack", external_id="Ev1", session_id="slack:T:C:1")
            )

    asyncio.run(seed())

    result = runner.invoke(app, ["inbox", "status", "--config", str(config)])

    assert result.exit_code == 0, result.output
    counts = {
        cells[1].strip(): cells[2].strip()
        for line in result.output.splitlines()
        if len(cells := line.split("\u2502")) > 3
    }
    assert counts == {"pending": "1", "leased": "0", "done": "0", "failed": "0"}


def test_inbox_retry_puts_a_dead_letter_back(tmp_path: Path) -> None:
    """Dead-lettering is a pause for a human. This is the human."""
    db = tmp_path / "kasa.db"
    config = config_for(db)

    async def seed() -> None:
        async with await Store.open(db) as store:
            inbox = Inbox(store, backoff=Backoff(max_attempts=1, base=0.0, cap=0.0))
            await inbox.enqueue(
                InboundEvent(source="slack", external_id="Ev1", session_id="slack:T:C:1")
            )
            await inbox.fail((await inbox.lease())[0], "the model was down all afternoon")

    asyncio.run(seed())

    listed = runner.invoke(app, ["inbox", "status", "--config", str(config)])
    assert "the model was down all afternoon" in listed.output, listed.output

    result = runner.invoke(app, ["inbox", "retry", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "requeued 1 event(s)" in result.output

    assert (
        "no dead letters" in runner.invoke(app, ["inbox", "retry", "--config", str(config)]).output
    )


def test_run_slack_without_tokens_says_so(tmp_path: Path) -> None:
    """It fails here, before the store is opened, rather than inside a socket
    library minutes into a deploy."""
    config = config_for(tmp_path / "kasa.db")

    result = runner.invoke(app, ["run", "--slack", "--config", str(config)])

    assert result.exit_code == 1, result.output
    assert "no Slack tokens configured" in result.output


def test_job_run_reports_a_job_that_failed(tmp_path: Path) -> None:
    """`kasa job run` exits non-zero with the reason, so it is usable in a
    script and readable in a terminal."""
    config = config_for(tmp_path / "kasa.db")

    result = runner.invoke(app, ["job", "run", "promote", "--config", str(config)])

    assert result.exit_code == 1, result.output
    assert "no job named 'promote'" in result.output


def test_job_run_names_the_job_it_ran_past_a_backlog(
    rig: tuple[Path, Path], tmp_path: Path
) -> None:
    """The reported row has to be the one the command queued, not whichever of
    that kind happened to be oldest. With rows already due, `kasa job run`
    printed `reindex pending: None` and exited 1 without running anything."""
    config, clone = rig
    bootstrap(clone)
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit("memory: seed")

    async def backlog() -> None:
        async with await Store.open(tmp_path / "kasa.db") as store:
            for n in range(5):
                await store.enqueue_job(
                    job_id=f"older-{n}",
                    kind="reindex",
                    payload=None,
                    run_after="2020-01-01T00:00:00.000+00:00",
                )

    asyncio.run(backlog())

    result = runner.invoke(app, ["job", "run", "reindex", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "reindex finished" in result.output

    conn = sqlite3.connect(tmp_path / "kasa.db")
    try:
        backlog_states = [
            row[0] for row in conn.execute("SELECT state FROM jobs WHERE id LIKE 'older-%'")
        ]
    finally:
        conn.close()
    # "Run one job now, in this process." Reaching the queued row by draining
    # the whole due backlog of its kind ran five other people's jobs, each of
    # which calls a frontier model, from a command that names one (#127).
    assert backlog_states == ["pending"] * 5, "the backlog is the daemon's, not this command's"


def test_job_run_does_not_call_a_reindex_that_could_not_lock_finished(
    rig: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The symptom, at the level somebody actually meets it.

    With `flock` failing the way an NFS mount with no lock daemon fails, this
    printed `reindex finished` and exited 0 with an empty index — because the
    lease could not tell "cannot lock" from "somebody else has it", and #116
    reads the latter as work that is already being done.
    """
    config, clone = rig
    bootstrap(clone)
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit("memory: seed")
    real_flock = fcntl.flock

    def enolck(fd: int, operation: int) -> None:
        if operation & fcntl.LOCK_EX:
            raise OSError(errno.ENOLCK, os.strerror(errno.ENOLCK))
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", enolck)

    result = runner.invoke(app, ["job", "run", "reindex", "--config", str(config)])

    assert result.exit_code == 1, result.output
    assert "No locks available" in result.output
    assert chunks(tmp_path / "kasa.db") == 0


def test_job_run_says_a_job_never_ran_rather_than_None(rig: tuple[Path, Path]) -> None:
    """A row with no recorded error did not fail. `None` names no reason, and
    "pending" reads as though the command did nothing at all."""
    config, _ = rig

    result = runner.invoke(app, ["job", "run", "reindex", "--config", str(config)])

    assert result.exit_code == 1, result.output
    assert "reindex retrying" in result.output
    assert "no memory skeleton" in result.output
    assert "None" not in result.output


def test_job_list_names_what_this_build_knows_when_nothing_is_queued(
    rig: tuple[Path, Path],
) -> None:
    config, _ = rig

    result = runner.invoke(app, ["job", "list", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "reindex" in result.output


def test_job_retry_puts_a_dead_letter_back(tmp_path: Path) -> None:
    db = tmp_path / "kasa.db"
    config = config_for(db)

    async def seed() -> None:
        async with await Store.open(db) as store:
            await store.enqueue_job(
                job_id="j1", kind="reindex", payload=None, run_after="2020-01-01T00:00:00.000+00:00"
            )
            await store.fail_job("j1", error="the clone was gone")

    asyncio.run(seed())

    listed = runner.invoke(app, ["job", "list", "--config", str(config)])
    assert "the clone was gone" in listed.output, listed.output

    result = runner.invoke(app, ["job", "retry", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert "requeued 1 job(s)" in result.output

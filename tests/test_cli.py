"""The CLI commands themselves, invoked the way a shell invokes them.

Most of the surface is covered by the modules underneath. What is not, and what
this file is for, is the *order* the commands do things in — a command that
writes before it validates fails in a way no unit test sees.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kasa.cli import app
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.manifest import Manifest

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

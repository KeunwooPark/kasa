"""The CLI commands themselves, invoked the way a shell invokes them.

Most of the surface is covered by the modules underneath. What is not, and what
this file is for, is the *order* the commands do things in — a command that
writes before it validates fails in a way no unit test sees.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kasa import __version__
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

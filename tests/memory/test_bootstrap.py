from __future__ import annotations

import json
from pathlib import Path

from kasa.memory import bootstrap as bootstrap_module
from kasa.memory.bootstrap import bootstrap, is_bootstrapped, refresh_schema
from kasa.memory.layout import (
    INDEX_PATH,
    MANIFEST_PATH,
    SCHEMA_PATH,
    TYPE_DIRS,
    is_machinery,
    is_memory_path,
)


def test_bootstrap_lays_out_the_whole_skeleton(tmp_path: Path) -> None:
    written = bootstrap(tmp_path)

    for directory in TYPE_DIRS:
        assert (tmp_path / "memory" / directory / ".gitkeep").exists()
    assert (tmp_path / "memory/archive/.gitkeep").exists()
    assert (tmp_path / SCHEMA_PATH).exists()
    assert (tmp_path / INDEX_PATH).exists()
    assert set(written) >= {SCHEMA_PATH, MANIFEST_PATH, INDEX_PATH, "README.md"}
    assert is_bootstrapped(tmp_path)


def test_bootstrap_never_overwrites(tmp_path: Path) -> None:
    bootstrap(tmp_path)
    (tmp_path / "README.md").write_text("mine")
    (tmp_path / MANIFEST_PATH).write_text('{"version": 1, "memories": {"mem_01": "x"}}')

    assert bootstrap(tmp_path) == [], "a second pass writes nothing"
    assert (tmp_path / "README.md").read_text() == "mine"
    assert json.loads((tmp_path / MANIFEST_PATH).read_text())["memories"] == {"mem_01": "x"}


def test_the_repo_readme_says_the_repo_must_stay_private(tmp_path: Path) -> None:
    bootstrap(tmp_path)
    assert "private" in (tmp_path / "README.md").read_text().lower()


def test_schema_is_refreshed_only_when_it_drifts(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    bootstrap(tmp_path)
    assert refresh_schema(tmp_path) is False

    # The generated contract is the one exception to never overwriting: a stale
    # schema is what the consolidation model would follow off a cliff.
    monkeypatch.setattr(bootstrap_module, "render_schema_md", lambda: "# v2 contract\n")
    assert refresh_schema(tmp_path) is True
    assert (tmp_path / SCHEMA_PATH).read_text() == "# v2 contract\n"


def test_bootstrap_is_partial_recovery_safe(tmp_path: Path) -> None:
    """A run interrupted halfway leaves a repo the next run can finish."""
    bootstrap(tmp_path)
    (tmp_path / MANIFEST_PATH).unlink()
    assert not is_bootstrapped(tmp_path)

    assert bootstrap(tmp_path) == [MANIFEST_PATH]
    assert is_bootstrapped(tmp_path)


def test_machinery_is_distinguishable_from_memories() -> None:
    assert is_memory_path("memory/people/jane.md")
    assert is_memory_path("memory/journal/2026/09/03.md")

    assert not is_memory_path(SCHEMA_PATH)
    assert not is_memory_path(MANIFEST_PATH)
    assert not is_memory_path(INDEX_PATH)
    assert not is_memory_path("README.md"), "outside memory/"
    assert not is_memory_path("memory/people/jane.txt"), "not markdown"
    assert not is_memory_path("memory/../../etc/passwd")
    assert not is_memory_path("/etc/passwd")

    assert is_machinery(SCHEMA_PATH)
    assert is_machinery(INDEX_PATH)
    assert not is_machinery("memory/people/jane.md")

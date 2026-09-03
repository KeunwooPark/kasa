"""Lay out an empty long-term memory repository.

Additive by construction: every write is skipped when the target already exists.
`kasa init` is expected to be re-run — after a config change, on a second
machine, or just because the user forgot they had run it — and the one thing it
must never do is overwrite a corpus that has been accumulating for months.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from kasa.memory.layout import ARCHIVE_DIR, INDEX_PATH, MANIFEST_PATH, SCHEMA_PATH, TYPE_DIRS
from kasa.memory.schema import (
    SCHEMA_VERSION,
    render_memory_index,
    render_repo_readme,
    render_schema_md,
)

MANIFEST_VERSION = 1

_GITIGNORE = """# Kasa keeps nothing derived in here; the index lives in SQLite.
.DS_Store
"""


def empty_manifest() -> dict[str, object]:
    return {
        "version": MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "memories": {},
    }


def bootstrap(root: Path) -> list[str]:
    """Create whatever is missing under `root`. Returns the paths written."""
    root = root.expanduser()
    written: list[str] = []

    for directory in TYPE_DIRS:
        # Git tracks files, not directories, so the skeleton needs a placeholder
        # to survive a clone. They are removed as real memories land.
        written += _write(root, f"memory/{directory}/.gitkeep", "")
    written += _write(root, f"{ARCHIVE_DIR}/.gitkeep", "")

    written += _write(root, "README.md", render_repo_readme())
    written += _write(root, ".gitignore", _GITIGNORE)
    written += _write(root, INDEX_PATH, render_memory_index())
    written += _write(root, SCHEMA_PATH, render_schema_md())
    written += _write(root, MANIFEST_PATH, json.dumps(empty_manifest(), indent=2) + "\n")
    return written


def is_bootstrapped(root: Path) -> bool:
    root = root.expanduser()
    return (root / SCHEMA_PATH).exists() and (root / MANIFEST_PATH).exists()


def refresh_schema(root: Path) -> bool:
    """Rewrite `.kasa/schema.md` if this version of Kasa disagrees with it.

    The exception to never overwriting: the schema is generated machinery, and a
    stale contract is precisely what the model would follow off a cliff.
    """
    target = root.expanduser() / SCHEMA_PATH
    current = render_schema_md()
    if target.exists() and target.read_text() == current:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(current)
    return True


def _write(root: Path, relative: str, content: str) -> list[str]:
    target = root / relative
    if target.exists():
        return []
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return [relative]

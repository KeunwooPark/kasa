"""Where things live inside the long-term memory repository.

One module so that the paths are stated once. `#13`'s patch validator and
`#14`'s indexer both need to agree with the bootstrapper about which parts of
the tree are memories and which are machinery, and agreeing by coincidence is
how a validator ends up letting a model rewrite its own manifest.
"""

from __future__ import annotations

from pathlib import Path

#: Everything the agent may ever write lives under here.
MEMORY_DIR = "memory"

#: Machinery: generated, written only by deterministic code. Inside `memory/`
#: because `docs/DESIGN.md` §4.5 draws it there, so patch validation carves it
#: out explicitly rather than relying on it being somewhere else.
KASA_DIR = f"{MEMORY_DIR}/.kasa"

SCHEMA_PATH = f"{KASA_DIR}/schema.md"
MANIFEST_PATH = f"{KASA_DIR}/manifest.json"
INDEX_PATH = f"{MEMORY_DIR}/README.md"

#: One directory per memory `type`, plus the soft-delete tier.
TYPE_DIRS = ("people", "projects", "topics", "facts", "journal")
ARCHIVE_DIR = f"{MEMORY_DIR}/archive"


#: The name every generated listing goes under, at any depth. `reorganize`
#: regenerates them from the manifest; a patch plan may not write one.
INDEX_NAME = "README.md"


def is_index_page(relative_path: str | Path) -> bool:
    """True for a generated listing under `memory/`.

    A directory index is a memory-shaped path holding something that is not a
    memory — no frontmatter, nothing anybody claimed. Naming it here is what
    keeps the manifest and the search index from walking into it and reporting
    it as a broken file every time they run.
    """
    path = Path(relative_path)
    if ".." in path.parts or path.is_absolute():
        return False
    return path.parts[:1] == (MEMORY_DIR,) and path.name == INDEX_NAME


def is_machinery(relative_path: str | Path) -> bool:
    """True for generated files no patch plan may touch."""
    path = Path(relative_path)
    return ".kasa" in path.parts or is_index_page(path)


def is_memory_path(relative_path: str | Path) -> bool:
    """True for a path a memory document may legitimately occupy."""
    path = Path(relative_path)
    if ".." in path.parts or path.is_absolute():
        return False
    if path.parts[:1] != (MEMORY_DIR,):
        return False
    return not is_machinery(path) and path.suffix == ".md"

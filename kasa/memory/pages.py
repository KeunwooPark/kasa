"""The generated listings: `memory/README.md` and one per directory.

The repo is meant to be opened by a person. A directory holding two hundred
files named after slugs is not something anybody browses, so `reorganize`
regenerates an index beside them — the human entry point the design asks for
(§4.4).

Generated from the manifest and never by a model. These are the one place in
`memory/` where a file is not somebody's claim about the world, which is why
`is_machinery` refuses them to patch plans and why the manifest and the search
index walk past them instead of reporting them as files that will not parse.

The listing is rebuilt whole every time, so nothing accumulates in it and a
memory that went away leaves nothing behind. Anything written above the marker
in `memory/README.md` is kept: it is the front page of somebody's repository,
and a job that erased the paragraph they wrote at the top of it every week is a
job they turn off.
"""

from __future__ import annotations

from collections.abc import Iterable

from kasa.memory.layout import ARCHIVE_DIR, INDEX_NAME, INDEX_PATH, MEMORY_DIR, TYPE_DIRS
from kasa.memory.manifest import Manifest, ManifestEntry

#: Everything below this line is regenerated. Above it is whoever's repo it is.
MARKER = "<!-- Kasa regenerates the listing below. Text above this comment is preserved. -->"


def render_pages(manifest: Manifest, *, preamble: str | None = None) -> dict[str, str]:
    """Every generated listing the corpus currently needs, keyed by path.

    A directory with nothing in it gets no page: an index of an empty folder is
    a file somebody has to open to learn there is nothing to read.
    """
    live = _live(manifest)
    pages = {INDEX_PATH: _root_page(live, preamble)}
    for directory in TYPE_DIRS:
        entries = [e for e in live if _directory_of(e.path) == directory]
        if entries:
            pages[f"{MEMORY_DIR}/{directory}/{INDEX_NAME}"] = _directory_page(directory, entries)
    return pages


def preamble_of(current: str) -> str | None:
    """Whatever the existing root index says above the marker."""
    head, marker, _ = current.partition(MARKER)
    return head.rstrip() if marker else None


def _root_page(entries: Iterable[ManifestEntry], preamble: str | None) -> str:
    listed = sorted(entries, key=lambda e: (e.path, e.title))
    lines = [preamble.rstrip() if preamble else "# Memory index", "", MARKER, ""]
    if not listed:
        return "\n".join(
            [
                *lines,
                "Nothing here yet. Memories appear as the agent promotes",
                "them out of conversation.",
                "",
            ]
        )

    lines += [f"{len(listed)} memories.", ""]
    for directory in TYPE_DIRS:
        group = [e for e in listed if _directory_of(e.path) == directory]
        if not group:
            continue
        lines += [f"## {directory}", ""]
        lines += [_row(entry) for entry in group]
        lines.append("")
    return "\n".join(lines)


def _directory_page(directory: str, entries: list[ManifestEntry]) -> str:
    lines = [f"# {directory}", "", MARKER, "", f"{len(entries)} memories.", ""]
    lines += [
        _row(entry, relative_to=directory) for entry in sorted(entries, key=lambda e: e.title)
    ]
    return "\n".join(lines) + "\n"


def _row(entry: ManifestEntry, *, relative_to: str | None = None) -> str:
    """One line of a listing.

    An ordinary Markdown link, not a wikilink: these files are read on GitHub,
    where a wikilink is four literal brackets and a relative path is a link
    somebody can click. Visibility is shown because a corpus where everything
    looks alike is one where nobody notices that half of it came from DMs.
    """
    target = entry.path.removeprefix(f"{MEMORY_DIR}/")
    if relative_to:
        target = target.removeprefix(f"{relative_to}/")
    scope = "" if entry.visibility == "workspace" else f" _({entry.visibility})_"
    return f"- [{entry.title}]({target}){scope}"


def _live(manifest: Manifest) -> list[ManifestEntry]:
    """Everything except the archive. A listing of the soft-delete tier would
    be a table of contents for things that stopped being true."""
    return [
        entry
        for entry in manifest.memories.values()
        if not entry.path.startswith(f"{ARCHIVE_DIR}/")
    ]


def _directory_of(path: str) -> str:
    parts = path.split("/")
    return parts[1] if len(parts) > 2 else ""

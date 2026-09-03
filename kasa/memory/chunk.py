"""Splitting a memory document into retrievable pieces.

Chunking is where retrieval quality is won or lost, and the tempting mistake is
to cut on a character count. A memory file is already structured — a title, some
tags, prose under headings — so the cuts follow that structure and fall back to
size only when a section is genuinely too long to return whole.

Each chunk carries the scope and salience of the memory it came from. That is
denormalization on purpose: retrieval has to filter by scope *before* it ranks,
and a filter that needs a join is a filter somebody eventually applies after.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from kasa.memory.document import MemoryDoc

#: Roughly 300 tokens. Big enough to hold an argument, small enough that several
#: fit in a retrieval budget.
MAX_CHUNK_CHARS = 1200

#: Below this a chunk is a fragment, and a fragment retrieved on its own reads as
#: a non-sequitur. Short tails are folded back into the chunk before them.
MIN_CHUNK_CHARS = 120

_HEADING = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_PARAGRAPH = re.compile(r"\n\s*\n")


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    memory_id: str
    path: str
    ordinal: int
    text: str
    scope: str
    salience: float
    updated_at: str


def chunk_document(doc: MemoryDoc, path: str) -> list[Chunk]:
    """Split one memory into chunks, header first."""
    frontmatter = doc.frontmatter
    texts = [_header_text(doc), *split_body(doc.body)]
    updated = frontmatter.updated.isoformat(timespec="seconds")
    return [
        Chunk(
            # Deterministic, so that deleting the database and rebuilding it
            # produces the same index rather than a differently-keyed one.
            id=f"{frontmatter.id}:{ordinal}",
            memory_id=frontmatter.id,
            path=path,
            ordinal=ordinal,
            text=text,
            scope=frontmatter.visibility,
            salience=frontmatter.salience,
            updated_at=updated,
        )
        for ordinal, text in enumerate(texts)
    ]


def split_body(body: str) -> list[str]:
    """Cut prose on headings, then paragraphs, then — reluctantly — on size."""
    pieces: list[str] = []
    for section in _split_on_headings(body):
        pieces.extend(_split_to_size(section))
    return _fold_short_tails(pieces)


def _header_text(doc: MemoryDoc) -> str:
    """A synthetic first chunk holding the title and tags.

    Without it, a memory titled "Deploy pipeline ownership" whose body never
    repeats the phrase cannot be found by searching for it — which is the first
    thing anyone tries.
    """
    frontmatter = doc.frontmatter
    lines = [frontmatter.title]
    if frontmatter.tags:
        lines.append(" ".join(frontmatter.tags))
    return "\n".join(lines)


def _split_on_headings(body: str) -> list[str]:
    boundaries = [match.start() for match in _HEADING.finditer(body)]
    if not boundaries:
        return [body.strip()] if body.strip() else []

    starts = [0, *boundaries] if boundaries[0] != 0 else boundaries
    sections = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(body)
        if section := body[start:end].strip():
            sections.append(section)
    return sections


def _split_to_size(section: str) -> list[str]:
    if len(section) <= MAX_CHUNK_CHARS:
        return [section]

    chunks: list[str] = []
    current = ""
    for paragraph in _PARAGRAPH.split(section):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        # A single paragraph longer than the cap has no boundary left to use.
        while len(current) > MAX_CHUNK_CHARS:
            chunks.append(current[:MAX_CHUNK_CHARS])
            current = current[MAX_CHUNK_CHARS:]
    if current:
        chunks.append(current)
    return chunks


def _fold_short_tails(pieces: list[str]) -> list[str]:
    folded: list[str] = []
    for piece in pieces:
        if (
            folded
            and len(piece) < MIN_CHUNK_CHARS
            and len(folded[-1]) + len(piece) <= MAX_CHUNK_CHARS
        ):
            folded[-1] = f"{folded[-1]}\n\n{piece}"
        else:
            folded.append(piece)
    return folded

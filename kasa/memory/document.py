"""A long-term memory document: YAML frontmatter, Markdown body.

The frontmatter is a contract; the body is prose. That split runs through the
whole module. Frontmatter is parsed into a validated model and **re-serialized
canonically**, so a hundred files written by a model over six months all look
the same in a diff. The body is passed through **byte for byte**, because a
person is going to open these files and rewrite them, and an agent that
reflows someone's paragraphs is an agent they stop trusting with the file.

Parsing is deliberately permissive about YAML — people hand-edit these — and
strict about the resulting values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from ulid import ULID

from kasa.errors import KasaError

MemoryType = Literal["person", "project", "topic", "fact", "journal"]

MEMORY_TYPES: tuple[str, ...] = ("person", "project", "topic", "fact", "journal")

#: Directory under `memory/` for each type. Journals nest further by date.
TYPE_DIRECTORY: dict[str, str] = {
    "person": "people",
    "project": "projects",
    "topic": "topics",
    "fact": "facts",
    "journal": "journal",
}

ID_PREFIX = "mem_"
_ID = re.compile(rf"^{ID_PREFIX}[0-7][0-9A-HJKMNP-TV-Z]{{25}}$")

#: `workspace`, or a Slack channel / user the memory is confined to.
_VISIBILITY = re.compile(r"^(workspace|channel:[A-Za-z0-9_-]+|private:[A-Za-z0-9_-]+)$")

_FENCE = "---"
_FRONTMATTER = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

#: `[[mem_01...]]` or `[[people/jane]]`, optionally `[[target|shown text]]`.
_WIKILINK = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")

Unit = Annotated[float, Field(ge=0.0, le=1.0)]


class MemoryError_(KasaError):
    """A memory document could not be parsed or is invalid.

    Keeps the reason separate from the file it came from. `str(exc)` is still
    `"<source>: <reason>"`, because an error read on its own has to say which
    file it is about — but a caller that already knows the path can compose its
    own line instead of printing the path twice (#70).
    """

    def __init__(self, reason: str, *, source: str = "") -> None:
        self.reason = reason
        self.source = source
        super().__init__(f"{source}: {reason}" if source else reason)


@dataclass(frozen=True, slots=True)
class Problem:
    """A memory file that could not be read, and why.

    `reason` is the bare reason. The path is the other field, and every caller
    renders the two together in its own shape.

    Here rather than in `manifest`, because the index refuses the same files
    for the same reasons and had no way to say so — it reported bare paths, so
    `kasa reindex` named a broken file once without a reason and once with one
    (#77).
    """

    path: str
    reason: str


class Frontmatter(BaseModel):
    """The typed header of a memory file.

    Field order here is the field order on disk: the serializer walks the model,
    so the two cannot drift, and neither can `.kasa/schema.md`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="ULID, assigned once and never rewritten")
    type: MemoryType = Field(description="which kind of thing this memory is about")
    title: str = Field(description="one line, human-readable; shown in the index")
    tags: list[str] = Field(default_factory=list, description="lowercase, for exact-match recall")
    visibility: str = Field(
        default="workspace",
        description="workspace | channel:C0123 | private:U0456; never widened once set",
    )
    created: datetime = Field(description="when this memory was first written")
    updated: datetime = Field(description="when it last changed")
    confidence: Unit = Field(default=0.8, description="how sure we are that it is true")
    salience: Unit = Field(default=0.5, description="access-weighted importance; decays over time")
    pinned: bool = Field(default=False, description="pinned memories are never forgotten")
    source_refs: list[str] = Field(
        default_factory=list,
        description="where this came from, e.g. slack://T01/C0123/1756890000.1",
    )
    supersedes: list[str] = Field(
        default_factory=list, description="ids this memory replaced; the chain keeps links alive"
    )

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        if not _ID.match(value):
            raise ValueError(f"{value!r} is not a memory id ({ID_PREFIX} followed by a ULID)")
        return value

    @field_validator("visibility")
    @classmethod
    def _valid_visibility(cls, value: str) -> str:
        if not _VISIBILITY.match(value):
            raise ValueError(
                f"{value!r} is not a visibility scope (workspace, channel:<id>, or private:<id>)"
            )
        return value

    @field_validator("supersedes")
    @classmethod
    def _valid_supersedes(cls, value: list[str]) -> list[str]:
        for entry in value:
            if not _ID.match(entry):
                raise ValueError(f"{entry!r} in `supersedes` is not a memory id")
        return value

    @field_validator("created", "updated")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        # A naive timestamp compares wrong against every other one in the corpus,
        # and recency decay is scored on these.
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @property
    def is_private(self) -> bool:
        return self.visibility != "workspace"

    def touch(self) -> Frontmatter:
        return self.model_copy(update={"updated": _now()})


class MemoryDoc(BaseModel):
    """A parsed memory file: validated header, verbatim body."""

    model_config = ConfigDict(extra="forbid")

    frontmatter: Frontmatter
    body: str = ""

    @classmethod
    def new(
        cls,
        *,
        type: MemoryType,
        title: str,
        body: str = "",
        **fields: Any,
    ) -> Self:
        now = _now()
        # A generated file gets the blank line after the fence that a person
        # would have left there. Parsed documents keep whatever they already had.
        if body and not body.startswith("\n"):
            body = f"\n{body}"
        return cls(
            frontmatter=Frontmatter(
                id=new_memory_id(), type=type, title=title, created=now, updated=now, **fields
            ),
            body=body,
        )

    @classmethod
    def parse(cls, text: str, *, source: str = "<memory>") -> Self:
        match = _FRONTMATTER.match(text)
        if match is None:
            raise MemoryError_("no YAML frontmatter (a file must open with `---`)", source=source)
        try:
            raw = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise MemoryError_(f"frontmatter is not valid YAML: {exc}", source=source) from exc
        if not isinstance(raw, dict):
            raise MemoryError_("frontmatter must be a mapping", source=source)

        try:
            frontmatter = Frontmatter.model_validate(raw)
        except Exception as exc:
            raise MemoryError_(_first_line(exc), source=source) from exc
        return cls(frontmatter=frontmatter, body=text[match.end() :])

    def render(self) -> str:
        """Serialize canonically. The body is emitted exactly as it came in.

        The newline after the closing fence belongs to the fence, not to the
        body — writing it any other way makes `parse(render(doc)).body` differ
        from `doc.body` by a leading newline, and the reorganizer rewrites every
        file it touches.
        """
        lines = [_FENCE]
        for name in Frontmatter.model_fields:
            lines.append(f"{name}: {_yaml_value(getattr(self.frontmatter, name))}")
        lines.append(_FENCE)
        rendered = "\n".join(lines) + "\n" + self.body
        return rendered if rendered.endswith("\n") else rendered + "\n"

    def links(self) -> list[str]:
        """Wikilink targets in the body, in order, without duplicates."""
        seen: dict[str, None] = {}
        for match in _WIKILINK.finditer(self.body):
            seen.setdefault(match.group(1).strip(), None)
        return list(seen)

    @property
    def id(self) -> str:
        return self.frontmatter.id

    def suggested_path(self, slug: str | None = None) -> str:
        """Where a memory of this type belongs, by convention."""
        directory = TYPE_DIRECTORY[self.frontmatter.type]
        return f"memory/{directory}/{slug or slugify(self.frontmatter.title)}.md"


def new_memory_id() -> str:
    return f"{ID_PREFIX}{ULID()}"


def is_memory_id(value: str) -> bool:
    return bool(_ID.match(value))


#: How much of a title becomes a filename. A filename is a handle, not the
#: title — the title is in the frontmatter and the id is the durable reference,
#: so truncating loses nothing readers need. Unbounded, a long title produced a
#: basename past the filesystem's 255-byte limit and the write failed with an
#: `OSError` from inside the patch validator (#93).
MAX_SLUG_CHARS = 80


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    # Trimmed after the strip, and stripped again: cutting mid-word can leave a
    # trailing separator, and `deploy-.md` is not a name anybody wrote.
    return slug[:MAX_SLUG_CHARS].strip("-") or "untitled"


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _yaml_value(value: Any) -> str:
    """One frontmatter value, YAML-escaped.

    Delegated to PyYAML per value rather than hand-quoted: getting `title: "a: b"`
    wrong writes a file that no longer parses, and the corpus is the source of
    truth. Flow style keeps lists on one line, as the schema shows them.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    dumped = yaml.safe_dump(
        value, default_flow_style=True, allow_unicode=True, width=10**6, sort_keys=False
    ).strip()
    # safe_dump ends a document with "\n...\n" for bare scalars; drop it.
    return dumped.removesuffix("...").strip()


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__

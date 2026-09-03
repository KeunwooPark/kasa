from __future__ import annotations

import re
from datetime import UTC, datetime

import pytest

from kasa.memory.document import (
    Frontmatter,
    MemoryDoc,
    MemoryError_,
    is_memory_id,
    new_memory_id,
    slugify,
)
from kasa.memory.schema import render_schema_md

ID = "mem_01K8XQ4W2N7B6VJ3ZC9F0RTKME"

HAND_WRITTEN = f"""---
id: {ID}
type: person
title: Deploy pipeline ownership
tags: [infra, ownership]
visibility: workspace
created: 2026-09-03T10:12:00Z
updated: 2026-09-03T10:12:00Z
confidence: 0.9
salience: 0.7
pinned: false
source_refs:
  - slack://T01/C0123/1756890000.123
supersedes: []
---

Jane owns the deploy pipeline.

See [[projects/deploy]] and [[{ID}]].
"""


# -- parsing -----------------------------------------------------------------


def test_a_hand_written_file_parses() -> None:
    doc = MemoryDoc.parse(HAND_WRITTEN)

    assert doc.id == ID
    assert doc.frontmatter.type == "person"
    assert doc.frontmatter.tags == ["infra", "ownership"]
    assert doc.frontmatter.salience == 0.7
    assert doc.frontmatter.source_refs == ["slack://T01/C0123/1756890000.123"]
    assert doc.frontmatter.created == datetime(2026, 9, 3, 10, 12, tzinfo=UTC)


def test_block_and_inline_lists_both_work() -> None:
    """People hand-edit these; YAML's two list syntaxes must both be accepted."""
    block = MemoryDoc.parse(HAND_WRITTEN.replace("tags: [infra, ownership]", "tags:\n  - infra"))
    assert block.frontmatter.tags == ["infra"]


def test_defaults_fill_in_for_an_abbreviated_header() -> None:
    text = (
        f"---\nid: {ID}\ntype: fact\ntitle: A\ncreated: 2026-09-03T00:00:00Z\n"
        "updated: 2026-09-03T00:00:00Z\n---\n\nbody\n"
    )
    doc = MemoryDoc.parse(text)

    assert doc.frontmatter.visibility == "workspace"
    assert doc.frontmatter.pinned is False
    assert doc.frontmatter.tags == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("no frontmatter at all\n", "no YAML frontmatter"),
        ("---\n: : :\nnot yaml\n---\nbody\n", "not valid YAML"),
        ("---\njust a string\n---\nbody\n", "must be a mapping"),
    ],
)
def test_malformed_files_say_what_is_wrong(text: str, expected: str) -> None:
    with pytest.raises(MemoryError_, match=expected):
        MemoryDoc.parse(text)


def test_the_source_name_is_in_the_error() -> None:
    """A corpus of hundreds of files needs the error to name the file."""
    with pytest.raises(MemoryError_, match=re.escape("memory/people/jane.md")):
        MemoryDoc.parse("nope\n", source="memory/people/jane.md")


@pytest.mark.parametrize(
    "bad",
    [
        "id: not-a-ulid",
        "type: creature",
        "visibility: everyone",
        "confidence: 1.4",
        "salience: -1",
        "supersedes: [nonsense]",
        "unexpected_key: 1",
    ],
)
def test_invalid_frontmatter_is_rejected(bad: str) -> None:
    key = bad.split(":")[0]
    lines = [line for line in HAND_WRITTEN.splitlines() if not line.startswith(f"{key}:")]
    lines.insert(1, bad)
    with pytest.raises(MemoryError_):
        MemoryDoc.parse("\n".join(lines) + "\n")


def test_a_naive_timestamp_is_read_as_utc() -> None:
    """Naive timestamps compare wrong against the rest of the corpus."""
    doc = MemoryDoc.parse(HAND_WRITTEN.replace("2026-09-03T10:12:00Z", "2026-09-03 10:12:00"))
    assert doc.frontmatter.created.tzinfo is not None
    assert doc.frontmatter.created == datetime(2026, 9, 3, 10, 12, tzinfo=UTC)


# -- serializing -------------------------------------------------------------


def test_the_body_survives_a_round_trip_byte_for_byte() -> None:
    doc = MemoryDoc.parse(HAND_WRITTEN)
    assert MemoryDoc.parse(doc.render()).body == doc.body


def test_rendering_is_idempotent() -> None:
    once = MemoryDoc.parse(HAND_WRITTEN).render()
    assert MemoryDoc.parse(once).render() == once


def test_human_formatting_in_the_body_is_left_alone() -> None:
    body = "\n# A heading\n\n    indented code\n\n\n\nthree blank lines above\n"
    doc = MemoryDoc.new(type="topic", title="T", body=body)
    assert MemoryDoc.parse(doc.render()).body == body


def test_frontmatter_is_canonicalized_even_when_the_body_is_not() -> None:
    """A hundred files written over six months should diff cleanly."""
    messy = HAND_WRITTEN.replace(
        "source_refs:\n  - slack://T01/C0123/1756890000.123",
        "source_refs: ['slack://T01/C0123/1756890000.123']",
    )
    assert MemoryDoc.parse(messy).render() == MemoryDoc.parse(HAND_WRITTEN).render()


def test_field_order_on_disk_follows_the_model() -> None:
    rendered = MemoryDoc.parse(HAND_WRITTEN).render()
    written = [line.split(":")[0] for line in rendered.splitlines()[1:13]]
    assert written == list(Frontmatter.model_fields)


def test_awkward_titles_survive() -> None:
    for title in ["a: colon", "quotes 'and' \"more\"", "- leading dash", "#hash", "emoji 🌱"]:
        doc = MemoryDoc.new(type="fact", title=title)
        assert MemoryDoc.parse(doc.render()).frontmatter.title == title


def test_a_body_containing_a_fence_does_not_confuse_the_parser() -> None:
    doc = MemoryDoc.new(type="fact", title="T", body="before\n\n---\n\nafter\n")
    assert "after" in MemoryDoc.parse(doc.render()).body


# -- ids and links -----------------------------------------------------------


def test_new_ids_are_valid_and_unique() -> None:
    ids = {new_memory_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(is_memory_id(i) for i in ids)


def test_an_id_is_not_rewritten_by_a_round_trip() -> None:
    doc = MemoryDoc.parse(HAND_WRITTEN)
    assert MemoryDoc.parse(doc.render()).id == ID


def test_links_are_extracted_in_order_without_duplicates() -> None:
    doc = MemoryDoc.parse(HAND_WRITTEN)
    assert doc.links() == ["projects/deploy", ID]


def test_link_forms_are_all_recognized() -> None:
    doc = MemoryDoc.new(
        type="fact",
        title="T",
        body="[[a/b]] [[a/b|shown text]] [[a/b#section]] [[ spaced ]] [[a/b]]\n",
    )
    assert doc.links() == ["a/b", "spaced"]


def test_suggested_path_follows_the_type() -> None:
    assert MemoryDoc.new(type="person", title="Jane Doe").suggested_path() == (
        "memory/people/jane-doe.md"
    )
    assert MemoryDoc.new(type="fact", title="X").suggested_path("custom") == (
        "memory/facts/custom.md"
    )


@pytest.mark.parametrize(
    ("title", "slug"),
    [("Jane Doe", "jane-doe"), ("A: B/C", "a-b-c"), ("  ", "untitled"), ("🌱", "untitled")],
)
def test_slugs_are_filesystem_safe(title: str, slug: str) -> None:
    assert slugify(title) == slug


# -- the generated contract --------------------------------------------------


def test_the_schema_documents_every_field() -> None:
    """The contract is generated from the model, so it cannot drift from it."""
    schema = render_schema_md()
    for name in Frontmatter.model_fields:
        assert f"`{name}`" in schema


def test_the_schema_table_is_not_broken_by_a_pipe_in_a_description() -> None:
    # `visibility` is described as "workspace | channel:C0123 | ..." — unescaped,
    # that silently splits the row into extra columns.
    row = next(
        line for line in render_schema_md().splitlines() if line.startswith("| `visibility`")
    )
    assert row.count("|") - row.count("\\|") == 6, "one row, six cell boundaries"


def test_the_schema_names_no_undefined_defaults() -> None:
    assert "PydanticUndefined" not in render_schema_md()

"""Subject normalization: the same entity has to produce the same key."""

from __future__ import annotations

import pytest

from kasa.memory.subject import MAX_SUBJECT_CHARS, normalize_subject


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Jane Doe", "jane doe"),
        ("  Jane   Doe  ", "jane doe"),
        ("Jane Doe's", "jane doe"),
        ("Jane Doe\u2019s", "jane doe"),  # the curly apostrophe every phone types
        ("JANE DOE", "jane doe"),
        ("Jane Doe.", "jane doe"),
        ("the deploy pipeline", "deploy pipeline"),
        ("The Deploy Pipeline", "deploy pipeline"),
        ("A deploy pipeline", "deploy pipeline"),
        # An internal hyphen is part of the name; punctuation between words is
        # a separator, not something to close up.
        ("kasa-ltm", "kasa-ltm"),
        ("kasa/ltm", "kasa ltm"),
    ],
)
def test_the_same_entity_normalizes_to_the_same_key(raw: str, expected: str) -> None:
    assert normalize_subject(raw) == expected


def test_normalizing_twice_changes_nothing() -> None:
    """The store normalizes what a caller may already have normalized, so this
    has to be a fixed point or the key depends on how many layers it crossed."""
    for raw in ("The Deploy Pipeline's", "Jane Doe", "a-b c", "kasa/ltm"):
        once = normalize_subject(raw)
        assert normalize_subject(once) == once


def test_a_subject_that_says_nothing_normalizes_to_nothing() -> None:
    """Empty is a real answer. A model asked for the subject of a claim it is
    not sure about will return punctuation, and storing `???` as an entity is
    how a group forms that nothing can ever join."""
    for raw in ("", "   ", "???", "...", "!!!"):
        assert normalize_subject(raw) == ""


def test_a_subject_is_a_key_not_a_paragraph() -> None:
    """Untrusted text reaches this. A subject is a grouping key, and one the
    length of a transcript groups with nothing while costing the same index."""
    assert len(normalize_subject("word " * 200)) <= MAX_SUBJECT_CHARS


def test_truncation_does_not_leave_a_dangling_separator() -> None:
    assert not normalize_subject(("a" * (MAX_SUBJECT_CHARS - 1)) + " b").endswith(("-", " "))

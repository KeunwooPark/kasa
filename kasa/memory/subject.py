"""One normalized key per entity, so the same thing groups across episodes.

`observations.subject` is a grouping key, not a label. `promote` collects every
pending observation about one subject and asks the model to reconcile them
against long-term memory in a single pass, and that only works if two people
saying the same name in two different conversations land on the same string.

Deliberately deterministic and deliberately dull. A model-driven entity
resolver would group better and would also mean the grouping changed under you
between runs; this one can be reasoned about, tested, and read in a diff. The
harder cases — "Jane" and "Jane Doe" being the same person — are for the
retrieval step in `promote`, which sees the existing corpus and can tell.
"""

from __future__ import annotations

import re
import unicodedata

#: Long enough for "the deploy pipeline release checklist" and short enough
#: that a subject is a key rather than a paragraph somebody smuggled in.
MAX_SUBJECT_CHARS = 120

_POSSESSIVE = re.compile(r"['\u2018\u2019\u02bc]s\b")
#: Anything that is not a word character, a space, or an internal hyphen. Kept
#: separate from whitespace collapsing so `kasa/ltm` becomes `kasa ltm` rather
#: than `kasaltm`.
_PUNCTUATION = re.compile(r"[^\w\s-]+", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_ARTICLE = re.compile(r"^(?:the|a|an)\s+")


def normalize_subject(raw: str) -> str:
    """The grouping key for a subject, or `""` if it says nothing.

    Empty is a real answer, not a failure: a model asked for the subject of a
    claim can return `"..."` or `"???"`, and the caller has to decide what to
    do about that rather than store punctuation as an entity.
    """
    text = unicodedata.normalize("NFKC", raw).casefold()
    text = _POSSESSIVE.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    # After punctuation, so "The deploy pipeline." and "the deploy pipeline"
    # meet before the article is taken off either of them.
    text = _ARTICLE.sub("", text)
    return text[:MAX_SUBJECT_CHARS].strip(" -")

"""Finding the memories that say the same thing twice.

`promote` writes a memory when the corpus does not already cover a subject, and
"does not already cover" is a judgement made by a model looking at whatever
retrieval put in front of it. It will sometimes be wrong, and the wrongness
accumulates: two files about one person, three about one decision, all of them
true and none of them the place to look.

Finding the pairs is deterministic on purpose. A model deciding what is worth
comparing would cost a call per memory and give a different answer every week;
this gives the same answer every week, can be read in a diff, and is only ever
a *candidate* — whether two memories actually say the same thing is the
question `reorganize` puts to a model afterwards, over the two full documents.

The measure is token overlap, which is crude and has the property that matters:
it never suggests a pair that shares no vocabulary, so a corpus of genuinely
distinct memories costs nothing to check.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from kasa.memory.document import MemoryDoc

#: Words too common to be evidence of anything. Not a stopword list for
#: retrieval — that lives in `retrieve.py` and is tuned for recall. This one
#: only has to stop "the" from making two memories look alike.
#: Written as prose rather than as a list literal, because what anybody
#: reviewing it wants to check is whether a word belongs, not whether it is
#: quoted.
_NOISE_WORDS = (
    "a an and are as at be by for from has have in is it its of on or that the to was were with"
    " this these those they them their he she his her we our you your not no"
)
_NOISE = frozenset(_NOISE_WORDS.split())

_WORD = re.compile(r"[a-z0-9]+")


def tokens(doc: MemoryDoc) -> frozenset[str]:
    """The words a memory is about, title included."""
    text = f"{doc.frontmatter.title} {' '.join(doc.frontmatter.tags)} {doc.body}".lower()
    return frozenset(word for word in _WORD.findall(text) if word not in _NOISE and len(word) > 2)


def overlap(first: frozenset[str], second: frozenset[str]) -> float:
    """Jaccard similarity. 0 when either says nothing at all."""
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def clusters(
    docs: Sequence[tuple[str, MemoryDoc]],
    *,
    threshold: float,
    max_cluster: int,
    max_clusters: int,
) -> list[list[tuple[str, MemoryDoc]]]:
    """Groups of memories that overlap enough to be worth asking about.

    Only within one `type` and one `visibility`. A person and a project that
    share vocabulary are not duplicates, and two memories with different
    audiences may not be merged at all — the patch validator refuses it, so
    proposing it would only spend a model call to be told no.

    Transitive within a group, capped by `max_cluster`: three files about one
    person are one question, not three, and a cap keeps a corpus full of
    near-identical notes from becoming a single unreadable merge.
    """
    grouped: dict[tuple[str, str], list[tuple[str, MemoryDoc]]] = {}
    for path, doc in docs:
        key = (doc.frontmatter.type, doc.frontmatter.visibility)
        grouped.setdefault(key, []).append((path, doc))

    found: list[list[tuple[str, MemoryDoc]]] = []
    for members in grouped.values():
        found.extend(_cluster(members, threshold=threshold, max_cluster=max_cluster))
    # Densest first: the pair that overlaps most is the likeliest duplicate,
    # and a bounded run should spend its calls on those.
    found.sort(key=_density, reverse=True)
    return found[:max_clusters]


def _cluster(
    members: Sequence[tuple[str, MemoryDoc]], *, threshold: float, max_cluster: int
) -> list[list[tuple[str, MemoryDoc]]]:
    fingerprints = {path: tokens(doc) for path, doc in members}
    taken: set[str] = set()
    out: list[list[tuple[str, MemoryDoc]]] = []

    for index, (path, doc) in enumerate(members):
        if path in taken:
            continue
        group = [(path, doc)]
        for other_path, other in members[index + 1 :]:
            if other_path in taken or len(group) >= max_cluster:
                continue
            if overlap(fingerprints[path], fingerprints[other_path]) >= threshold:
                group.append((other_path, other))
        if len(group) > 1:
            taken |= {member for member, _ in group}
            out.append(group)
    return out


def _density(group: Sequence[tuple[str, MemoryDoc]]) -> float:
    fingerprints = [tokens(doc) for _, doc in group]
    pairs = [
        overlap(fingerprints[i], fingerprints[j])
        for i in range(len(fingerprints))
        for j in range(i + 1, len(fingerprints))
    ]
    return sum(pairs) / len(pairs) if pairs else 0.0

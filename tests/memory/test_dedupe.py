"""Which memories are worth asking a model about."""

from __future__ import annotations

from kasa.memory.dedupe import clusters, overlap, tokens
from kasa.memory.document import MemoryDoc

SAME_A = "Priya Raman owns the deploy pipeline and runs the release checklist every Thursday."
SAME_B = "The deploy pipeline is owned by Priya Raman; she runs the release checklist weekly."
OTHER = "Coffee orders live in the kitchen spreadsheet, maintained by the office manager."


def fact(title: str, body: str, **fields: object) -> MemoryDoc:
    return MemoryDoc.new(type="fact", title=title, body=body, **fields)  # type: ignore[arg-type]


def group(*docs: MemoryDoc) -> list[tuple[str, MemoryDoc]]:
    return [(f"memory/facts/{n}.md", doc) for n, doc in enumerate(docs)]


def find(*docs: MemoryDoc, threshold: float = 0.45) -> list[list[str]]:
    found = clusters(group(*docs), threshold=threshold, max_cluster=4, max_clusters=10)
    return [[doc.frontmatter.title for _, doc in cluster] for cluster in found]


def test_two_memories_saying_one_thing_are_a_candidate() -> None:
    assert find(fact("Deploy ownership", SAME_A), fact("Deploys", SAME_B)) == [
        ["Deploy ownership", "Deploys"]
    ]


def test_two_memories_about_different_things_are_not() -> None:
    """The property that matters: a corpus of distinct memories costs nothing
    to check, because nothing in it is ever offered."""
    assert find(fact("Deploy ownership", SAME_A), fact("Coffee", OTHER)) == []


def test_a_person_and_a_project_are_never_compared() -> None:
    """They share vocabulary by construction, and merging across types would
    produce a file that is neither."""
    person = MemoryDoc.new(type="person", title="Priya", body=SAME_A)
    project = MemoryDoc.new(type="project", title="Deploys", body=SAME_B)

    assert clusters(group(person, project), threshold=0.45, max_cluster=4, max_clusters=10) == []


def test_two_audiences_are_never_compared() -> None:
    """The patch validator refuses to merge them, so proposing it would only
    spend a call to be told no."""
    assert (
        find(
            fact("Deploy ownership", SAME_A),
            fact("Deploys", SAME_B, visibility="private:U1"),
        )
        == []
    )


def test_three_files_about_one_thing_are_one_question() -> None:
    third = fact("Release checklist", SAME_A + " It is run by Priya Raman on the pipeline.")

    found = find(fact("Deploy ownership", SAME_A), fact("Deploys", SAME_B), third)

    assert len(found) == 1
    assert len(found[0]) == 3


def test_a_cluster_is_capped() -> None:
    """A corpus of near-identical notes must not become one unreadable file."""
    docs = [fact(f"Deploys {n}", SAME_A) for n in range(6)]

    found = clusters(group(*docs), threshold=0.45, max_cluster=2, max_clusters=10)

    assert all(len(cluster) <= 2 for cluster in found)


def test_a_memory_belongs_to_at_most_one_cluster() -> None:
    docs = [fact(f"Deploys {n}", SAME_A) for n in range(4)]

    found = clusters(group(*docs), threshold=0.45, max_cluster=2, max_clusters=10)

    seen = [path for cluster in found for path, _ in cluster]
    assert len(seen) == len(set(seen))


def test_the_densest_cluster_goes_first() -> None:
    """A bounded run should spend its calls on the likeliest duplicates."""
    identical = [fact("A", SAME_A), fact("B", SAME_A)]
    looser = [fact("C", SAME_A), fact("D", SAME_B)]

    found = clusters(group(*looser, *identical), threshold=0.45, max_cluster=2, max_clusters=10)

    assert [doc.frontmatter.title for _, doc in found[0]] == ["A", "B"]


def test_a_memory_made_entirely_of_noise_overlaps_with_nothing() -> None:
    """`the of and` against `the of and` is not evidence of anything, and a
    division by zero is not an answer to give the clusterer."""
    empty = tokens(fact("The", "the of and"))

    assert empty == frozenset()
    assert overlap(empty, tokens(fact("Deploys", SAME_A))) == 0.0
    assert overlap(empty, empty) == 0.0

"""`reorganize`: merges, splits, link repair, listings — one revertable commit."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from kasa.config import MemorySettings, ReorganizeSettings
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.types import ChatRequest, ChatResponse, Delta, Message, Usage
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.layout import INDEX_PATH
from kasa.memory.ltm import MemoryStore
from kasa.memory.manifest import Manifest
from kasa.runner.reorganize import Librarian
from kasa.store import Store

#: Two files that plainly say the same thing, and one that plainly does not.
DEPLOYS_A = "Priya Raman owns the deploy pipeline and runs the release checklist every Thursday."
DEPLOYS_B = "The deploy pipeline is owned by Priya Raman; she runs the release checklist weekly."
DISTINCT = "Coffee orders live in the kitchen spreadsheet, maintained by the office manager."


class Scripted:
    name = "scripted"
    model = "m"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def prompts(self) -> list[str]:
        return [req.messages[0].text for req in self.requests]

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        reply = self.replies.pop(0) if self.replies else "[]"
        return ChatResponse(
            message=Message.assistant(reply), stop_reason="end_turn", usage=Usage(), model="m"
        )

    def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    repo = tmp_path / "ltm"
    GitRepo.init(repo, branch="main")
    bootstrap(repo)
    Manifest.rebuild(repo)[0].save(repo)
    GitRepo.at(repo).commit("memory: bootstrap")
    return repo


def write_memory(clone: Path, doc: MemoryDoc, *, path: str | None = None) -> str:
    target = clone / (path or doc.suggested_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(doc.render())
    Manifest.rebuild(clone)[0].save(clone)
    GitRepo.at(clone).commit(f"memory: seed {doc.id}")
    return str(target.relative_to(clone))


def fact(title: str, body: str, **fields: Any) -> MemoryDoc:
    return MemoryDoc.new(type="fact", title=title, body=body, **fields)


def librarian_for(
    clone: Path, store: Store, provider: Scripted, *, policy: Any = None, **settings: Any
) -> Librarian:
    return Librarian(
        store,
        MemoryStore(GitRepo.at(clone), store, branch="main", push=False),
        ProviderRegistry({ModelRole.CHAT: [provider]}),
        settings=ReorganizeSettings(**settings),
        policy=policy,
    )


def merging(into: str, from_ids: list[str], body: str) -> str:
    return json.dumps([{"type": "merge", "into": into, "from_ids": from_ids, "body": body}])


def live_facts(clone: Path) -> list[str]:
    return sorted(p.name for p in (clone / "memory/facts").glob("*.md") if p.name != "README.md")


# -- merging near-duplicates -------------------------------------------------


async def test_a_corpus_with_known_duplicates_converges(clone: Path, store: Store) -> None:
    """The acceptance criterion. Two files saying one thing become one, the
    distinct fact is left alone, and a second pass finds nothing to do."""
    first = fact("Deploy ownership", DEPLOYS_A)
    second = fact("Deploys", DEPLOYS_B)
    distinct = fact("Coffee", DISTINCT)
    write_memory(clone, first)
    write_memory(clone, second)
    write_memory(clone, distinct)
    provider = Scripted(merging(first.id, [second.id], DEPLOYS_A))

    result = await librarian_for(clone, store, provider).run()

    assert result.merged == 1
    assert len(live_facts(clone)) == 2, "the merged one and the distinct one"
    assert DISTINCT in (clone / distinct.suggested_path()).read_text()

    second_pass = await librarian_for(clone, store, Scripted()).run()
    assert second_pass.merged == 0
    assert second_pass.summary() == "the corpus is already tidy"


async def test_nothing_distinct_is_ever_offered_for_merging(clone: Path, store: Store) -> None:
    """The filter never suggests a pair sharing no vocabulary, so a tidy corpus
    costs nothing to check."""
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    write_memory(clone, fact("Coffee", DISTINCT))
    provider = Scripted()

    await librarian_for(clone, store, provider).run()

    assert provider.requests == []


async def test_a_merged_source_is_archived_rather_than_deleted(clone: Path, store: Store) -> None:
    """Its id goes on resolving through the supersedes chain, so nothing that
    linked to it breaks."""
    first = fact("Deploy ownership", DEPLOYS_A)
    second = fact("Deploys", DEPLOYS_B)
    write_memory(clone, first)
    write_memory(clone, second)

    await librarian_for(clone, store, Scripted(merging(first.id, [second.id], DEPLOYS_A))).run()

    assert list((clone / "memory/archive").glob("*.md")), "the source is still in the repo"
    assert Manifest.load(clone).resolve(second.id) is not None, "and its id still resolves"


async def test_two_scopes_are_never_merged(clone: Path, store: Store) -> None:
    """The validator refuses it, so proposing it would only spend a call to be
    told no. The filter does not get that far."""
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    write_memory(clone, fact("Deploys", DEPLOYS_B, visibility="private:U1"))
    provider = Scripted()

    await librarian_for(clone, store, provider).run()

    assert provider.requests == []


async def test_a_merge_plan_that_reaches_outside_its_cluster_is_ignored(
    clone: Path, store: Store
) -> None:
    """The model was asked one question about two memories. A plan naming a
    third is a plan for a question nobody asked."""
    first = fact("Deploy ownership", DEPLOYS_A)
    second = fact("Deploys", DEPLOYS_B)
    elsewhere = fact("Coffee", DISTINCT)
    write_memory(clone, first)
    write_memory(clone, second)
    write_memory(clone, elsewhere)
    provider = Scripted(merging(first.id, [second.id, elsewhere.id], "everything"))
    before = GitRepo.at(clone).head()

    result = await librarian_for(clone, store, provider).run()

    assert result.merged == 0
    assert len(live_facts(clone)) == 3
    del before  # the listings still regenerate; the memories do not change


async def test_a_reply_that_is_not_a_plan_changes_no_memory(clone: Path, store: Store) -> None:
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    write_memory(clone, fact("Deploys", DEPLOYS_B))
    provider = Scripted("I have merged them for you and also deleted the rest.")

    result = await librarian_for(clone, store, provider).run()

    assert result.merged == 0
    assert len(live_facts(clone)) == 2


# -- splitting ---------------------------------------------------------------


async def test_an_oversized_memory_is_split_and_the_original_archived(
    clone: Path, store: Store
) -> None:
    big = fact("Everything", "One subject.\n\n" + ("filler words here. " * 400))
    write_memory(clone, big)
    parts = [
        MemoryDoc.new(type="fact", title="Part one", body="One subject."),
        MemoryDoc.new(type="fact", title="Part two", body="Another subject."),
    ]
    provider = Scripted(
        json.dumps(
            [
                *({"type": "create", "memory": part.model_dump(mode="json")} for part in parts),
                {"type": "archive", "id": big.id, "reason": "split"},
            ]
        )
    )

    result = await librarian_for(clone, store, provider, split_above_bytes=2_000).run()

    assert result.split == 1
    assert live_facts(clone) == ["part-one.md", "part-two.md"]
    assert Manifest.load(clone).resolve(big.id) is not None, "archived, still resolving"


async def test_a_split_that_is_not_creates_and_one_archive_is_ignored(
    clone: Path, store: Store
) -> None:
    big = fact("Everything", "One subject.\n\n" + ("filler words here. " * 400))
    write_memory(clone, big)
    provider = Scripted(json.dumps([{"type": "archive", "id": big.id, "reason": "gone"}]))

    result = await librarian_for(clone, store, provider, split_above_bytes=2_000).run()

    assert result.split == 0
    assert (clone / big.suggested_path()).exists(), "an archive on its own is not a split"


async def test_the_parts_of_a_private_memory_stay_private(clone: Path, store: Store) -> None:
    """A create is a new document; nothing else would catch a widened one."""
    big = fact(
        "Everything", "One subject.\n\n" + ("filler words here. " * 400), visibility="private:U1"
    )
    write_memory(clone, big)
    parts = [
        MemoryDoc.new(type="fact", title="Part one", body="One."),
        MemoryDoc.new(type="fact", title="Part two", body="Two."),
    ]
    provider = Scripted(
        json.dumps(
            [
                *({"type": "create", "memory": part.model_dump(mode="json")} for part in parts),
                {"type": "archive", "id": big.id, "reason": "split"},
            ]
        )
    )

    await librarian_for(clone, store, provider, split_above_bytes=2_000).run()

    written = MemoryDoc.parse((clone / "memory/facts/part-one.md").read_text())
    assert written.frontmatter.visibility == "private:U1"


# -- link repair -------------------------------------------------------------


async def test_a_link_to_a_merged_away_memory_is_rewritten_to_its_successor(
    clone: Path, store: Store
) -> None:
    """Six months of reorganizing leaves a corpus whose links all still work
    and none of which say what they point at."""
    old = fact("Old", "The old note.")
    successor = fact("New", "The new note.", supersedes=[old.id])
    pointer = fact("Pointer", f"See [[{old.id}]] for the details.")
    write_memory(clone, old, path="memory/archive/old.md")
    write_memory(clone, successor)
    pointer_path = write_memory(clone, pointer)

    result = await librarian_for(clone, store, Scripted()).run()

    assert result.repaired == 1
    body = (clone / pointer_path).read_text()
    assert f"[[{successor.id}]]" in body
    assert old.id not in body


async def test_a_repaired_link_keeps_its_display_text(clone: Path, store: Store) -> None:
    old = fact("Old", "The old note.")
    successor = fact("New", "The new note.", supersedes=[old.id])
    pointer = fact("Pointer", f"See [[{old.id}|the runbook]].")
    write_memory(clone, old, path="memory/archive/old.md")
    write_memory(clone, successor)
    pointer_path = write_memory(clone, pointer)

    await librarian_for(clone, store, Scripted()).run()

    assert f"[[{successor.id}|the runbook]]" in (clone / pointer_path).read_text()


async def test_a_link_the_manifest_cannot_account_for_is_reported_not_rewritten(
    clone: Path, store: Store, caplog: Any
) -> None:
    """Not repairable *through the manifest*. The bracketed text may be the
    only record that the thing ever existed."""
    pointer = fact("Pointer", "See [[mem_01ZZZZZZZZZZZZZZZZZZZZZZZZ]].")
    pointer_path = write_memory(clone, pointer)
    before = (clone / pointer_path).read_text()

    with caplog.at_level("WARNING", logger="kasa.runner.reorganize"):
        result = await librarian_for(clone, store, Scripted()).run()

    assert result.repaired == 0
    assert (clone / pointer_path).read_text() == before
    assert "resolves to nothing" in caplog.text


# -- the listings ------------------------------------------------------------


async def test_the_index_lists_what_is_in_the_repo(clone: Path, store: Store) -> None:
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    write_memory(clone, MemoryDoc.new(type="person", title="Priya Raman", body="Owns deploys."))

    await librarian_for(clone, store, Scripted()).run()

    root = (clone / INDEX_PATH).read_text()
    assert "Deploy ownership" in root and "Priya Raman" in root
    assert "## facts" in root and "## people" in root
    people = (clone / "memory/people/README.md").read_text()
    assert "[Priya Raman](priya-raman.md)" in people


async def test_a_listing_is_not_a_memory(clone: Path, store: Store) -> None:
    """It has no frontmatter and nobody claimed it. The manifest and the search
    index have to walk past it rather than report it as a broken file."""
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    await librarian_for(clone, store, Scripted()).run()

    manifest, problems = Manifest.rebuild(clone)

    assert problems == []
    assert all("README" not in entry.path for entry in manifest.memories.values())


async def test_the_index_reflects_the_merge_in_the_same_commit(clone: Path, store: Store) -> None:
    """Rendered from the manifest as it will be, not as it is: an index naming
    a memory this very commit archived is a week of a wrong front page."""
    first = fact("Deploy ownership", DEPLOYS_A)
    second = fact("Deploys", DEPLOYS_B)
    write_memory(clone, first)
    write_memory(clone, second)

    await librarian_for(clone, store, Scripted(merging(first.id, [second.id], DEPLOYS_A))).run()

    root = (clone / INDEX_PATH).read_text()
    assert "Deploy ownership" in root
    assert "- [Deploys](" not in root, "the archived source is not in the listing"


async def test_whatever_somebody_wrote_above_the_marker_survives(clone: Path, store: Store) -> None:
    """It is the front page of somebody's repository. A job that erased the
    paragraph they wrote at the top of it every week is a job they turn off."""
    index = clone / INDEX_PATH
    index.write_text(
        "# Our memory\n\nRead `people/` first.\n\n"
        "<!-- Kasa regenerates the listing below. Text above this comment is preserved. -->\n\n"
        "stale listing\n"
    )
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))

    await librarian_for(clone, store, Scripted()).run()

    root = index.read_text()
    assert "Read `people/` first." in root
    assert "stale listing" not in root
    assert "Deploy ownership" in root


async def test_a_tidy_week_writes_no_commit(clone: Path, store: Store) -> None:
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    await librarian_for(clone, store, Scripted()).run()
    settled = GitRepo.at(clone).head()

    result = await librarian_for(clone, store, Scripted()).run()

    assert GitRepo.at(clone).head() == settled
    assert result.sha is None


# -- bounds and reversibility ------------------------------------------------


async def test_every_reorganization_is_a_single_revertable_commit(
    clone: Path, store: Store
) -> None:
    """The other acceptance criterion, and the reason it is safe to let a model
    make these decisions at all."""
    first = fact("Deploy ownership", DEPLOYS_A)
    second = fact("Deploys", DEPLOYS_B)
    write_memory(clone, first)
    write_memory(clone, second)
    before = GitRepo.at(clone).head()
    repo = GitRepo.at(clone)

    await librarian_for(clone, store, Scripted(merging(first.id, [second.id], DEPLOYS_A))).run()

    commits = repo.run("log", "--format=%H", f"{before}..HEAD").split()
    assert len(commits) == 1
    assert "Kasa-Job: reorganize" in repo.run("log", "-1", "--format=%B")

    repo.run("revert", "--no-edit", commits[0])
    assert len(live_facts(clone)) == 2, "every decision put back"


async def test_a_run_stops_at_the_file_cap(clone: Path, store: Store) -> None:
    """A librarian pass that rewrote a hundred files a week would be
    unreviewable while technically remaining reversible."""
    old = fact("Old", "The old note.")
    successor = fact("New", "The new note.", supersedes=[old.id])
    write_memory(clone, old, path="memory/archive/old.md")
    write_memory(clone, successor)
    for n in range(5):
        write_memory(clone, fact(f"Pointer {n}", f"See [[{old.id}]] number {n}."))

    result = await librarian_for(
        clone, store, Scripted(), policy=MemorySettings(max_files_per_commit=2)
    ).run()

    assert result.repaired == 5, "all five needed repair"
    assert len([p for p in result.changed if "pointer" in p]) == 2, "two landed this week"


async def test_a_run_spends_at_most_its_operations(clone: Path, store: Store) -> None:
    for n in range(4):
        write_memory(clone, fact(f"Deploy ownership {n}", DEPLOYS_A + f" Note {n}."))
        write_memory(clone, fact(f"Deploys {n}", DEPLOYS_B + f" Note {n}."))
    provider = Scripted()

    await librarian_for(clone, store, provider, max_operations=2).run()

    assert len(provider.requests) == 2


async def test_the_corpus_travels_as_untrusted_data(clone: Path, store: Store) -> None:
    write_memory(clone, fact("Deploy ownership", DEPLOYS_A))
    write_memory(clone, fact("Deploys", DEPLOYS_B))
    provider = Scripted()

    await librarian_for(clone, store, provider).run()

    assert "KASA_UNTRUSTED_" in provider.prompts[0]
    assert provider.requests[0].tools == ()

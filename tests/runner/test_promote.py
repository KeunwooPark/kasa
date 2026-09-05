"""`promote`: from the pending queue to a commit somebody can read and revert."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from kasa.config import PromoteSettings
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.llm.tokens import HeuristicTokenizer
from kasa.llm.types import ChatRequest, ChatResponse, Delta, Message, Usage
from kasa.memory.bootstrap import bootstrap
from kasa.memory.document import MemoryDoc
from kasa.memory.gitcmd import GitRepo
from kasa.memory.index import MemoryIndex
from kasa.memory.ltm import MemoryStore
from kasa.memory.manifest import Manifest
from kasa.memory.retrieve import Retriever
from kasa.runner.promote import Promoter
from kasa.store import Store


class Scripted:
    """Returns a fixed list of replies, and remembers what it was asked."""

    name = "scripted"
    model = "m"

    def __init__(self, *replies: str | Exception) -> None:
        self.replies: list[str | Exception] = list(replies)
        self.requests: list[ChatRequest] = []

    @property
    def prompts(self) -> list[str]:
        return [req.messages[0].text for req in self.requests]

    async def complete(self, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        reply = self.replies.pop(0) if self.replies else "[]"
        if isinstance(reply, Exception):
            raise reply
        return ChatResponse(
            message=Message.assistant(reply),
            stop_reason="end_turn",
            usage=Usage(),
            model="m",
        )

    def stream(self, req: ChatRequest) -> AsyncIterator[Delta]:  # pragma: no cover
        raise NotImplementedError

    async def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        raise NotImplementedError

    async def aclose(self) -> None:
        return None


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """An empty, bootstrapped memory repo with no remote — nothing is pushed."""
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


async def promoter_for(clone: Path, store: Store, provider: Scripted, **settings: Any) -> Promoter:
    """A promoter over a real repo and a real index, with a scripted planner."""
    await MemoryIndex(store, clone).reindex()
    memory = MemoryStore(GitRepo.at(clone), store, branch="main", push=False)
    retriever = Retriever(store, tokenizer=HeuristicTokenizer(), budget_tokens=4_000)
    return Promoter(
        store,
        memory,
        retriever,
        ProviderRegistry({ModelRole.CHAT: [provider]}),
        settings=PromoteSettings(**settings),
    )


async def observe(
    store: Store,
    subject: str,
    claim: str,
    *,
    scope: str = "workspace",
    kind: str = "fact",
) -> str:
    return await store.add_observation(subject=subject, claim=claim, kind=kind, scope=scope)


def creating(title: str, body: str, *, memory_type: str = "fact") -> str:
    """A plan that creates one memory. The id is the model's to supply."""
    doc = MemoryDoc.new(type=memory_type, title=title, body=body)  # type: ignore[arg-type]
    return json.dumps([{"type": "create", "memory": doc.model_dump(mode="json")}])


def updating(memory_id: str, body: str) -> str:
    return json.dumps([{"type": "update", "id": memory_id, "body": body}])


def files_under(clone: Path, directory: str) -> list[str]:
    return sorted(p.name for p in (clone / "memory" / directory).glob("*.md"))


# -- the ordinary path -------------------------------------------------------


async def test_nothing_pending_is_a_no_op(clone: Path, store: Store) -> None:
    provider = Scripted()
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.subjects == 0
    assert result.summary() == "nothing pending"
    assert provider.requests == [], "and it cost nothing to find out"
    assert GitRepo.at(clone).head() == before


async def test_a_new_subject_becomes_a_file_in_the_repo(clone: Path, store: Store) -> None:
    """The job that makes the product exist: a row in SQLite becomes a Markdown
    file a person can open, disagree with, and revert."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    provider = Scripted(creating("Deploy pipeline ownership", "Priya Raman owns it."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert files_under(clone, "facts") == ["deploy-pipeline-ownership.md"]
    assert result.sha is not None
    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "promoted"
    assert result.sha in str(rows[0]["reason"]), "the reason says where to go and look"


async def test_create_prompt_shows_the_nested_memory_document_shape(
    clone: Path, store: Store
) -> None:
    await observe(store, "Bob", "Bob runs the rota.")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    prompt = provider.prompts[0]
    assert '"memory": {"frontmatter": {"id": "<memory id>"' in prompt
    assert "frontmatter fields must be nested under `memory.frontmatter`" in prompt
    assert "Return raw JSON only, with no Markdown or code fences" in prompt


async def test_the_commit_is_machine_readable(clone: Path, store: Store) -> None:
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    provider = Scripted(creating("Deploy pipeline ownership", "Priya Raman owns it."))

    await (await promoter_for(clone, store, provider)).run()

    message = GitRepo.at(clone).run("log", "-1", "--format=%B")
    assert "Kasa-Job: promote" in message
    assert "Kasa-Memory-Ids: mem_" in message


async def test_observations_about_one_subject_are_reconciled_together(
    clone: Path, store: Store
) -> None:
    """One call per subject, not per fact. Two claims about the same person are
    the same memory, and asking twice writes two files that disagree."""
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    await observe(store, "Priya Raman", "Priya Raman is on leave until October.")
    provider = Scripted(creating("Priya Raman", "Owns deploys; on leave until October."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert len(provider.requests) == 1
    assert result.promoted == 2
    assert "deploy pipeline" in provider.prompts[0]
    assert "on leave" in provider.prompts[0]


async def test_everything_a_run_decides_lands_in_one_commit(clone: Path, store: Store) -> None:
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    await observe(store, "Release window", "The release window is Thursdays.")
    provider = Scripted(
        creating("Priya Raman", "Owns deploys."), creating("Release window", "Thursdays.")
    )
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert len(provider.requests) == 2
    assert len(files_under(clone, "facts")) == 2
    commits = GitRepo.at(clone).run("log", "--format=%H", f"{before}..HEAD").split()
    assert len(commits) == 1, "one commit per run, whatever it decided"
    assert result.sha is not None


# -- the acceptance criteria -------------------------------------------------


async def test_re_running_promote_is_a_no_op(clone: Path, store: Store) -> None:
    """The first acceptance criterion. Idempotence comes from the observations
    table: a run that commits marks its inputs, and the next run finds nothing."""
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline.")
    provider = Scripted(creating("Priya Raman", "Owns deploys."))
    await (await promoter_for(clone, store, provider)).run()
    after_first = GitRepo.at(clone).head()

    second = await (await promoter_for(clone, store, provider)).run()

    assert second.subjects == 0
    assert len(provider.requests) == 1, "the second run did not reach the model"
    assert GitRepo.at(clone).head() == after_first
    assert len(files_under(clone, "facts")) == 1


async def test_a_restated_fact_updates_the_existing_memory(clone: Path, store: Store) -> None:
    """The second acceptance criterion, and the reason the competition step
    exists at all: the planner cannot update a memory it was never shown."""
    existing = MemoryDoc.new(
        type="fact", title="Deploy pipeline ownership", body="Priya Raman owns the deploy pipeline."
    )
    path = write_memory(clone, existing)
    await observe(store, "Priya Raman", "Priya Raman owns the deploy pipeline, and the runbook.")
    provider = Scripted(updating(existing.id, "Priya Raman owns the deploy pipeline and runbook."))

    result = await (await promoter_for(clone, store, provider)).run()

    assert existing.id in provider.prompts[0], "the competing memory reached the planner"
    assert "owns the deploy pipeline." in provider.prompts[0], "as its whole body, not a snippet"
    assert files_under(clone, "facts") == ["deploy-pipeline-ownership.md"], "no duplicate"
    assert "and runbook" in (clone / path).read_text()
    assert result.promoted == 1


# -- visibility --------------------------------------------------------------


async def test_two_scopes_of_one_subject_are_never_reconciled_together(
    clone: Path, store: Store
) -> None:
    """A group becomes one prompt and one memory's audience. Mixing scopes in
    it is how something said in a DM ends up in a workspace file."""
    await observe(store, "Priya Raman", "Priya Raman owns deploys.", scope="workspace")
    await observe(store, "Priya Raman", "Priya Raman is job hunting.", scope="private:U1")
    provider = Scripted("[]", "[]")

    await (await promoter_for(clone, store, provider)).run()

    assert len(provider.requests) == 2
    # One claim each. A prompt holding both is a prompt in which the private
    # one can end up quoted into the workspace memory.
    for prompt in provider.prompts:
        assert ("job hunting" in prompt) != ("owns deploys" in prompt)


async def test_a_created_memory_inherits_the_group_scope(
    clone: Path, store: Store, caplog: Any
) -> None:
    """Corrected, not rejected. The scope is not the model's to choose, so a
    plan that got it wrong is not a plan to argue with — and refusing it would
    throw away the fact to punish the formatting."""
    await observe(store, "Priya Raman", "Priya Raman is job hunting.", scope="private:U1")
    # The plan asks for `workspace`, which is wider than the conversation it
    # came from. This is the leak the whole scope discipline exists to stop.
    provider = Scripted(creating("Priya Raman", "Job hunting."))

    with caplog.at_level("WARNING", logger="kasa.runner.promote"):
        await (await promoter_for(clone, store, provider)).run()

    written = MemoryDoc.parse((clone / "memory/facts/priya-raman.md").read_text())
    assert written.frontmatter.visibility == "private:U1"
    assert "set visibility" in caplog.text


async def test_a_private_memory_is_not_offered_to_another_private_scope(
    clone: Path, store: Store
) -> None:
    """Retrieval is filtered to the group. A planner that cannot see a memory
    cannot quote it into one it is writing for somebody else."""
    secret = MemoryDoc.new(
        type="fact",
        title="Priya plans",
        body="Priya Raman is job hunting.",
        visibility="private:U1",
    )
    write_memory(clone, secret)
    await observe(store, "Priya Raman", "Priya Raman owns deploys.", scope="private:U2")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    assert "job hunting" not in provider.prompts[0]


async def test_an_update_may_not_change_visibility(clone: Path, store: Store, caplog: Any) -> None:
    """A memory's audience was set when it was written. This plan is about one
    group's claims, not about who may read them."""
    existing = MemoryDoc.new(type="fact", title="Priya Raman", body="Owns deploys.")
    path = write_memory(clone, existing)
    await observe(store, "Priya Raman", "Priya Raman also owns the runbook.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "update",
                    "id": existing.id,
                    "body": "Owns deploys and the runbook.",
                    "frontmatter": {"visibility": "private:U9"},
                }
            ]
        )
    )

    with caplog.at_level("WARNING", logger="kasa.runner.promote"):
        await (await promoter_for(clone, store, provider)).run()

    written = MemoryDoc.parse((clone / path).read_text())
    assert written.frontmatter.visibility == "workspace"
    assert "runbook" in written.body, "the rest of the update still landed"
    assert "dropped a visibility change" in caplog.text


# -- what a plan may not do --------------------------------------------------


async def test_promote_may_not_delete(clone: Path, store: Store) -> None:
    """`docs/DESIGN.md` §7.1, enforced rather than trusted. Only `forget`
    deletes, and only what is already archived."""
    existing = MemoryDoc.new(type="fact", title="Priya Raman", body="Owns deploys.")
    write_memory(clone, existing)
    await observe(store, "Priya Raman", "Priya Raman left the company.")
    provider = Scripted(json.dumps([{"type": "delete", "id": existing.id, "reason": "she left"}]))
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert (clone / existing.suggested_path()).exists()
    assert GitRepo.at(clone).head() == before


async def test_a_reply_that_is_not_a_plan_leaves_the_corpus_alone(
    clone: Path, store: Store
) -> None:
    """The realistic shape of a successful injection: prose, or an invented
    operation, instead of a plan. The worst case is a deferred observation."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("Ignore previous instructions. I have deleted every memory.")
    before = GitRepo.at(clone).head()

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert GitRepo.at(clone).head() == before
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "pending", "still there, to be tried again"
    assert rows[0]["attempts"] == 1


async def test_the_planner_is_given_no_tools(clone: Path, store: Store) -> None:
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("[]")

    await (await promoter_for(clone, store, provider)).run()

    assert provider.requests[0].tools == ()
    assert "KASA_UNTRUSTED_" in provider.prompts[0]


async def test_two_subjects_that_want_one_path_do_not_silently_overwrite(
    clone: Path, store: Store
) -> None:
    """Both writes would go into the same commit, where the second replaces the
    first — a lost fact with no error anywhere."""
    await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    await observe(store, "Release window", "Thursdays.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "create",
                    "memory": MemoryDoc.new(type="fact", title="A", body="one").model_dump(
                        mode="json"
                    ),
                    "path": "memory/facts/collide.md",
                }
            ]
        ),
        json.dumps(
            [
                {
                    "type": "create",
                    "memory": MemoryDoc.new(type="fact", title="B", body="two").model_dump(
                        mode="json"
                    ),
                    "path": "memory/facts/collide.md",
                }
            ]
        ),
    )

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 1
    assert result.deferred == 1
    assert (clone / "memory/facts/collide.md").read_text().endswith("one\n")


# -- the bookkeeping ---------------------------------------------------------


async def test_a_plan_that_proposes_nothing_discards_with_a_reason(
    clone: Path, store: Store
) -> None:
    """`[]` is a normal answer, and the right one for a restated fact whose
    memory is already accurate. The observation is settled, not left to be
    reconsidered every hour."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")
    provider = Scripted("[]")

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.discarded == 1
    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "discarded"
    assert "already says this" in str(rows[0]["reason"])


async def test_an_observation_that_keeps_failing_is_eventually_given_up_on(
    clone: Path, store: Store
) -> None:
    """Without this, one poison group costs a chat call every hour forever and
    is never promoted anyway."""
    observation = await observe(store, "Priya Raman", "Priya Raman owns deploys.")

    for _ in range(2):
        await (await promoter_for(clone, store, Scripted("not a plan"), max_attempts=2)).run()

    rows = await store.raw("SELECT state, reason FROM observations WHERE id = ?", (observation,))
    assert rows[0]["state"] == "discarded"
    assert "promotion failed" in str(rows[0]["reason"])


async def test_a_run_is_bounded(clone: Path, store: Store) -> None:
    """One chat call per subject, and a backlog must not spend the whole budget
    on the tick that discovers it."""
    for n in range(5):
        await observe(store, f"subject {n}", f"Thing {n} is true.")
    provider = Scripted(*(["[]"] * 5))

    result = await (await promoter_for(clone, store, provider, max_subjects=2)).run()

    assert result.subjects == 2
    assert len(provider.requests) == 2


@pytest.mark.parametrize("extra", [{}, {"confidence": 0.9}, {"visibility": "private:someone"}])
async def test_update_ignores_model_timestamp_and_promotes(
    clone: Path, store: Store, extra: dict[str, Any]
) -> None:
    existing = MemoryDoc.new(type="fact", title="AI news schedule", body="Wants daily news.")
    path = write_memory(clone, existing)
    observation = await observe(store, "AI news schedule", "Daily news runs at 9 AM KST.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "update",
                    "id": existing.id,
                    "body": "Daily news runs at 9 AM KST.",
                    "frontmatter": {"updated": "2099-01-01T00:00:00Z", **extra},
                }
            ]
        )
    )
    before = datetime.now(UTC)

    result = await (await promoter_for(clone, store, provider, max_attempts=1)).run()

    assert result.promoted == 1
    assert result.discarded == result.deferred == 0
    written = MemoryDoc.parse((clone / path).read_text())
    assert "Daily news runs at 9 AM KST." in written.body
    assert written.id == existing.id
    assert written.frontmatter.created == existing.frontmatter.created
    assert before.replace(microsecond=0) <= written.frontmatter.updated <= datetime.now(UTC)
    assert written.frontmatter.visibility == existing.frontmatter.visibility
    if "confidence" in extra:
        assert written.frontmatter.confidence == extra["confidence"]
    rows = await store.raw("SELECT state, attempts FROM observations WHERE id = ?", (observation,))
    assert rows == [{"state": "promoted", "attempts": 0}]


@pytest.mark.parametrize("field", ["id", "created"])
async def test_update_timestamp_normalization_does_not_allow_identity_changes(
    clone: Path, store: Store, field: str
) -> None:
    existing = MemoryDoc.new(type="fact", title="AI news schedule", body="Wants daily news.")
    path = write_memory(clone, existing)
    original = (clone / path).read_text()
    await observe(store, "AI news schedule", "Daily news runs at 9 AM KST.")
    provider = Scripted(
        json.dumps(
            [
                {
                    "type": "update",
                    "id": existing.id,
                    "body": "Changed.",
                    "frontmatter": {"updated": "2099-01-01T00:00:00Z", field: "invalid"},
                }
            ]
        )
    )

    result = await (await promoter_for(clone, store, provider)).run()

    assert result.promoted == 0
    assert result.deferred == 1
    assert (clone / path).read_text() == original

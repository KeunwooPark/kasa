"""`promote`: move distilled knowledge from SQLite into the git repo.

The job that makes the product exist. Everything before it accumulates
candidate facts in a database nobody reads; this is what turns them into files
a person can open, disagree with, and revert.

The shape of one run:

1. Read the pending observations and group them by `(subject, scope)`. The pair
   rather than the subject, because two visibility scopes must never meet
   inside one prompt — a group is reconciled as a unit, and the unit inherits
   one audience.
2. For each group, retrieve the memories already in the corpus that compete
   with it. This is the step that makes a restated fact an *update* instead of
   a second file saying the same thing.
3. Ask the chat model for a typed patch plan over that group.
4. Compile every accepted plan and apply them as **one commit**.
5. Mark the observations promoted or discarded, with the reason recorded.

Three rules it is built around.

**The model proposes; deterministic code disposes.** The reply is a JSON array
of patches and nothing else — it cannot mark an observation, choose a path, or
touch git. `promote` may not emit `Delete` at all, which `PatchCompiler`
enforces rather than trusts (#13); only `forget` deletes, and only what is
already archived.

**Visibility is inherited.** Every group carries one scope, retrieval is
filtered to it, and a plan that sets anything else on a document it creates is
corrected to the group's scope before compiling — loudly, because a model
getting that wrong is worth knowing about, and rejecting the plan over it would
lose the fact instead.

**Re-running is a no-op.** Idempotence comes from the observations table: a run
that commits marks its inputs `promoted`, and the next run finds nothing
pending. A crash between the commit and the marking re-promotes, and the model
sees its own memory in the competition and updates it rather than duplicating —
which is why the competition step is not only about quality.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kasa.config import MemorySettings, PromoteSettings
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.memory.consolidate import ConsolidationInput, build_request, decode_plan
from kasa.memory.document import Frontmatter, MemoryDoc, is_visibility, new_memory_id
from kasa.memory.ltm import ApplyResult, Change, CommitMeta, MemoryStore, MemoryStoreError, Write
from kasa.memory.patch import (
    Create,
    MemoryPatch,
    Merge,
    PatchCompiler,
    PatchError,
    Supersede,
    Update,
)
from kasa.memory.retrieve import Retriever
from kasa.memory.schema import render_schema_md
from kasa.store import Store

log = logging.getLogger(__name__)

JOB = "promote"

#: Fresh ids offered to the model per group, on top of one per observation. A
#: `Create` needs an id that passes `Frontmatter._valid_id`, and a model asked
#: to invent a ULID produces something that fails validation often enough to
#: matter. Handing it valid ones costs nothing and removes the failure mode.
SPARE_IDS = 2

TASK = """Reconcile these candidate observations about one subject against what
long-term memory already says.

`channel_messages` holds the observations, one per line, numbered. Each is a
claim distilled from a conversation. `memory_files` holds the memories already
in the corpus that compete with them, keyed by path — these are what you are
reconciling against, and updating one of them is almost always better than
writing another file about the same thing.

Return a JSON array of patch objects. The allowed operations are:

- `{{"type": "create", "memory": <document>, "path": "memory/<dir>/<slug>.md"}}`
  — a subject the corpus says nothing about yet. `path` is optional; omit it
  and the conventional path for the document's `type` is used.
- `{{"type": "update", "id": "<memory id>", "body": "<full new body>",
  "frontmatter": {{...}}}}` — the corpus already covers this subject. `body`
  replaces the old body entirely, so write the whole thing, not the change.
  `frontmatter` carries only the fields you are changing.
- `{{"type": "merge", "into": "<memory id>", "from_ids": [...], "body": "..."}}`
  — two existing memories say the same thing. The sources are archived, not
  deleted, and their ids keep resolving.
- `{{"type": "supersede", "old_id": "<memory id>", "new": <document>}}` — the
  new claim contradicts an existing memory rather than extending it. The old
  one is archived and the new one records that it replaced it.

There is no delete. Nothing you return can remove a memory.

Return `[]` when the corpus already says everything these observations say.
That is a normal answer, and it is the right one for a restated fact whose
memory is already accurate.

Write for a person reading the file in a year. One claim per memory; split
rather than append when a file starts covering two subjects.

Use these ids for any memory you create, each at most once:
{ids}

Set `created` and `updated` to {now} on anything you create.
Set `visibility` to exactly `{scope}` on anything you create. Every observation
here came from a conversation with that audience, and a memory may not be
written to a wider one.

{schema}"""


@dataclass(frozen=True, slots=True)
class Group:
    """The observations about one subject, from one audience."""

    subject: str
    scope: str
    rows: list[dict[str, Any]]

    @property
    def ids(self) -> list[str]:
        return [str(row["id"]) for row in self.rows]

    def claims(self) -> list[str]:
        return [
            f"[{n}] ({row['kind']}, confidence {row['confidence']}) {row['claim']}"
            for n, row in enumerate(self.rows, start=1)
        ]


@dataclass(slots=True)
class Promotion:
    """What one run did, in the terms a person would ask about it."""

    subjects: int = 0
    promoted: int = 0
    discarded: int = 0
    deferred: int = 0
    changed: list[str] = field(default_factory=list)
    sha: str | None = None
    pull_request_url: str | None = None

    def summary(self) -> str:
        if not self.subjects:
            return "nothing pending"
        parts = [f"{self.subjects} subject(s)", f"{self.promoted} observation(s) promoted"]
        if self.discarded:
            parts.append(f"{self.discarded} discarded")
        if self.deferred:
            parts.append(f"{self.deferred} left pending")
        if self.changed:
            parts.append(f"{len(self.changed)} file(s) in {self.sha or 'no commit'}")
        return ", ".join(parts)


class Promoter:
    """One `promote` run, from the pending queue to a commit."""

    def __init__(
        self,
        store: Store,
        memory: MemoryStore,
        retriever: Retriever,
        registry: ProviderRegistry,
        *,
        policy: MemorySettings | None = None,
        settings: PromoteSettings | None = None,
        job_id: str | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._retriever = retriever
        self._registry = registry
        self._policy = policy or MemorySettings()
        self._settings = settings or PromoteSettings()
        self._job_id = job_id

    async def run(self) -> Promotion:
        rows = await self._store.pending_observations(self._settings.max_observations)
        groups = _group(rows)[: self._settings.max_subjects]
        if not groups:
            return Promotion()

        manifest = self._memory.manifest()
        changes: list[Change] = []
        claimed: set[str] = set()
        promoted: list[Group] = []
        discarded: list[tuple[Group, str]] = []
        deferred: list[tuple[Group, str]] = []
        touched_ids: list[str] = []

        for group in groups:
            plan, problem = await self._plan(group)
            if problem is not None:
                deferred.append((group, problem))
                continue
            if not plan:
                discarded.append((group, "the corpus already says this; no change was proposed"))
                continue

            compiler = PatchCompiler(self._memory.path, manifest, policy=self._policy)
            try:
                compiled = compiler.compile(plan, job=JOB)
            except PatchError as exc:
                deferred.append((group, str(exc)))
                continue

            # Each group compiles against the corpus as it stands, not against
            # what the groups before it proposed, so two of them can land on
            # one path — and both writes would go into the same commit, where
            # the second silently replaces the first. Rare, and a lost fact
            # with no error is not a thing to leave to chance. The loser waits
            # for the next run, by which point the winner is on disk and shows
            # up as competition.
            paths = {c.path for c in compiled if isinstance(c, Write)}
            if overlap := paths & claimed:
                deferred.append((group, f"another subject in this run already writes {overlap}"))
                continue

            claimed |= paths
            changes.extend(compiled)
            promoted.append(group)
            touched_ids.extend(_memory_ids(plan))

        result = await self._commit(changes, promoted, touched_ids)
        return await self._record(result, promoted, discarded, deferred, len(groups))

    # -- one group -----------------------------------------------------------

    async def _plan(self, group: Group) -> tuple[list[MemoryPatch], str | None]:
        """The model's plan for one group, or why there is not one."""
        if not is_visibility(group.scope):
            # The scope came off a session row, so this is a bug upstream
            # rather than anything the model did. Writing the memory anyway
            # would put an unparseable `visibility` in the corpus, and every
            # later read of that file fails.
            return [], f"{group.scope!r} is not a visibility scope a memory may carry"
        competing = await self._competing(group)
        request = build_request(
            job=JOB,
            task=TASK.format(
                ids="\n".join(f"  {i}" for i in _fresh_ids(len(group.rows) + SPARE_IDS)),
                now=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                scope=group.scope,
                schema=render_schema_md(),
            ),
            content=ConsolidationInput(channel_messages=group.claims(), memory_files=competing),
        )
        response = await self._registry.complete(ModelRole.CHAT, request, tag="promote.plan")
        try:
            plan = decode_plan(response.text, job=JOB)
        except PatchError as exc:
            return [], str(exc)
        return _with_scope(plan, group.scope), None

    async def _competing(self, group: Group) -> dict[str, str]:
        """The memories already in the corpus that this group is about.

        Scoped to the group, so a private observation is never reconciled
        against — or into — a memory it is not allowed to see. Read as whole
        files rather than as the retriever's snippets, because an `Update`
        rewrites a body and a model shown half of one would write half of one
        back.
        """
        query = " ".join([group.subject, *(str(row["claim"]) for row in group.rows)])
        retrieval = await self._retriever.retrieve(
            query,
            scope=group.scope,
            include_pinned=False,
            limit=self._settings.competing_memories,
        )
        manifest = self._memory.manifest()
        files: dict[str, str] = {}
        for memory_id in dict.fromkeys(retrieval.memory_ids):
            entry = manifest.resolve(memory_id)
            if entry is None or entry.path in files:
                continue
            try:
                content = self._memory.read(entry.path)
            except MemoryStoreError as exc:
                log.warning("promote: could not read competing memory %s: %s", entry.path, exc)
                continue
            if len(content) > self._settings.max_memory_chars:
                # Skipped, not truncated. See `PromoteSettings.max_memory_chars`.
                log.warning(
                    "promote: %s is %d chars and was not offered as competition; "
                    "reorganize should be splitting it",
                    entry.path,
                    len(content),
                )
                continue
            files[entry.path] = content
            if len(files) >= self._settings.competing_memories:
                break
        return files

    # -- the commit, and the bookkeeping -------------------------------------

    async def _commit(
        self, changes: Sequence[Change], promoted: Sequence[Group], memory_ids: Sequence[str]
    ) -> ApplyResult:
        if not changes:
            return ApplyResult()
        observations = sum(len(group.rows) for group in promoted)
        subjects = ", ".join(group.subject for group in promoted[:3])
        if len(promoted) > 3:
            subjects += f" and {len(promoted) - 3} more"
        return await self._memory.apply(
            changes,
            CommitMeta(
                summary=f"promote {observations} observation(s) about {subjects}",
                job=JOB,
                job_id=self._job_id,
                memory_ids=list(dict.fromkeys(memory_ids)),
            ),
        )

    async def _record(
        self,
        result: ApplyResult,
        promoted: Sequence[Group],
        discarded: Sequence[tuple[Group, str]],
        deferred: Sequence[tuple[Group, str]],
        subjects: int,
    ) -> Promotion:
        """Move every observation this run decided about out of `pending`.

        After the commit, deliberately. A row marked `promoted` before the
        write lands is a fact that was never written and will never be
        proposed again; the other order re-proposes at worst, and the model
        sees the memory it already wrote and updates it.
        """
        landed = result.sha is not None or result.pull_request_url is not None
        where = result.pull_request_url or result.sha or "no commit"
        outcome = Promotion(
            subjects=subjects,
            changed=list(result.changed),
            sha=result.sha,
            pull_request_url=result.pull_request_url,
        )

        waiting = list(deferred)
        for group in promoted:
            if landed:
                outcome.promoted += await self._store.resolve_observations(
                    group.ids, state="promoted", reason=f"written to long-term memory in {where}"
                )
            else:
                # The plan compiled and the write did not land, so nothing was
                # promoted. Deferred like a rejected plan, attempt cap
                # included: a group that compiles to a commit git decides is
                # empty would otherwise be re-planned every hour forever.
                waiting.append((group, "the plan compiled but nothing was committed"))

        for group, reason in discarded:
            outcome.discarded += await self._store.resolve_observations(
                group.ids, state="discarded", reason=reason
            )

        for group, reason in waiting:
            await self._store.note_observation_attempt(group.ids)
            exhausted = [
                row for row in group.rows if int(row["attempts"]) + 1 >= self._settings.max_attempts
            ]
            if exhausted:
                log.warning(
                    "promote: giving up on %d observation(s) about %r after %d attempt(s): %s",
                    len(exhausted),
                    group.subject,
                    self._settings.max_attempts,
                    reason,
                )
                outcome.discarded += await self._store.resolve_observations(
                    [str(row["id"]) for row in exhausted],
                    state="discarded",
                    reason=f"promotion failed {self._settings.max_attempts} time(s): {reason}",
                )
            else:
                log.info("promote: deferring %r to the next run: %s", group.subject, reason)
            outcome.deferred += len(group.rows) - len(exhausted)

        return outcome


# -- helpers -----------------------------------------------------------------


def _group(rows: Sequence[dict[str, Any]]) -> list[Group]:
    """Pending observations, gathered by the unit `promote` reconciles.

    `(subject, scope)` and not `subject`: a group becomes one prompt and one
    memory's audience, and mixing two scopes in it is how something said in a
    DM ends up in a workspace file.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["subject"]), str(row["scope"])), []).append(row)
    return [
        Group(subject=subject, scope=scope, rows=members)
        for (subject, scope), members in grouped.items()
    ]


def _fresh_ids(count: int) -> list[str]:
    return [new_memory_id() for _ in range(count)]


def _memory_ids(plan: Sequence[MemoryPatch]) -> list[str]:
    ids = []
    for patch in plan:
        match patch:
            case Create():
                ids.append(patch.memory.id)
            case Update():
                ids.append(patch.id)
            case Merge():
                ids.append(patch.into)
            case Supersede():
                ids.append(patch.new.id)
            case _:
                pass
    return ids


def _with_scope(plan: Sequence[MemoryPatch], scope: str) -> list[MemoryPatch]:
    """Force every document the plan creates to the group's audience.

    Corrected rather than rejected. The scope is not the model's to choose, so
    a plan that got it wrong is not a plan to argue with — and rejecting it
    would throw away the fact to punish the formatting. `PatchCompiler` still
    refuses to *widen* an existing memory, which is the case this cannot reach.
    """
    corrected: list[MemoryPatch] = []
    for patch in plan:
        match patch:
            case Create():
                corrected.append(patch.model_copy(update={"memory": _scoped(patch.memory, scope)}))
            case Supersede():
                corrected.append(patch.model_copy(update={"new": _scoped(patch.new, scope)}))
            case Update() if "visibility" in patch.frontmatter:
                # An update may not change visibility at all: the memory's
                # audience was set when it was written, and this plan is about
                # one group's claims, not about who may read it.
                log.warning("promote: dropped a visibility change from an update to %s", patch.id)
                corrected.append(
                    patch.model_copy(
                        update={
                            "frontmatter": {
                                k: v for k, v in patch.frontmatter.items() if k != "visibility"
                            }
                        }
                    )
                )
            case _:
                corrected.append(patch)
    return corrected


def _scoped(doc: MemoryDoc, scope: str) -> MemoryDoc:
    if doc.frontmatter.visibility == scope:
        return doc
    log.warning(
        "promote: a plan set visibility %r on a new memory; the observations came from %r",
        doc.frontmatter.visibility,
        scope,
    )
    # Re-validated rather than `model_copy`d in: `model_copy` does not run the
    # validators, and an unparseable `visibility` written to a file is one
    # every later read of that file fails on. `_plan` has already checked the
    # scope, so this cannot raise; it is here because "cannot" is a property of
    # today's callers.
    fields = doc.frontmatter.model_dump() | {"visibility": scope}
    return doc.model_copy(update={"frontmatter": Frontmatter.model_validate(fields)})

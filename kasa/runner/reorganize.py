"""`reorganize`: the weekly librarian pass. Bounded and reversible by construction.

`promote` writes what it is told and `reflect` scores what is there. Neither of
them moves anything, so the shape of the corpus is whatever a year of
one-file-at-a-time decisions left behind: two files about one person, one file
that grew into three subjects, links pointing at ids that were merged away, and
directories nobody can browse.

Four passes, in the order that makes each one see the last one's work:

1. **Repair links.** Deterministic, through the manifest. A memory that was
   merged away keeps its id and its successor records it, so the link still
   resolves — this makes the file say so.
2. **Merge near-duplicates.** Candidate clusters are found deterministically by
   token overlap; whether two memories actually say the same thing is a
   question for a model, asked over the whole of both documents.
3. **Split what grew too big.** Same shape: the candidate is arithmetic, the
   judgement is a model's, and the plan it returns goes through the validator
   like every other plan.
4. **Regenerate the listings.** `memory/README.md` and one per directory, from
   the manifest as it will be *after* the plan.

Two properties this is built to have.

**Bounded.** Model calls per run, memories per cluster, and files per commit,
each capped. A librarian pass that rewrote a thousand files would be
unreviewable, and the whole value of the corpus being in git is that a person
can read what changed.

**Reversible.** One commit per run. `git revert` puts every one of these
decisions back, which is the only reason it is safe to let a model make them —
and it is why nothing here deletes: a merge archives its sources, and their ids
go on resolving through the `supersedes` chain.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

from kasa.config import MemorySettings, ReorganizeSettings
from kasa.llm.registry import ModelRole, ProviderRegistry
from kasa.memory.consolidate import ConsolidationInput, build_request, decode_plan
from kasa.memory.dedupe import clusters
from kasa.memory.document import MemoryDoc, MemoryError_, new_memory_id
from kasa.memory.layout import ARCHIVE_DIR, INDEX_PATH
from kasa.memory.links import Broken, repair
from kasa.memory.ltm import ApplyResult, Change, CommitMeta, MemoryStore, MemoryStoreError, Write
from kasa.memory.manifest import Manifest
from kasa.memory.pages import preamble_of, render_pages
from kasa.memory.patch import (
    Archive,
    Create,
    MemoryPatch,
    Merge,
    PatchCompiler,
    PatchError,
    Update,
)
from kasa.memory.schema import render_schema_md
from kasa.store import Store

log = logging.getLogger(__name__)

JOB = "reorganize"

MERGE_TASK = """These memories overlap enough that they may be describing the
same thing. Decide whether they are, and if so merge them into one.

`memory_files` holds each of them in full, keyed by path.

Return a JSON array holding **at most one** patch, and only of this form:

`{{"type": "merge", "into": "<the id to keep>", "from_ids": [...the others...],
"body": "<the whole merged body>"}}`

The sources are archived rather than deleted and their ids go on resolving, so
nothing that links to them breaks.

Keep the id of whichever memory is the better home for the merged claim — the
one with the more accurate title, or the older one when there is nothing to
choose between them. `body` replaces it entirely, so write the whole thing.

**No fact may be lost.** Every distinct claim in every source has to appear in
the merged body. If merging them would mean dropping something, or if they are
about genuinely different things that happen to share vocabulary, return `[]`.
`[]` is the right answer far more often than a merge is.

{schema}"""

SPLIT_TASK = """This memory has grown to cover more than one subject. One claim
per memory is the rule the corpus is written to, and this file has stopped
following it.

`memory_files` holds it in full.

Return a JSON array of `create` patches — one per subject — followed by a
single `archive` of the original:

`{{"type": "create", "memory": <document>}}`
`{{"type": "archive", "id": "{original}", "reason": "split"}}`

Every part of the original body must end up in exactly one of the new
memories; nothing may be dropped and nothing invented. Give each new memory a
title that says what it is about, and link them to each other with
`[[<id>]]` where the prose refers across.

If the file is long but genuinely about one thing, return `[]`. A long memory
about one subject is not a problem.

Use these ids, each at most once:
{ids}

Set `created` and `updated` to {now}, and `visibility` to exactly `{scope}`.

{schema}"""


@dataclass(slots=True)
class Reorganization:
    """What one week's pass did."""

    repaired: int = 0
    merged: int = 0
    split: int = 0
    broken: list[Broken] = field(default_factory=list)
    pages: int = 0
    changed: list[str] = field(default_factory=list)
    sha: str | None = None
    pull_request_url: str | None = None

    def summary(self) -> str:
        parts = []
        if self.merged:
            parts.append(f"{self.merged} merge(s)")
        if self.split:
            parts.append(f"{self.split} split(s)")
        if self.repaired:
            parts.append(f"{self.repaired} link repair(s)")
        if self.pages:
            parts.append(f"{self.pages} listing(s)")
        if not parts:
            return "the corpus is already tidy"
        where = self.pull_request_url or self.sha or "nothing committed"
        return f"{', '.join(parts)} in {where}"


class Librarian:
    """One `reorganize` run: four passes and a single commit."""

    def __init__(
        self,
        store: Store,
        memory: MemoryStore,
        registry: ProviderRegistry,
        *,
        settings: ReorganizeSettings | None = None,
        policy: MemorySettings | None = None,
        job_id: str | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._registry = registry
        self._settings = settings or ReorganizeSettings()
        self._policy = policy or MemorySettings()
        self._job_id = job_id

    async def run(self) -> Reorganization:
        manifest = self._memory.manifest()
        corpus = self._read(manifest)
        outcome = Reorganization()

        patches, outcome.broken = self._repairs(corpus, manifest)
        outcome.repaired = len(patches)
        calls = 0

        for group in clusters(
            corpus,
            threshold=self._settings.duplicate_overlap,
            max_cluster=self._settings.max_cluster,
            max_clusters=self._settings.max_operations,
        ):
            if calls >= self._settings.max_operations:
                break
            calls += 1
            if merge := await self._merge(group):
                patches.append(merge)
                outcome.merged += 1

        for path, doc in self._oversized(corpus):
            if calls >= self._settings.max_operations:
                break
            calls += 1
            if split := await self._split(path, doc):
                patches.extend(split)
                outcome.split += 1

        changes, projected = self._compile(patches, manifest)
        pages = self._pages(projected)
        outcome.pages = len(pages)
        result = await self._apply(changes, pages, outcome)

        outcome.changed = list(result.changed)
        outcome.sha = result.sha
        outcome.pull_request_url = result.pull_request_url
        for broken in outcome.broken:
            # Not repairable through the manifest — the id resolves to nothing
            # at all. Reported rather than unlinked: the bracketed text may be
            # the only record that the thing ever existed, and editing somebody
            # else's prose to tidy a pointer is not this job's to do.
            log.warning(
                "reorganize: %s links to [[%s]], which resolves to nothing",
                broken.path,
                broken.target,
            )
        return outcome

    # -- the four passes -----------------------------------------------------

    def _repairs(
        self, corpus: Sequence[tuple[str, MemoryDoc]], manifest: Manifest
    ) -> tuple[list[MemoryPatch], list[Broken]]:
        patches: list[MemoryPatch] = []
        broken: list[Broken] = []
        for path, doc in corpus:
            repaired, unresolvable = repair(doc, path, manifest)
            broken.extend(unresolvable)
            if repaired is None:
                continue
            log.info(
                "reorganize: %s links to %s, which is now %s",
                path,
                ", ".join(repaired.rewrites),
                ", ".join(repaired.rewrites.values()),
            )
            patches.append(Update(id=repaired.memory_id, body=repaired.body))
        return patches, broken

    async def _merge(self, group: Sequence[tuple[str, MemoryDoc]]) -> MemoryPatch | None:
        ids = {doc.id for _, doc in group}
        plan = await self._plan(
            MERGE_TASK.format(schema=render_schema_md()),
            {path: doc.render() for path, doc in group},
        )
        if plan is None or not plan:
            return None
        patch = plan[0]
        # The model was asked for one merge over these memories. Anything else
        # — a create, a merge pulling in an id from elsewhere in the corpus, a
        # second patch — is a plan for a question nobody asked, and the cheap
        # thing to do with it is nothing.
        if len(plan) != 1 or not isinstance(patch, Merge):
            log.warning("reorganize: a merge plan proposed %d patch(es); ignored", len(plan))
            return None
        if {patch.into, *patch.from_ids} != ids:
            log.warning(
                "reorganize: a merge plan named memories outside the cluster it was given; ignored"
            )
            return None
        return patch

    async def _split(self, path: str, doc: MemoryDoc) -> list[MemoryPatch] | None:
        plan = await self._plan(
            SPLIT_TASK.format(
                original=doc.id,
                ids="\n".join(f"  {i}" for i in (new_memory_id() for _ in range(4))),
                now=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                scope=doc.frontmatter.visibility,
                schema=render_schema_md(),
            ),
            {path: doc.render()},
        )
        if plan is None or not plan:
            return None

        creates = [p for p in plan if isinstance(p, Create)]
        archives = [p for p in plan if isinstance(p, Archive) and p.id == doc.id]
        if len(creates) < 2 or len(archives) != 1 or len(creates) + 1 != len(plan):
            log.warning(
                "reorganize: a split plan was not two-or-more creates and one archive of %s; "
                "ignored",
                doc.id,
            )
            return None
        # The parts inherit the audience of the memory they came out of. The
        # validator refuses a widened *update*; a create is a new document, and
        # nothing else would catch it.
        scoped: list[MemoryPatch] = [
            create.model_copy(update={"memory": _scoped(create.memory, doc.frontmatter.visibility)})
            for create in creates
        ]
        return [*scoped, *archives]

    def _pages(self, manifest: Manifest) -> dict[str, str]:
        """The listings that would differ from what is on disk.

        Only the ones that changed, so a quiet week writes no commit at all
        rather than one that rewrites six identical files.
        """
        rendered = render_pages(manifest, preamble=preamble_of(self._read_page(INDEX_PATH)))
        return {
            path: content for path, content in rendered.items() if self._read_page(path) != content
        }

    def _read_page(self, path: str) -> str:
        try:
            return self._memory.read(path)
        except MemoryStoreError:
            return ""

    # -- the model call ------------------------------------------------------

    async def _plan(self, task: str, files: dict[str, str]) -> list[MemoryPatch] | None:
        """One question, or None if the answer was not a plan."""
        request = build_request(job=JOB, task=task, content=ConsolidationInput(memory_files=files))
        response = await self._registry.complete(ModelRole.CHAT, request, tag="reorganize.plan")
        try:
            return decode_plan(response.text, job=JOB)
        except PatchError as exc:
            log.warning("reorganize: %s", exc)
            return None

    # -- compiling and committing --------------------------------------------

    def _compile(
        self, patches: Sequence[MemoryPatch], manifest: Manifest
    ) -> tuple[list[Change], Manifest]:
        """Everything the plan means, and the corpus it leaves behind.

        Compiled one patch at a time rather than all at once. They come from
        four independent passes over one corpus, so one bad plan among them
        should cost its own operation and not the week's — and the file cap is
        checked here, as they accumulate, rather than by rejecting the whole
        plan after the last one crosses it.
        """
        changes: list[Change] = []
        claimed: set[str] = set()
        projected = manifest.model_copy(deep=True)

        for patch in patches:
            compiler = PatchCompiler(self._memory.path, projected, policy=self._policy)
            try:
                compiled = compiler.compile([patch], job=JOB)
            except PatchError as exc:
                log.warning("reorganize: a patch was rejected and dropped: %s", exc)
                continue
            paths = {change.path for change in compiled}
            if overlap := paths & claimed:
                log.info("reorganize: leaving %s to next week; already rewritten", overlap)
                continue
            if len(claimed | paths) > self._policy.max_files_per_commit:
                log.info(
                    "reorganize: stopping at %d file(s); the rest waits for next week",
                    len(claimed),
                )
                break
            claimed |= paths
            changes.extend(compiled)
            projected = _project(projected, compiled)
        return changes, projected

    async def _apply(
        self, changes: Sequence[Change], pages: dict[str, str], outcome: Reorganization
    ) -> ApplyResult:
        writes = [*changes, *(Write(path, content) for path, content in pages.items())]
        if not writes:
            return ApplyResult()
        return await self._memory.apply(
            writes,
            CommitMeta(summary=_headline(outcome), job=JOB, job_id=self._job_id),
        )

    # -- reading the corpus --------------------------------------------------

    def _read(self, manifest: Manifest) -> list[tuple[str, MemoryDoc]]:
        """Every live memory, parsed. The archive is deliberately absent: it is
        where merged sources go, and clustering over it would propose merging a
        memory with the copy of itself it just replaced."""
        corpus: list[tuple[str, MemoryDoc]] = []
        for entry in manifest.memories.values():
            if entry.path.startswith(f"{ARCHIVE_DIR}/"):
                continue
            try:
                raw = self._memory.read(entry.path)
                corpus.append((entry.path, MemoryDoc.parse(raw, source=entry.path)))
            except (MemoryStoreError, MemoryError_) as exc:
                log.warning("reorganize: skipping %s: %s", entry.path, exc)
        corpus.sort(key=lambda pair: pair[0])
        return corpus[: self._settings.max_candidates]

    def _oversized(self, corpus: Sequence[tuple[str, MemoryDoc]]) -> list[tuple[str, MemoryDoc]]:
        big = [
            (path, doc)
            for path, doc in corpus
            if len(doc.render().encode()) > self._settings.split_above_bytes
        ]
        # Largest first: the worst offender is the one most worth a call.
        big.sort(key=lambda pair: len(pair[1].render()), reverse=True)
        return big


# -- helpers -----------------------------------------------------------------


def _project(manifest: Manifest, changes: Sequence[Change]) -> Manifest:
    """The manifest as it will be once `changes` are on disk.

    The listings are regenerated from this rather than from what is currently
    there, so a run that merges two memories does not publish an index naming
    both of them until the following week.
    """
    after = manifest.model_copy(deep=True)
    for change in changes:
        if isinstance(change, Write):
            try:
                doc = MemoryDoc.parse(change.content, source=change.path)
            except MemoryError_:
                continue
            after.record(change.path, doc, checksum="pending")
        elif (memory_id := after.id_at(change.path)) is not None:
            after.forget(memory_id)
    return after


def _scoped(doc: MemoryDoc, scope: str) -> MemoryDoc:
    if doc.frontmatter.visibility == scope:
        return doc
    log.warning(
        "reorganize: a split set visibility %r on a part of a %r memory",
        doc.frontmatter.visibility,
        scope,
    )
    fields = doc.frontmatter.model_dump() | {"visibility": scope}
    return doc.model_copy(update={"frontmatter": type(doc.frontmatter).model_validate(fields)})


def _headline(outcome: Reorganization) -> str:
    parts = []
    if outcome.merged:
        parts.append(f"merge {outcome.merged} duplicate(s)")
    if outcome.split:
        parts.append(f"split {outcome.split} oversized memor{'y' if outcome.split == 1 else 'ies'}")
    if outcome.repaired:
        parts.append(f"repair {outcome.repaired} link(s)")
    return f"reorganize: {', '.join(parts) or 'regenerate the listings'}"

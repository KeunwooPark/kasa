"""`forget`: bounded, conservative forgetting. The most cautious job here.

Everything else in Kasa adds or rearranges. This is the one job that removes,
and the design (§6.2) is built around the observation that a memory wrongly
kept costs a little disk and a slightly worse search, while a memory wrongly
removed costs the thing the product exists to do.

**There is no model in it.** The design table lists a utility model beside this
row, and it turned out not to need one: salience is a number, age is a number,
`pinned` is a boolean, and "is anything still linking to this" is a graph
question. Every input to every decision here is already in the corpus, so the
most dangerous job in the system is the one with no judgement in it at all —
which means it can be read, and argued with, and tested exhaustively.

**Two transitions, and the first is mandatory.** A memory whose salience has
fallen below the threshold moves to `memory/archive/`. A memory that has been
sitting in `memory/archive/` past the grace period is `git rm`'d. There is no
path from the first state to the last, and the patch validator refuses a delete
of anything not already archived, so neither this job nor a bug in it can
remove something in one step.

**Four things it never touches**, checked here before anything is proposed and
again by the validator afterwards:

- anything `pinned`
- anything younger than the retention floor
- anything a *live* memory still links to
- more than `max_per_run` files in one week

The last removal is still recoverable: delete is `git rm`, the blob stays in
history, and `forget` runs supervised by default — it opens a pull request
rather than pushing, and somebody reads it before anything leaves the branch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from kasa.config import ForgetSettings, MemorySettings
from kasa.memory.document import MemoryDoc, MemoryError_
from kasa.memory.layout import ARCHIVE_DIR
from kasa.memory.ltm import ApplyResult, Change, CommitMeta, MemoryStore, MemoryStoreError
from kasa.memory.manifest import Manifest
from kasa.memory.patch import Archive, Delete, MemoryPatch, PatchCompiler, PatchError
from kasa.store import Store

log = logging.getLogger(__name__)

JOB = "forget"

#: Why a memory that looked forgettable was left alone. Counted rather than
#: listed, because on a corpus of any size the list is most of the corpus — but
#: counted at all, because "nothing was forgotten this week" and "everything
#: was pinned" are different facts about a run.
PINNED = "pinned"
TOO_RECENT = "younger than the retention floor"
STILL_LINKED = "still linked from a live memory"
OVER_BUDGET = "over this run's budget"


@dataclass(frozen=True, slots=True)
class Doomed:
    """One memory and the transition it is due for."""

    memory_id: str
    path: str
    title: str
    salience: float


@dataclass(slots=True)
class Forgetting:
    """What one week's pass did, and what it deliberately did not."""

    archived: list[Doomed] = field(default_factory=list)
    collected: list[Doomed] = field(default_factory=list)
    protected: dict[str, int] = field(default_factory=dict)
    changed: list[str] = field(default_factory=list)
    sha: str | None = None
    pull_request_url: str | None = None

    def summary(self) -> str:
        parts = []
        if self.archived:
            parts.append(f"{len(self.archived)} archived")
        if self.collected:
            parts.append(f"{len(self.collected)} collected")
        if not parts:
            parts.append("nothing forgotten")
        if self.protected:
            parts.append(
                "protected: "
                + ", ".join(f"{count} {reason}" for reason, count in sorted(self.protected.items()))
            )
        where = self.pull_request_url or self.sha
        return ", ".join(parts) + (f" in {where}" if where else "")


class Collector:
    """One `forget` run. Deterministic from end to end."""

    def __init__(
        self,
        store: Store,
        memory: MemoryStore,
        *,
        settings: ForgetSettings | None = None,
        policy: MemorySettings | None = None,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._settings = settings or ForgetSettings()
        self._policy = policy or MemorySettings()
        self._job_id = job_id
        self._now = now or datetime.now(UTC)

    async def run(self) -> Forgetting:
        manifest = self._memory.manifest()
        corpus = self._read(manifest)
        linked = _linked_from_live(corpus, manifest)
        outcome = Forgetting()

        archive = self._due_for_archiving(corpus, linked, outcome)
        collect = self._due_for_collection(corpus, linked, outcome)

        # Archives first, and the budget is shared. Archiving is the reversible
        # half — the file is still in the tree and its id still resolves — so
        # when a week cannot do everything, it should do that half.
        budget = self._settings.max_per_run
        outcome.archived = archive[:budget]
        outcome.collected = collect[: max(budget - len(outcome.archived), 0)]
        dropped = (len(archive) - len(outcome.archived)) + (len(collect) - len(outcome.collected))
        if dropped:
            _protect(outcome, OVER_BUDGET, dropped)

        patches: list[MemoryPatch] = [
            Archive(id=doomed.memory_id, reason=f"salience {doomed.salience:.2f}")
            for doomed in outcome.archived
        ]
        patches += [
            Delete(
                id=doomed.memory_id,
                reason=f"archived and untouched for {self._settings.archive_grace_days}d",
            )
            for doomed in outcome.collected
        ]

        result = await self._apply(patches, outcome)
        outcome.changed = list(result.changed)
        outcome.sha = result.sha
        outcome.pull_request_url = result.pull_request_url
        log.info("forget: %s", outcome.summary())
        return outcome

    # -- the two transitions -------------------------------------------------

    def _due_for_archiving(
        self, corpus: Sequence[tuple[str, MemoryDoc]], linked: frozenset[str], outcome: Forgetting
    ) -> list[Doomed]:
        """Live memories whose salience says nobody needs them any more."""
        due = []
        for path, doc in corpus:
            if path.startswith(f"{ARCHIVE_DIR}/"):
                continue
            if doc.frontmatter.salience >= self._settings.archive_below:
                continue
            if self._protected(doc, linked, outcome):
                continue
            due.append(
                Doomed(
                    memory_id=doc.id,
                    path=path,
                    title=doc.frontmatter.title,
                    salience=doc.frontmatter.salience,
                )
            )
        # Coldest first. A bounded run should spend its budget on the memories
        # furthest past the threshold, not on whichever the manifest lists.
        due.sort(key=lambda doomed: doomed.salience)
        return due

    def _due_for_collection(
        self, corpus: Sequence[tuple[str, MemoryDoc]], linked: frozenset[str], outcome: Forgetting
    ) -> list[Doomed]:
        """Archived memories that have sat out the grace period.

        Measured from `updated`, which archiving stamps — so the clock starts
        when the memory was archived, not when it was last true.
        """
        grace = timedelta(days=self._settings.archive_grace_days)
        due = []
        for path, doc in corpus:
            if not path.startswith(f"{ARCHIVE_DIR}/"):
                continue
            if self._now - doc.frontmatter.updated < grace:
                continue
            if self._protected(doc, linked, outcome):
                continue
            due.append(
                Doomed(
                    memory_id=doc.id,
                    path=path,
                    title=doc.frontmatter.title,
                    salience=doc.frontmatter.salience,
                )
            )
        due.sort(key=lambda doomed: doomed.salience)
        return due

    def _protected(self, doc: MemoryDoc, linked: frozenset[str], outcome: Forgetting) -> bool:
        """The three rules that outrank every number. Checked at both stages.

        The validator enforces the first two again on a delete. This is not
        redundancy for its own sake: the validator's job is to refuse a bad
        plan, and this one's is not to propose one — a `forget` that spent
        every week emitting plans the validator threw out would forget nothing
        at all, and nobody would notice until they went looking.
        """
        if doc.frontmatter.pinned:
            _protect(outcome, PINNED)
            return True
        floor = timedelta(days=self._policy.retention_floor_days)
        # The newer of the two timestamps. `updated` is what the validator
        # measures; `created` is what §6.2 says. Taking whichever is more
        # recent satisfies both and can only ever protect more.
        youngest = max(doc.frontmatter.created, doc.frontmatter.updated)
        if self._now - youngest < floor:
            _protect(outcome, TOO_RECENT)
            return True
        if doc.id in linked:
            # Archiving does not dangle a link — the id goes on resolving —
            # but a memory something still points at is a memory something
            # still needs, whatever its salience says about how often it is
            # searched for.
            _protect(outcome, STILL_LINKED)
            return True
        return False

    # -- reading and writing -------------------------------------------------

    def _read(self, manifest: Manifest) -> list[tuple[str, MemoryDoc]]:
        corpus: list[tuple[str, MemoryDoc]] = []
        for entry in manifest.memories.values():
            try:
                raw = self._memory.read(entry.path)
                corpus.append((entry.path, MemoryDoc.parse(raw, source=entry.path)))
            except (MemoryStoreError, MemoryError_) as exc:
                # A file this run cannot read is a file this run does not
                # touch. It is also one nothing else can see the links of, so
                # skipping it is what keeps a broken file from making the
                # memories it points at look unreferenced.
                log.warning("forget: skipping %s: %s", entry.path, exc)
        corpus.sort(key=lambda pair: pair[0])
        return corpus

    async def _apply(self, patches: Sequence[MemoryPatch], outcome: Forgetting) -> ApplyResult:
        if not patches:
            return ApplyResult()
        compiler = PatchCompiler(
            self._memory.path, self._memory.manifest(), policy=self._policy, now=self._now
        )
        try:
            changes: list[Change] = compiler.compile(list(patches), job=JOB)
        except PatchError as exc:
            # Every one of these was checked here first, so a rejection is a
            # disagreement between this job and the validator — which is a bug
            # in one of them, and the validator is the one to believe.
            log.error("forget: the validator refused this week's plan: %s", exc)
            return ApplyResult()
        return await self._memory.apply(
            changes,
            CommitMeta(
                summary=_headline(outcome),
                job=JOB,
                job_id=self._job_id,
                memory_ids=[d.memory_id for d in (*outcome.archived, *outcome.collected)],
            ),
        )


# -- helpers -----------------------------------------------------------------


def _linked_from_live(
    corpus: Sequence[tuple[str, MemoryDoc]], manifest: Manifest
) -> frozenset[str]:
    """Every memory id something still points at from outside the archive.

    Resolved through the manifest rather than matched as text, because half the
    links in a hand-written corpus are paths — `[[people/jane]]` protects Jane
    exactly as much as her id does, and a version of this that only understood
    ids would collect the memories people link to most readably.

    Links *from* the archive do not count. The archive is where things go to
    stop being the current answer, and letting one archived memory's links keep
    another alive is how a corpus of dead references never shrinks.

    Generous where it is ambiguous: both the id written down and the id it
    resolves to are protected. Every error this makes leaves one more memory in
    the corpus, which is the direction to be wrong in.
    """
    linked: set[str] = set()
    for path, doc in corpus:
        if path.startswith(f"{ARCHIVE_DIR}/"):
            continue
        for target in doc.links():
            found = {target} if target in manifest.memories else set()
            entry = manifest.resolve(target)
            if entry is not None and (resolved := manifest.id_at(entry.path)) is not None:
                found.add(resolved)
            # A memory linking to itself is not evidence that anything needs
            # it. It happens after a split hands the parts each other's ids.
            linked |= found - {doc.id}
    return frozenset(linked)


def _protect(outcome: Forgetting, reason: str, count: int = 1) -> None:
    outcome.protected[reason] = outcome.protected.get(reason, 0) + count


def _headline(outcome: Forgetting) -> str:
    parts = []
    if outcome.archived:
        parts.append(f"archive {len(outcome.archived)} memor{_y(len(outcome.archived))}")
    if outcome.collected:
        parts.append(f"collect {len(outcome.collected)} archived memor{_y(len(outcome.collected))}")
    return f"forget: {', '.join(parts)}"


def _y(count: int) -> str:
    return "y" if count == 1 else "ies"

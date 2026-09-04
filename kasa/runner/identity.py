"""`identity`: give every Slack user id one `people/` memory, and only one.

`Directory` (in the Slack adapter) learns who somebody is on the turn path and
writes it to SQLite. This is the other half: it moves that mapping into the
corpus, where it is durable, readable, and something a person can correct.

Why a job rather than a write during the turn. A commit takes the memory write
lease and may push; doing that the first time each person speaks would put git
in the middle of a conversation and contend with every consolidation job for
the lease. Batching a sweep is also what makes a workspace's first busy hour
one commit instead of forty.

The two rules it exists to keep:

**One person, one file.** A uid is matched on `source_refs`, never on a name.
Names change — that is the normal case, not the edge case — and a version of
this that matched on the display name would write a second Jane every time Jane
edited her profile, which is exactly the forking #23 is about.

**It owns a block, not a file.** `promote` writes prose about people into these
same files, and a person may edit them by hand. So the generated mapping lives
between two markers and nothing outside them is ever rewritten. Previous names
are not kept in the block either: `git log` is this corpus's audit trail, and
the rename is a diff in it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kasa.adapters.slack.identity import user_ref
from kasa.config import MemorySettings
from kasa.memory.document import MemoryDoc, MemoryError_, slugify
from kasa.memory.ltm import ApplyResult, Change, CommitMeta, MemoryStore, MemoryStoreError
from kasa.memory.manifest import Manifest
from kasa.memory.patch import Create, MemoryPatch, PatchCompiler, PatchError, Update
from kasa.store import Store

log = logging.getLogger(__name__)

JOB = "identity"

#: The generated block, and the only part of a `people/` file this job writes.
#: HTML comments because they are invisible in rendered Markdown and survive a
#: hand edit that reflows everything around them.
OPEN = "<!-- kasa:slack-identity -->"
CLOSE = "<!-- /kasa:slack-identity -->"
_BLOCK = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.DOTALL)

#: How many people one run will write. Below `MemorySettings.max_files_per_commit`
#: on purpose: the cap there rejects the whole plan, and a workspace joining a
#: hundred people at once should produce five commits rather than five refusals.
DEFAULT_LIMIT = 20

#: A directory entry is not a conversation, and this is the one place in the
#: Slack surface where `workspace` is the right scope (`docs/DESIGN.md` §10.1
#: says nothing from Slack is workspace-scoped, and means what conversations
#: put into memory). What is written here is `users.info` — the name already on
#: every message this person has ever posted, visible to the whole workspace.
#: Scoping it `private:<uid>` instead would hide Jane's own name from a channel
#: conversation about Jane, which would defeat the point of mapping her at all.
VISIBILITY = "workspace"


@dataclass(frozen=True, slots=True)
class Linked:
    """One person, and what this run did about them."""

    team_id: str
    user_id: str
    memory_id: str
    path: str
    name: str
    #: The row's `display_name`, which is what the sweep compares against. Kept
    #: distinct from `name`: the title falls back to the real name and then to
    #: the id, and linking on the fallback would leave the row perpetually due.
    display_name: str
    created: bool


@dataclass(slots=True)
class Identities:
    linked: list[Linked] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    sha: str | None = None
    pull_request_url: str | None = None

    @property
    def created(self) -> int:
        return sum(1 for one in self.linked if one.created)

    @property
    def updated(self) -> int:
        return sum(1 for one in self.linked if not one.created)

    def summary(self) -> str:
        if not self.linked:
            return "no Slack identities to map"
        parts = []
        if self.created:
            parts.append(f"{self.created} mapped")
        if self.updated:
            # "Updated" covers both a rename written into an existing memory
            # and a uid folded into one that was already there; from the
            # corpus's side they are the same event.
            parts.append(f"{self.updated} updated")
        where = self.pull_request_url or self.sha
        return ", ".join(parts) + (f" in {where}" if where else "")


class Registrar:
    """One `identity` run. Deterministic from end to end — no model in it."""

    def __init__(
        self,
        store: Store,
        memory: MemoryStore,
        *,
        limit: int = DEFAULT_LIMIT,
        policy: MemorySettings | None = None,
        job_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self._store = store
        self._memory = memory
        self._limit = limit
        self._policy = policy or MemorySettings()
        self._job_id = job_id
        self._now = now or datetime.now(UTC)

    async def run(self) -> Identities:
        outcome = Identities()
        pending = await self._store.slack_users_awaiting_memory(self._limit)
        if not pending:
            return outcome

        manifest = self._memory.manifest()
        corpus = self._people(manifest)
        by_ref = _index_by_ref(corpus)
        taken = _known_paths(manifest)

        writes: list[tuple[MemoryPatch, Linked]] = []
        # Rows the corpus already answers, needing a link and no commit. A run
        # that committed and died before marking its rows leaves exactly these,
        # and without them those rows would be swept, found identical, and left
        # pending again on every tick for the life of the workspace.
        settled: list[Linked] = []
        for row in pending:
            patch, linked = self._plan(row, by_ref=by_ref, corpus=corpus, taken=taken)
            taken.add(linked.path)
            if patch is None:
                settled.append(linked)
            else:
                writes.append((patch, linked))

        outcome.linked = [linked for _, linked in writes]
        result = await self._apply([patch for patch, _ in writes], outcome)
        if writes and result.sha is None and not result.pull_request_url:
            # Nothing was committed, so nothing it would have written may be
            # marked mapped — those rows have to come back next run. Reporting
            # a mapping that was never written is how a uid ends up pointing at
            # a file that is not there.
            log.warning("identity: %d mapping(s) were not written; still pending", len(writes))
            outcome.linked = []

        outcome.linked += settled
        outcome.changed = list(result.changed)
        outcome.sha = result.sha
        outcome.pull_request_url = result.pull_request_url
        await self._mark(outcome)
        log.info("identity: %s", outcome.summary())
        return outcome

    # -- deciding ------------------------------------------------------------

    def _plan(
        self,
        row: Mapping[str, Any],
        *,
        by_ref: Mapping[str, tuple[str, MemoryDoc]],
        corpus: Sequence[tuple[str, MemoryDoc]],
        taken: set[str],
    ) -> tuple[MemoryPatch | None, Linked]:
        """What this person needs written, and what to link them to afterwards.

        A `None` patch is a person the corpus already describes correctly. They
        are still linked — the row is what makes the sweep stop returning them.
        """
        team = str(row["team_id"])
        uid = str(row["user_id"])
        name = str(row["display_name"]) or str(row["real_name"] or "") or uid
        ref = user_ref(team, uid)

        found = by_ref.get(ref) or self._adoptable(name, corpus, team)
        if found is not None:
            path, doc = found
            return self._rename(row, path, doc, name=name, ref=ref)
        return self._create(row, name=name, ref=ref, taken=taken)

    def _adoptable(
        self, name: str, corpus: Sequence[tuple[str, MemoryDoc]], team: str
    ) -> tuple[str, MemoryDoc] | None:
        """A person memory that is plainly already about this person.

        The "link" half of "create or link": a corpus written before Kasa met
        the workspace — or by `promote`, from a conversation about somebody who
        had not spoken yet — already has a Jane, and writing a second one keyed
        by uid would be the same fork by another route.

        Matched on the slug rather than on the title, so punctuation and case
        do not decide it, and only when the candidate carries no Slack id of
        its own for this workspace: a file already claimed by another uid is
        somebody else with the same name, and adopting it would merge two
        people into one memory — the one mistake here that is worse than a
        duplicate.
        """
        slug = slugify(name)
        for path, doc in corpus:
            if slugify(doc.frontmatter.title) != slug:
                continue
            if any(r.startswith(f"slack://{team}/") for r in doc.frontmatter.source_refs):
                continue
            return path, doc
        return None

    def _create(
        self, row: Mapping[str, Any], *, name: str, ref: str, taken: set[str]
    ) -> tuple[MemoryPatch | None, Linked]:
        doc = MemoryDoc.new(
            type="person",
            title=name,
            body=_block(row),
            visibility=VISIBILITY,
            source_refs=[ref],
            tags=["slack"],
        )
        path = _free_path(doc, name, taken)
        return Create(memory=doc, path=path), Linked(
            team_id=str(row["team_id"]),
            user_id=str(row["user_id"]),
            memory_id=doc.id,
            path=path,
            name=name,
            display_name=str(row["display_name"]),
            created=True,
        )

    def _rename(
        self, row: Mapping[str, Any], path: str, doc: MemoryDoc, *, name: str, ref: str
    ) -> tuple[MemoryPatch | None, Linked]:
        """Fold this uid into a memory that already exists, in place."""
        body = _with_block(doc.body, row)
        frontmatter: dict[str, Any] = {}
        if ref not in doc.frontmatter.source_refs:
            frontmatter["source_refs"] = [*doc.frontmatter.source_refs, ref]

        # The title moves only when it is still the one this job wrote. A title
        # somebody else chose — a person, or `promote` reconciling what a
        # conversation said about them — is not ours to overwrite because a
        # display name changed.
        written = str(row["memory_name"] or "")
        if written and doc.frontmatter.title == written and written != name:
            frontmatter["title"] = name

        unchanged = body == doc.body and not frontmatter
        linked = Linked(
            team_id=str(row["team_id"]),
            user_id=str(row["user_id"]),
            memory_id=doc.id,
            path=path,
            name=name,
            display_name=str(row["display_name"]),
            created=False,
        )
        if unchanged:
            return None, linked
        return Update(id=doc.id, body=body, frontmatter=frontmatter), linked

    # -- reading and writing -------------------------------------------------

    def _people(self, manifest: Manifest) -> list[tuple[str, MemoryDoc]]:
        corpus: list[tuple[str, MemoryDoc]] = []
        for entry in manifest.memories.values():
            if entry.type != "person":
                continue
            try:
                raw = self._memory.read(entry.path)
                corpus.append((entry.path, MemoryDoc.parse(raw, source=entry.path)))
            except (MemoryStoreError, MemoryError_) as exc:
                # A file this run cannot read is one it cannot match a uid
                # against either. Skipping it risks a duplicate; guessing at it
                # risks writing over somebody. The duplicate is recoverable.
                log.warning("identity: skipping %s: %s", entry.path, exc)
        corpus.sort(key=lambda pair: pair[0])
        return corpus

    async def _apply(self, patches: Sequence[MemoryPatch], outcome: Identities) -> ApplyResult:
        if not patches:
            return ApplyResult()
        compiler = PatchCompiler(
            self._memory.path, self._memory.manifest(), policy=self._policy, now=self._now
        )
        try:
            changes: list[Change] = compiler.compile(list(patches), job=JOB)
        except PatchError as exc:
            log.error("identity: the validator refused this run's mappings: %s", exc)
            return ApplyResult()
        return await self._memory.apply(
            changes,
            CommitMeta(
                summary=_headline(outcome),
                job=JOB,
                job_id=self._job_id,
                memory_ids=[one.memory_id for one in outcome.linked],
            ),
        )

    async def _mark(self, outcome: Identities) -> None:
        """Point the rows at what was just written. After the commit, always.

        A row marked before the commit that then failed is a uid Kasa believes
        it has mapped and never will again — the one state this job cannot
        recover from on its own.
        """
        for one in outcome.linked:
            await self._store.link_slack_user(
                team_id=one.team_id,
                user_id=one.user_id,
                memory_id=one.memory_id,
                memory_name=one.display_name,
            )


# -- helpers -----------------------------------------------------------------


def _block(row: Mapping[str, Any]) -> str:
    """The generated mapping, as it appears in the file."""
    team = str(row["team_id"])
    uid = str(row["user_id"])
    handle = str(row["display_name"] or "")
    real = str(row["real_name"] or "")

    said = [f"Slack `{uid}` in workspace `{team}`."]
    if handle:
        said.append(f"Goes by `@{handle}`.")
    if real and real != handle:
        said.append(f"Profile name: {real}.")
    if bool(row["deleted"]):
        # Worth saying in the file rather than only in the row: a memory whose
        # subject has left the workspace still explains every conversation they
        # are already in, and a reader should know why they went quiet.
        said.append("This account has been deactivated.")
    return f"{OPEN}\n{' '.join(said)}\n{CLOSE}\n"


def _with_block(body: str, row: Mapping[str, Any]) -> str:
    """Replace the generated block, or append one, and touch nothing else."""
    block = _block(row)
    if _BLOCK.search(body):
        return _BLOCK.sub(lambda _: block.rstrip("\n"), body)
    separator = "" if body.endswith("\n\n") else "\n" if body.endswith("\n") else "\n\n"
    return f"{body}{separator}{block}"


def _index_by_ref(corpus: Sequence[tuple[str, MemoryDoc]]) -> dict[str, tuple[str, MemoryDoc]]:
    """uid ref → the memory that claims it. First by path wins a clash.

    Two files carrying one `slack://` ref is a corpus somebody edited into a
    contradiction. Taking the first and leaving the second alone keeps this run
    deterministic; `kasa reindex` is what reports the duplicate.
    """
    found: dict[str, tuple[str, MemoryDoc]] = {}
    for path, doc in corpus:
        for ref in doc.frontmatter.source_refs:
            if ref.startswith("slack://"):
                found.setdefault(ref, (path, doc))
    return found


def _known_paths(manifest: Manifest) -> set[str]:
    """Every path the corpus already occupies, not only the people in it."""
    return {entry.path for entry in manifest.memories.values()}


def _free_path(doc: MemoryDoc, name: str, taken: set[str]) -> str:
    """`memory/people/<slug>.md`, or the next one along if that is somebody.

    Reached only when the name matched no adoptable memory, so a collision here
    is two different people who share a name and cannot share a file.
    """
    base = doc.suggested_path(slugify(name))
    if base not in taken:
        return base
    stem = base.removesuffix(".md")
    for suffix in range(2, 100):
        candidate = f"{stem}-{suffix}.md"
        if candidate not in taken:
            return candidate
    raise MemoryStoreError(f"cannot find a free path for {base}")


def _headline(outcome: Identities) -> str:
    names = ", ".join(one.name for one in outcome.linked[:3])
    if len(outcome.linked) > 3:
        names += f" and {len(outcome.linked) - 3} more"
    return f"map {len(outcome.linked)} Slack identity(s): {names}"

"""The agent's own access to long-term memory.

Pre-injected retrieval handles the common case at no extra latency, and it will
still miss — someone asks about a memory whose wording shares nothing with the
question, or asks a second question the first retrieval did not anticipate.
These three tools are the tail, and shipping both is the point: injection covers
the ninety percent, tools cover the rest.

Two constraints shape all of them:

- **Scope comes from the session, never from the model.** A tool argument that
  could widen visibility would make every other scope check decorative.
- **`memory_write` does not write.** It appends an observation, and the
  `promote` job turns observations into a validated patch plan. The interactive
  path and the background path therefore share one write path — the one that
  has been through the patch validator.
"""

from __future__ import annotations

import logging
from typing import Any

from kasa.core.tools import Tool, ToolContext
from kasa.memory.document import MemoryDoc, MemoryError_, is_memory_id
from kasa.memory.ltm import MemoryStore, MemoryStoreError
from kasa.memory.observation import OBSERVATION_KINDS
from kasa.memory.retrieve import Retriever, permits, render_snippet
from kasa.memory.subject import normalize_subject
from kasa.store import Store

log = logging.getLogger(__name__)


MAX_SEARCH_LIMIT = 20

WROTE = (
    "Noted. This is queued as a candidate for long-term memory; it is reviewed "
    "and written by the consolidation job, not immediately."
)


def memory_tools(*, retriever: Retriever, memory: MemoryStore, store: Store) -> list[Tool]:
    """The three memory tools, bound to one repo and one database."""
    return [
        _search_tool(retriever),
        _read_tool(memory),
        _write_tool(store),
    ]


# -- memory_search -----------------------------------------------------------


def _search_tool(retriever: Retriever) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        query = str(args["query"]).strip()
        limit = min(int(args.get("limit", 5)), MAX_SEARCH_LIMIT)

        scope = context.scope
        if hint := args.get("scope_hint"):
            # A hint may narrow the search, never widen it. Asking to search a
            # scope this session cannot see is refused rather than downgraded,
            # so the model is told what happened instead of quietly getting
            # results from somewhere else.
            if not permits(context.scope, str(hint)):
                return f"This conversation cannot search the scope {hint!r}."
            scope = str(hint)

        # Pinned memories are excluded from the pool and the results are read
        # in rank order. Both matter here and nowhere else: an explicit search
        # asked for what matches, and every pinned memory is already in the
        # prompt under `# Pinned memory`. Leading with one answered a question
        # nobody asked, and spent a result slot doing it. A pinned memory that
        # does match the query still ranks, and still gets its pinned bonus.
        # The limit is passed down rather than applied to the result. The
        # retriever's own limit is how many memories fit in a prompt beside the
        # conversation; slicing what it had already packed to that bound meant
        # `limit` could only ever shrink a list of eight, and a request for
        # twenty was answered with eight and no indication of it (#61).
        retrieval = await retriever.retrieve(query, scope=scope, include_pinned=False, limit=limit)
        if not retrieval.kept:
            return f"No memories matched {query!r}."
        return "\n\n".join(render_snippet(c) for c in retrieval.kept)

    return Tool(
        name="memory_search",
        description=(
            "Search long-term memory. Use it when the working context does not "
            "already contain what you need, or to check a second thing the "
            "conversation has moved on to. Returns ranked snippets, each headed "
            "by its memory id."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for, in words likely to appear in the memory.",
                },
                "scope_hint": {
                    "type": "string",
                    "description": (
                        "Optional visibility scope to restrict the search to, e.g. "
                        "'channel:C0123'. Can only narrow what this conversation "
                        "may already see."
                    ),
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_SEARCH_LIMIT},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=handler,
    )


# -- memory_read -------------------------------------------------------------


def _read_tool(memory: MemoryStore) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        memory_id = str(args["memory_id"]).strip()
        if not is_memory_id(memory_id):
            return f"{memory_id!r} is not a memory id. Ids look like mem_01K8XQ…"

        entry = memory.manifest().resolve(memory_id)
        if entry is None:
            return f"No memory with id {memory_id}."

        # The scope check happens on the document, not the manifest entry: the
        # entry is a copy, and a copy is a thing that can go stale.
        try:
            raw = memory.read(entry.path)
            doc = MemoryDoc.parse(raw, source=entry.path)
        except (MemoryStoreError, MemoryError_) as exc:
            return f"Could not read {entry.path}: {exc}"

        if not permits(context.scope, doc.frontmatter.visibility):
            log.info("refused a cross-scope memory_read of %s", memory_id)
            return f"No memory with id {memory_id}."

        return raw

    return Tool(
        name="memory_read",
        description=(
            "Read one long-term memory in full, by its id. Use it after "
            "memory_search when a snippet is not enough."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "A mem_… id from a search result."}
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
        handler=handler,
    )


# -- memory_write ------------------------------------------------------------


def _write_tool(store: Store) -> Tool:
    async def handler(args: dict[str, Any], context: ToolContext) -> str:
        kind = str(args["kind"])
        if kind not in OBSERVATION_KINDS:
            return f"kind must be one of {', '.join(OBSERVATION_KINDS)}."

        # Normalized here as well as in the store, because the store cannot
        # answer the model. A subject of "???" is a grouping key of "", and an
        # observation nothing can ever be grouped with is one nobody will read.
        subject = normalize_subject(str(args["subject"]))
        claim = str(args["claim"]).strip()
        # `minLength` in the schema rejects the empty string upstream, the way
        # the `kind` enum does; it cannot see a string of spaces. Both used to
        # get through, and the observation sat `pending` forever with nothing
        # in it — after the model had been told the write succeeded, which is
        # the one answer a write tool must not give for a no-op (#79).
        if not subject or not claim:
            return "subject and claim must each say something; nothing was recorded."

        await store.add_observation(
            subject=subject,
            claim=claim,
            kind=kind,
            # Inherited, never supplied: an observation from a DM stays private
            # even if the model would rather it were general knowledge.
            scope=context.scope,
            session_id=context.session_id,
        )
        return WROTE

    return Tool(
        name="memory_write",
        description=(
            "Record something worth remembering beyond this conversation. This "
            "queues a candidate fact for review; it does not write to memory "
            "directly, and nothing you record here is visible to a later "
            "conversation until it has been consolidated."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": list(OBSERVATION_KINDS)},
                "subject": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Who or what this is about, e.g. a person or project name.",
                },
                "claim": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The thing to remember, as one self-contained sentence.",
                },
            },
            "required": ["kind", "subject", "claim"],
            "additionalProperties": False,
        },
        handler=handler,
    )

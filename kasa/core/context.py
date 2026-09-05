"""Fixed-allocation, tokenizer-aware context assembly.

Two properties this module exists to guarantee:

1. **The recent turns are never starved.** Each segment gets a share of the
   budget and truncates at its own boundary, so an overlong retrieval cannot
   evict the conversation the user is actually having.
2. **The cacheable prefix is byte-stable.** `system` is built from inputs that
   do not vary turn to turn, and per-turn material goes in `context` instead.
   Prompt caching is worth roughly an order of magnitude on a long session, and
   a single interpolated timestamp in the prefix destroys it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from kasa.errors import ConfigError
from kasa.llm.tokens import Tokenizer, count_message
from kasa.llm.types import Message, ToolDef

#: Counts tokens in a string.
Counter = Callable[[str], int]

PINNED_SEPARATOR = "\n\n"
TRUNCATION_MARKER = "\n[…truncated]"
CONTEXT_HEADER = "# Working context"
RETRIEVED_HEADER = "## Retrieved memory"
EPISODE_HEADER = "## Conversation so far"

#: Pinned memories stay in the cacheable prefix — they are the stable-across-
#: turns half of retrieval — but they are still recalled material, and #45 was
#: that they arrived fused to the end of the system prompt with no header at
#: all. Anything the model is told to treat as background needs to be inside a
#: section the system prompt can name; content the consolidator wrote from a
#: conversation must not read as though the operator wrote it.
PINNED_HEADER = "# Pinned memory"

#: Kasa's own note about the turn in progress, and the reason it is a headed
#: section rather than a loose line (#201). Everything else in `context` is
#: recalled material the system prompt tells the model to treat as background
#: rather than as instructions. The tool budget is the opposite of that — it is
#: operational fact about the turn, and the sentence that scopes the memory
#: sections must not be able to reach it.
STATUS_HEADER = "# Turn status"


@dataclass(frozen=True, slots=True)
class ContextBudget:
    """Token budget and how it is divided.

    Shares are of the whole window, and include headroom for the completion
    itself, so they must sum to 1.0.
    """

    total: int = 128_000
    system: float = 0.05
    pinned: float = 0.10
    retrieved: float = 0.30
    episode: float = 0.10
    recent: float = 0.35
    headroom: float = 0.10

    def __post_init__(self) -> None:
        shares = (
            self.system,
            self.pinned,
            self.retrieved,
            self.episode,
            self.recent,
            self.headroom,
        )
        if abs(sum(shares) - 1.0) > 1e-6:
            raise ConfigError(f"context budget shares must sum to 1.0, got {sum(shares)}")
        if self.total <= 0:
            raise ConfigError("context budget total must be positive")

    def tokens_for(self, share: float) -> int:
        return int(self.total * share)


@dataclass(frozen=True, slots=True)
class SegmentTrace:
    name: str
    budget: int
    used: int
    kept: int
    dropped: int
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class PackTrace:
    total_budget: int
    used: int
    segments: tuple[SegmentTrace, ...] = ()

    def render(self) -> str:
        lines = [f"budget {self.total_budget} tokens, used {self.used}"]
        for seg in self.segments:
            note = " (truncated)" if seg.truncated else ""
            lines.append(
                f"  {seg.name:<10} {seg.used:>7}/{seg.budget:<7} "
                f"kept={seg.kept} dropped={seg.dropped}{note}"
            )
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PackedContext:
    system: str
    """The cacheable prefix. Byte-stable given the same inputs."""

    context: str | None
    """Per-turn material. Deliberately outside the cached prefix."""

    messages: tuple[Message, ...]
    trace: PackTrace = field(default=PackTrace(0, 0))


class ContextPacker:
    def __init__(self, budget: ContextBudget | None = None, *, tokenizer: Tokenizer) -> None:
        self.budget = budget or ContextBudget()
        self._tok = tokenizer

    def pack(
        self,
        *,
        system_prompt: str,
        pinned: Sequence[str] = (),
        retrieved: Sequence[str] = (),
        episode_summary: str | None = None,
        recent: Sequence[Message] = (),
        tools: Sequence[ToolDef] = (),
        status: str | None = None,
    ) -> PackedContext:
        """`status` is what Kasa has to say about the turn itself, if anything.

        It rides in `context` rather than the prefix because it changes on
        every pass — a tool-budget countdown in the cached prefix would
        invalidate the cache on every request, which is the whole reason the
        two are separate.
        """
        traces: list[SegmentTrace] = []

        system_text, prefix_traces = self._pack_prefix(system_prompt, pinned, tools, status)
        traces.extend(prefix_traces)

        context_text, context_traces = self._pack_context(retrieved, episode_summary, status)
        traces.extend(context_traces)

        messages, recent_trace = self._pack_recent(recent)
        traces.append(recent_trace)

        used = sum(t.used for t in traces)
        return PackedContext(
            system=system_text,
            context=context_text,
            messages=messages,
            trace=PackTrace(total_budget=self.budget.total, used=used, segments=tuple(traces)),
        )

    # -- segments ------------------------------------------------------------

    def _pack_prefix(
        self,
        system_prompt: str,
        pinned: Sequence[str],
        tools: Sequence[ToolDef],
        status: str | None = None,
    ) -> tuple[str, list[SegmentTrace]]:
        # Tool schemas are serialized into the prompt by every provider, so they
        # are charged against the system share even though they are not ours to
        # truncate.
        tool_tokens = sum(
            self._tok.count(t.name)
            + self._tok.count(t.description)
            + self._tok.count(repr(t.input_schema))
            for t in tools
        )
        system_budget = self.budget.tokens_for(self.budget.system)
        pinned_budget = self.budget.tokens_for(self.budget.pinned)

        kept_pinned, dropped = _fit(list(pinned), pinned_budget, self._tok.count)
        parts = [system_prompt]
        if kept_pinned:
            # Labelled, not concatenated: the system prompt tells the model to
            # treat recalled material as background rather than as instructions,
            # and that sentence can only scope to a section it can name.
            parts.append(PINNED_HEADER + "\n" + PINNED_SEPARATOR.join(kept_pinned))
        text = PINNED_SEPARATOR.join(p for p in parts if p)

        # Two traces, not one. Reporting the prefix as a single `system` row hid
        # how much of it was memory rather than prompt, which is the number
        # anyone reading `/trace` about a bloated prefix is looking for.
        return text, [
            SegmentTrace(
                name="system",
                budget=system_budget,
                # Tool schemas and the turn status are both charged here rather
                # than where they sit on the wire. Neither is memory competing
                # for a share; both are prompt Kasa wrote, and the system share
                # is the only row a reader can hold responsible for them.
                used=self._tok.count(system_prompt) + tool_tokens + self._tok.count(status or ""),
                kept=1 if system_prompt else 0,
                dropped=0,
            ),
            SegmentTrace(
                name="pinned",
                budget=pinned_budget,
                used=sum(self._tok.count(p) for p in kept_pinned),
                kept=len(kept_pinned),
                dropped=dropped,
            ),
        ]

    def _pack_context(
        self, retrieved: Sequence[str], episode_summary: str | None, status: str | None = None
    ) -> tuple[str | None, list[SegmentTrace]]:
        traces: list[SegmentTrace] = []
        sections: list[str] = []

        episode_budget = self.budget.tokens_for(self.budget.episode)
        episode_text = ""
        truncated = False
        if episode_summary:
            episode_text, truncated = _truncate(episode_summary, episode_budget, self._tok.count)
            if episode_text:
                sections.append(f"{EPISODE_HEADER}\n{episode_text}")
        traces.append(
            SegmentTrace(
                name="episode",
                budget=episode_budget,
                used=self._tok.count(episode_text),
                kept=1 if episode_text else 0,
                dropped=0,
                truncated=truncated,
            )
        )

        retrieved_budget = self.budget.tokens_for(self.budget.retrieved)
        kept, dropped = _fit(list(retrieved), retrieved_budget, self._tok.count)
        if kept:
            sections.append(RETRIEVED_HEADER + "\n" + "\n\n".join(kept))
        traces.append(
            SegmentTrace(
                name="retrieved",
                budget=retrieved_budget,
                used=sum(self._tok.count(k) for k in kept),
                kept=len(kept),
                dropped=dropped,
            )
        )

        # Status leads, and stays outside the working-context block: it is not
        # recalled material and must not be read as any.
        blocks: list[str] = []
        if status:
            blocks.append(f"{STATUS_HEADER}\n{status}")
        if sections:
            blocks.append(f"{CONTEXT_HEADER}\n\n" + "\n\n".join(sections))
        if not blocks:
            return None, traces
        return "\n\n".join(blocks), traces

    def _pack_recent(self, recent: Sequence[Message]) -> tuple[tuple[Message, ...], SegmentTrace]:
        budget = self.budget.tokens_for(self.budget.recent)
        groups = group_turns(recent)

        kept: list[list[Message]] = []
        used = 0
        # Newest first: the most recent exchange is the one that must survive.
        for group in reversed(groups):
            cost = sum(count_message(m, self._tok) for m in group)
            if used + cost > budget and kept:
                break
            kept.append(group)
            used += cost

        kept.reverse()
        messages = tuple(m for group in kept for m in group)
        return messages, SegmentTrace(
            name="recent",
            budget=budget,
            used=used,
            kept=len(kept),
            dropped=len(groups) - len(kept),
        )


def group_turns(messages: Sequence[Message]) -> list[list[Message]]:
    """Split a transcript into atomic exchanges.

    A group is a user message plus everything that answers it, including the
    assistant's tool calls and their results. Truncation operates on whole
    groups because an assistant `tool_use` whose `tool_result` was dropped is a
    hard 400 on both provider families — the single most likely way for a naive
    packer to break a long conversation.
    """
    groups: list[list[Message]] = []
    current: list[Message] = []

    for msg in messages:
        starts_turn = msg.role == "user" and not msg.tool_results_in
        if starts_turn and current:
            groups.append(current)
            current = []
        current.append(msg)

    if current:
        groups.append(current)
    return groups


def _fit(items: list[str], budget: int, count: Counter) -> tuple[list[str], int]:
    """Keep as many leading items as fit. Items are assumed ranked, best first."""
    kept: list[str] = []
    used = 0
    for item in items:
        cost = count(item)
        if used + cost > budget:
            break
        kept.append(item)
        used += cost
    return kept, len(items) - len(kept)


def _truncate(text: str, budget: int, count: Counter) -> tuple[str, bool]:
    """Trim text to a token budget, preferring a paragraph boundary."""
    if count(text) <= budget:
        return text, False

    # The marker is part of what gets sent, so it comes out of the budget before
    # trimming rather than being appended to an already-full segment.
    target = max(0, budget - count(TRUNCATION_MARKER))
    if target == 0:
        return "", True

    # Token counts grow monotonically with length, so a proportional first cut
    # lands close and the loop only has to walk it back a little.
    ratio = target / max(1, count(text))
    cut = max(1, int(len(text) * ratio))
    while cut > 1 and count(text[:cut]) > target:
        cut = int(cut * 0.9)

    trimmed = text[:cut]
    if (boundary := trimmed.rfind("\n\n")) > cut * 0.5:
        trimmed = trimmed[:boundary]
    return trimmed.rstrip() + TRUNCATION_MARKER, True

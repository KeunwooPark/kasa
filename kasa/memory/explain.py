"""Rendering a retrieval trace for `kasa why`.

Every quality complaint about a memory system bottoms out in "why did it not
remember X", and the only honest answer is the whole pipeline: what was
searched for, what came back, what each scoring step did to it, what the scope
filter removed, and what actually fitted in the budget.

Rendering lives here rather than in the CLI so that the same trace can be shown
in a terminal now and in a Slack thread later without the explanation being
rewritten twice.
"""

from __future__ import annotations

from kasa.memory.retrieve import Retrieval, RetrievalTrace


def render_trace(retrieval: Retrieval, *, limit: int = 20) -> str:
    trace = retrieval.trace
    if trace is None:
        return "no trace was recorded"

    lines = _query(trace)
    lines += _candidates(trace, limit)
    lines += _denied(trace)
    lines += _packed(retrieval, trace)
    return "\n".join(lines)


def _query(trace: RetrievalTrace) -> list[str]:
    lines = ["QUERY", f"  asked      {trace.question}"]
    if trace.rewritten:
        lines.append(f"  rewritten  {trace.query}")
    else:
        lines.append("  rewritten  no — the message was already self-contained")
    lines.append(f"  match      {trace.match_expression or '(nothing searchable)'}")
    lines.append(f"  scope      {trace.scope}")
    return [*lines, ""]


def _candidates(trace: RetrievalTrace, limit: int) -> list[str]:
    if not trace.candidates:
        return ["CANDIDATES", "  none matched", ""]

    kept = {c.chunk_id for c in trace.kept}
    lines = [
        f"CANDIDATES ({len(trace.candidates)}, best {min(limit, len(trace.candidates))} shown)",
        f"  {'':<3} {'final':>8} {'fused':>8} {'salien':>7} {'recency':>8}  {'sources':<18} memory",
    ]
    for position, candidate in enumerate(trace.candidates[:limit], start=1):
        mark = "*" if candidate.chunk_id in kept else " "
        lines.append(
            f"  {mark}{position:<2} {candidate.final:>8.5f} {candidate.fused:>8.5f} "
            f"{candidate.salience:>7.2f} {candidate.recency:>8.3f}  "
            f"{','.join(candidate.sources) or '—':<18} "
            f"{candidate.memory_id} {candidate.path}#{candidate.ordinal}"
        )
    lines.append("  * = packed into the prompt")
    return [*lines, ""]


def _denied(trace: RetrievalTrace) -> list[str]:
    if not trace.denied:
        return [
            "SCOPE FILTER",
            "  nothing was excluded by scope",
            "",
        ]
    lines = ["SCOPE FILTER", f"  {len(trace.denied)} chunk(s) never entered the ranking:"]
    for candidate in trace.denied:
        lines.append(f"    {candidate.memory_id} {candidate.path} — {candidate.denied}")
    return [*lines, ""]


def _packed(retrieval: Retrieval, trace: RetrievalTrace) -> list[str]:
    lines = [
        "PACKED",
        f"  {trace.used_tokens} of {trace.budget_tokens} retrieval tokens used, "
        f"{len(trace.kept)} memories",
    ]
    if not trace.kept:
        lines.append("  nothing was injected")
        return [*lines, ""]

    for snippet in [*retrieval.pinned, *retrieval.snippets]:
        head, _, rest = snippet.partition("\n")
        preview = " ".join(rest.split())
        lines.append(f"    {head}")
        lines.append(f"      {preview[:120]}{'…' if len(preview) > 120 else ''}")
    return [*lines, ""]

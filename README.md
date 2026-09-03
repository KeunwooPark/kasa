# kasa

A long-running, memory-native AI agent server reachable over chat.

Short-term memory lives in a local SQLite database. Long-term memory lives in a
private GitHub repository as Markdown files, curated by background jobs and
readable by a human. Background jobs promote what matters out of the former into
the latter, reorganize it as it grows, and forget what stopped mattering.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design.

## Status

**v0 — it talks.** Terminal adapter, agent turn loop, tool dispatch, and both
provider families. No memory yet beyond the current conversation; that starts in
v1. See the [milestones](https://github.com/KeunwooPark/kasa/milestones).

## Quick start

```bash
uv sync --dev
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY
uv run kasa run
```

With no config file present, Kasa synthesizes one from whatever API key is
exported. To configure it properly, write `~/.config/kasa/config.toml` — see
Appendix A of the design doc, or `uv run kasa config` to print what is currently
resolved.

```
kasa run          start the terminal adapter
kasa config       print the resolved configuration
kasa cost         token and spend totals
kasa db migrate   apply pending migrations
```

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy kasa
```

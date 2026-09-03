# kasa

A long-running, memory-native AI agent server reachable over chat.

Short-term memory lives in a local SQLite database. Long-term memory lives in a
private GitHub repository as Markdown files, curated by background jobs and
readable by a human. Background jobs promote what matters out of the former into
the latter, reorganize it as it grows, and forget what stopped mattering.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design.

## Status

**v0 — it talks.** Terminal adapter, agent turn loop, tool dispatch, and both
provider families. **v1 — it remembers on purpose** is in progress: `kasa init`
now sets up the private memory repo, and retrieval follows. See the
[milestones](https://github.com/KeunwooPark/kasa/milestones).

## Quick start

```bash
uv sync --dev
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY
export KASA_GITHUB_TOKEN=...     # fine-grained PAT, contents: write
uv run kasa init
uv run kasa run
```

`kasa init` walks through the private GitHub repo that holds long-term memory —
creating it if it does not exist — clones it, lays out the memory skeleton, and
writes `~/.config/kasa/config.toml`. It refuses to configure a public repo, and
it is safe to re-run: nothing already in the repo is overwritten.

The config file holds **no secrets**, only the names of the environment
variables that carry them. See Appendix A of the design doc for its shape, or
`uv run kasa config` to print what is currently resolved.

To try it without any of that, skip `init`: with no config file, Kasa
synthesizes one from whatever API key is exported and runs without memory.

```
kasa init         interactive setup; bootstraps the memory repo
kasa run          start the terminal adapter
kasa reindex      rebuild the search index from the memory repo
kasa doctor       check config, tokens, repo privacy, and the clone
kasa config       print the resolved configuration
kasa cost         token and spend totals
kasa db migrate   apply pending migrations
```

`kasa doctor` exits non-zero if any check failed, so it works as a health check.
Kasa also re-checks on every start that the memory repo is still private, and
refuses to run if it is not.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy kasa
```

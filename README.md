# kasa

A long-running, memory-native AI agent server reachable over chat.

Short-term memory lives in a local SQLite database. Long-term memory lives in a
private GitHub repository as Markdown files, curated by background jobs and
readable by a human. Background jobs promote what matters out of the former into
the latter, reorganize it as it grows, and forget what stopped mattering.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design.

## Status

**v1 — it remembers on purpose.** Long-term memory lives in a private git repo
of Markdown files. Retrieval is lexical (FTS5 + BM25, scope-filtered) and runs
on every turn; the agent can also search, read, and propose memories with tools.
Consolidation is still manual — background jobs arrive in v3. See the
[milestones](https://github.com/KeunwooPark/kasa/milestones).

**v2 in progress — it lives in Slack.** `kasa run --slack` connects over Socket
Mode, so a self-hosted daemon needs no public ingress. Events land in a durable
queue and are acknowledged immediately; one actor per thread answers them in
order, and many threads at once.

Background jobs are rows in the same database, run by a scheduler inside the
daemon: a restart loses nothing and a crashed job runs again. The jobs that
consolidate memory arrive in v3; today the scheduler runs `reindex`.

Nothing writes to the memory repo without going through a typed patch plan that
deterministic code validates first, and no delete is ever a force-push.

## Quick start

```bash
uv sync --dev
export ANTHROPIC_API_KEY=...     # or OPENAI_API_KEY
export KASA_GITHUB_TOKEN=...     # fine-grained PAT, contents: write
uv run kasa init
uv run kasa run
```

For Slack, install the `slack` extra, create an app with Socket Mode enabled,
and export the two tokens `kasa init` asked for the names of:

```bash
uv sync --extra slack
export SLACK_APP_TOKEN=xapp-...   # Socket Mode, connections:write
export SLACK_BOT_TOKEN=xoxb-...   # app_mentions:read, chat:write, im:history
uv run kasa run --slack
```

Kasa answers when it is mentioned in a channel it has been invited to, in any
thread it is already part of, and in a DM. Everything else it hears, it ignores.
Set `slack.allowed_channels` to narrow that further.

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
kasa run --slack  serve Slack over Socket Mode
kasa reindex      rebuild the search index from the memory repo
kasa why "<q>"    show the full retrieval trace for a question
kasa doctor       check config, tokens, repo privacy, and the clone
kasa config       print the resolved configuration
kasa cost         token and spend totals
kasa inbox status what is queued, and what stopped being retried
kasa inbox retry  requeue every dead-lettered event
kasa job run <k>  run a background job now
kasa job list     what each job is doing, and when it last ran
kasa job retry    requeue every dead-lettered job
kasa db migrate   apply pending migrations
```

`kasa doctor` exits non-zero if any check failed, so it works as a health check.
Kasa also re-checks on every start that the memory repo is still private, and
refuses to run if it is not.

## Memory

Every memory is a Markdown file with YAML frontmatter, under `memory/` in the
repo `kasa init` set up. Correct a belief by editing the file; the agent reads
your version. See what it decided to believe with `git log`; undo it with
`git revert`.

During a conversation the agent can:

- `memory_search` — look for something the pre-injected context did not carry
- `memory_read` — read one memory in full, by id
- `memory_write` — *propose* something worth remembering

`memory_write` does not write a file. It queues a candidate fact that the
consolidation job reviews, so the interactive path and the background path share
one validated write path. Anything scoped to a DM or a private channel stays
there: retrieval filters on visibility before it ranks.

## Development

```bash
uv sync --all-extras --dev   # `--all-extras` brings in Slack
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy kasa
```

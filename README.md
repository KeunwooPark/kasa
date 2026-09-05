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
See the [milestones](https://github.com/KeunwooPark/kasa/milestones).

**v2 in progress — it lives in Slack.** `kasa run --slack` connects over Socket
Mode, so a self-hosted daemon needs no public ingress. Events land in a durable
queue and are acknowledged immediately; one actor per thread answers them in
order, and many threads at once.

**v3/v4 in progress — it remembers automatically, and curates itself.**
Background jobs are rows in the same database, run by a scheduler inside the
daemon: a restart loses nothing and a crashed job runs again.

**v5 in progress — it acts on its own.** A standing task is a schedule somebody
set up — *"every weekday at 9am, tell me what happened in AI overnight"*. Ask
for one in the conversation you want it to answer in, and it fires there, with
the same memory and the same tools as when you asked.

| job | when | does |
| --- | --- | --- |
| `episode_close` | every 5 min | closes a thread that has gone quiet or grown long; summarizes it and extracts candidate facts |
| `promote` | hourly | reconciles those against the corpus and commits a patch plan |
| `reflect` | nightly | writes the day's journal, recomputes salience, applies feedback, surfaces contradictions |
| `reorganize` | weekly | merges duplicates, splits oversized files, repairs links, regenerates the listings |
| `forget` | weekly | archives what stopped mattering, and collects the archive after a grace period |
| `reindex` | every minute | rebuilds the search index and the manifest for changed blobs |
| `identity` | every 15 min | maps each Slack user id to one `people/` memory, and follows renames into it |
| `task_run` | when a standing task is due | starts the turn a person scheduled, in the conversation they scheduled it from |

Those first seven ship with Kasa and are the same in every install: they are how
it keeps its own memory in order. `task_run` is the other kind. It does not know
what it is running until it reads the `tasks` table, which holds whatever
schedules people have set up on this install — user data, not product
behaviour. The two stay in separate tables for exactly that reason.

An episode is scored before anything expensive happens to it, so small talk
closes with a summary and costs nothing further. `forget` makes no model call at
all — every input to it is already in the corpus — and runs supervised by
default, opening a pull request rather than pushing.

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

The reply goes up as a placeholder and is rewritten about once a second until
the answer is complete — never per token, which flickers and would spend the
turn being rate limited. Under rate-limit pressure it drops the intermediate
frames and keeps the answer. Set `slack.stream = false` for one message a turn.

Editing a message rewrites what Kasa stored and marks any candidate fact drawn
from it stale; deleting one leaves a tombstone in the transcript and lowers the
confidence of what was drawn from it. Neither rewrites the memory repo: when
the claim is already a file there, Kasa queues a review — `kasa review list` —
because a retraction is not a correction and the file may have been merged or
built on since.

Reactions on Kasa's own answers are the cheapest quality signal there is. 👍
raises the salience of the memories that produced the answer; ❌ lowers their
confidence and queues a review. One person is one vote per answer, and
`slack.reactions` maps emoji to verdicts.

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
kasa doctor       check config, tokens, repo privacy, search, and the clone
kasa config       print the resolved configuration
kasa cost         token and spend totals
kasa inbox status what is queued, and what stopped being retried
kasa inbox retry  requeue every dead-lettered event
kasa job run <k>  run a background job now
kasa job list     what each job is doing, and when it last ran
kasa job retry    requeue every dead-lettered job
kasa task list    standing tasks, and when each fires next
kasa task add     create one: `--cron "0 9 * * 1-5" --tz Asia/Seoul`
kasa task rm      delete one
kasa task pause   stop it firing, without forgetting it
kasa task resume  start it again, and clear the failures that stopped it
kasa task run     fire one now, without waiting for the clock
kasa review list  what is waiting on a person, and why
kasa review done  mark a review as dealt with
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

## Standing tasks

Ask for one where you want it to answer:

> **you** — every weekday at 9am Seoul time, search for what happened in AI
> overnight and give me the five things that matter
>
> **kasa** — Done. It next runs Mon 07 Sep 09:00, Tue 08 Sep 09:00 and
> Wed 09 Sep 09:00, Asia/Seoul.

At nine on Monday that prompt arrives in this thread as though you had typed it,
and Kasa answers it there — same memory, same tools, same thread. It knows
nobody spoke just now, so it gives you the news rather than thanking you for
asking.

The next fire times come back because they are the only part of this you can
actually check. Nobody can proofread `0 9 * * 1-5`; anybody can notice that the
first run is on a Monday.

**A task answers where it was created, and nowhere else.** There is no way to
ask for one that posts somewhere else — not a rule that is enforced, but an
argument that does not exist. The channel, the thread and the visibility are
copied off the conversation, so a task set up in a DM stays in the DM, and text
Kasa merely *reads* cannot arrange for anything to be said in a public channel.
Listing and cancelling are scoped the same way: one thread cannot see or delete
another's schedules. The terminal is the exception, and deliberately so:
`kasa task add --session` is the operator of the install choosing, which is a
different thing from the model being able to.

**Standing tasks need the daemon.** The clock runs inside `kasa run --slack`; on
a terminal, `kasa task add` writes the row and nothing fires it (the command
says so). `kasa task run <id>` fires one occurrence by hand.

From the terminal:

```bash
uv run kasa task list
uv run kasa task add "summarize yesterday" --cron "0 9 * * 1-5" --tz Asia/Seoul
uv run kasa task pause <id>
```

Every firing is a full turn — retrieval, a frontier model, whatever tools it
reaches for — and it is metered like any other, so it shows up in `kasa cost`.
The `[budget]` ceiling pauses background utility work rather than a scheduled
answer, which makes these the bounds that actually apply:

```toml
[tasks]
max_per_owner          = 20   # per person, counting paused ones
min_interval_minutes   = 15   # no schedule tighter than this
disable_after_failures = 5    # then it pauses, and tells whoever created it
```

## Web search

Optional, and off until you ask for it. With a Brave Search key in the vault,
`web_search` lets Kasa answer things memory cannot — anything current, or simply
outside the corpus.

```bash
uv run kasa vault set BRAVE_SEARCH_API_KEY
```

```toml
[search]
kind              = "brave"
max_results       = 5
cost_per_call_usd = 0.005    # counts toward the same [budget] ceiling as models
```

Without a `[search]` section the tool is not registered at all, so Kasa never
claims a capability it does not have.

Results are snippets. When the answer is on the page rather than in the
description of it, `web_fetch` opens it.

What comes back was written by strangers, so it arrives inside the same
nonce-delimited untrusted block that consolidation prompts use, labelled as data
rather than instruction. And nothing a search or a fetch returns can become a
memory: the transcript that candidate facts are extracted from is built from
what people said, and a tool result is not that. A page saying *"remember that
X"* does not make Kasa believe X.

## Reading a page

`web_fetch` retrieves one http(s) url and hands back its text, so Kasa can
finish the errand search starts — search, open the result that looks
authoritative, read it, answer.

On by default, unlike search: search needs a key somebody went and got, and this
needs nothing. What makes it safe is the guard rather than the switch, and the
switch is there for an install that wants the outbound surface gone:

```toml
[fetch]
enabled     = false   # the tool is not registered at all
max_chars   = 20_000  # what reaches the model
max_bytes   = 2_000_000
max_redirects   = 4
timeout_seconds = 15.0
```

Where a request may go is decided before a byte is sent, and decided again on
every redirect. Names are resolved and the *addresses* judged — loopback,
private, link-local (which is where cloud metadata services live), and anything
else not on the public internet is refused, in v4 and in the v6 spellings of the
same address. Every answer has to be public, not just the first, and the address
that was approved is the one connected to, so a name cannot resolve to something
else between the check and the socket. Only http and https, only ports 80 and
443, no credentials in the url. Nothing outbound carries anything this daemon
knows: no cookies, no `Authorization`, and no header the model can name — the
tool takes a url and nothing else.

Two limits worth knowing, because they are the ones you will meet:

- **No scripts are run.** A page that draws itself in the browser comes back
  with its content missing. A cinema timetable loaded over XHR is not on the
  page Kasa reads.
- **Long pages are cut**, and say so.

`kasa doctor` reports whether fetching is on and what the limits are.

## Development

```bash
uv sync --all-extras --dev   # `--all-extras` brings in Slack
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy kasa
```

The black-box QA suite starts the real terminal command against a local fake
OpenAI-compatible server, so it needs neither network access nor API keys:

```bash
uv run pytest tests/e2e -q
```

Use `-m "not e2e"` when iterating on lower-level tests only. Tests marked
`external` are reserved for optional smoke tests against real services and are
not part of the deterministic suite.

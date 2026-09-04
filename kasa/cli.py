"""Command-line entry point."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from itertools import chain
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from kasa import __version__
from kasa.adapters.cli import run_repl
from kasa.config import Config, config_path, load_config
from kasa.core.agent import Agent
from kasa.core.context import ContextPacker
from kasa.core.inbox import Inbox
from kasa.core.memory_tools import memory_tools
from kasa.core.tools import ToolRegistry, builtin_tools
from kasa.doctor import Report, Status, diagnose, verify_repo_visibility
from kasa.errors import KasaError
from kasa.init import run_init
from kasa.llm.tokens import default_tokenizer
from kasa.memory.document import Problem
from kasa.memory.explain import render_trace
from kasa.memory.index import MemoryIndex
from kasa.memory.ltm import MemoryStore, MemoryStoreError
from kasa.memory.manifest import Manifest
from kasa.memory.retrieve import Retriever
from kasa.redact import Redactor
from kasa.runner.jobs import default_specs
from kasa.runner.scheduler import Scheduler, UnknownJob
from kasa.store import Store

app = typer.Typer(
    name="kasa",
    help="A long-running, memory-native agent server.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
app.add_typer(db_app, name="db")
inbox_app = typer.Typer(help="The durable ingress queue.", no_args_is_help=True)
app.add_typer(inbox_app, name="inbox")
job_app = typer.Typer(help="Background jobs.", no_args_is_help=True)
app.add_typer(job_app, name="job")
review_app = typer.Typer(help="Things Kasa decided not to decide alone.", no_args_is_help=True)
app.add_typer(review_app, name="review")

console = Console()
#: Everything on stderr is a single diagnostic line, never a table, and it
#: usually carries a path somebody is about to act on. Wrapping it at rich's
#: 80-column fallback split those messages mid-word when stderr was a pipe.
err = Console(stderr=True, soft_wrap=True)

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Path to config.toml.")]


@app.command()
def version() -> None:
    """Print the version."""
    emit(__version__)


@app.command()
def init(config: ConfigOption = None) -> None:
    """Interactive setup: the memory repo, the models, and the config file."""

    async def main() -> None:
        result = await run_init(ConsolePrompter(), path=config)
        console.print()
        console.print(f"[green]Ready.[/green] Memory repo: {result.repo_path}", soft_wrap=True)
        console.print("Run [bold]kasa run[/bold] to start talking to it.")

    _run(main())


@app.command()
def run(
    config: ConfigOption = None,
    slack: Annotated[bool, typer.Option("--slack", help="Serve Slack over Socket Mode.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start Kasa.

    With no flags this is the terminal adapter. `--slack` runs the Socket Mode
    daemon instead: the connection is outbound, so there is no public ingress
    to expose and nothing to put a certificate on.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = _load(config)
    Redactor.from_config(cfg).install()
    if slack:
        if not cfg.slack.configured:
            err.print("[red]error[/red]: no Slack tokens configured; run `kasa init`")
            raise typer.Exit(1)
        _run(_serve_slack(cfg))
        return
    _run(_repl(cfg))


@app.command("config")
def show_config(config: ConfigOption = None) -> None:
    """Print the resolved configuration.

    Safe to paste: the file holds environment-variable *names*, never secrets.
    """
    cfg = _load(config)
    # The path goes to stderr: it says where the JSON came from, which makes it
    # a comment on the output rather than part of it. On stdout it was the
    # first thing `kasa config | jq` choked on.
    err.print(f"[dim]{config or config_path()}[/dim]")
    console.print_json(json.dumps(cfg.redacted(), indent=2))


@app.command()
def reindex(
    config: ConfigOption = None,
    full: Annotated[bool, typer.Option("--full", help="Rebuild from scratch.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Rebuild the search index from the memory repo.

    Safe at any time: the index is derived, and the repo is the source of
    truth. Not safe twice at once — a second run is refused rather than
    interleaved with the first.
    """
    # Quiet by default. This command has its own report, and without any
    # logging configured the WARNING records behind that report went out
    # through Python's `lastResort` handler — unformatted, above the summary,
    # saying what the summary was about to say (#77). `-v` matches `kasa run`.
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async def main() -> None:
        cfg = _load(config)
        if not cfg.ltm.configured:
            err.print("[red]error[/red]: no memory repo configured; run `kasa init`")
            raise typer.Exit(1)
        async with await Store.open(cfg.store.resolved()) as store:
            # Opened before anything is written, because opening is what
            # validates the clone. Discovering afterwards that there is no repo
            # to rebuild the manifest from would leave the index rebuilt, the
            # manifest untouched, and the command reporting failure.
            memory = await MemoryStore.open(cfg, store)
            embedding = cfg.llm.get("embedding")
            registry = cfg.build_registry(store=store) if embedding else None
            try:
                result = await MemoryIndex(
                    store,
                    cfg.ltm.resolved_clone_path(),
                    embedder=registry.embed if registry else None,
                    embedding_model=embedding.model if embedding else None,
                ).reindex(full=full)
            finally:
                if registry is not None:
                    await registry.aclose()
            # Both artifacts are derived from the repo, and rebuilding only the
            # SQLite half is what let them disagree about which memories exist.
            manifest = await memory.refresh_manifest()
        console.print(result.summary())
        console.print(f"[dim]{manifest.summary()}[/dim]")
        # One line per file. The index and the manifest read the same files and
        # refuse them for the same reasons, so a file both halves rejected was
        # named twice — once without a reason (#77). They are not always the
        # same set: a duplicate id indexes fine and only the manifest minds.
        for problem in _one_per_file(result.problems, manifest.problems):
            err.print(f"[yellow]![/yellow] {problem.path}: {problem.reason}")

    _run(main())


@app.command()
def why(
    question: Annotated[str, typer.Argument(help="The question to trace retrieval for.")],
    config: ConfigOption = None,
    scope: Annotated[str, typer.Option(help="Answer as a session in this scope.")] = "workspace",
) -> None:
    """Show the full retrieval trace for a question.

    Every complaint about this system arrives as "why did it not remember X".
    This is the answer: the query, every candidate and its scores, what scope
    filtering removed, and what actually fitted in the budget.
    """

    async def main() -> None:
        cfg = _load(config)
        if not cfg.ltm.configured:
            err.print("[red]error[/red]: no memory repo configured; run `kasa init`")
            raise typer.Exit(1)
        async with await Store.open(cfg.store.resolved()) as store:
            embedding = cfg.llm.get("embedding")
            registry = cfg.build_registry(store=store) if embedding else None
            retriever = Retriever(
                store,
                tokenizer=default_tokenizer(),
                budget_tokens=cfg.context.tokens_for_retrieval(),
                # Scrubbed here too, even though this prints to a terminal
                # rather than to a model. `kasa why` output is what gets pasted
                # into a bug report; the file itself is one `cat` away for an
                # operator who actually needs the raw value.
                scrub=Redactor.from_config(cfg).scrub,
                embedder=registry.embed if registry else None,
                embedding_model=embedding.model if embedding else None,
            )
            try:
                retrieval = await retriever.retrieve(question, scope=scope, explain=True)
            finally:
                if registry is not None:
                    await registry.aclose()
        # markup=False because a memory id arrives as `[[mem_01...]]`, and rich
        # reads square brackets as style tags — it renders the ids away entirely.
        # soft_wrap because the score table is meant to be read in columns.
        console.print(render_trace(retrieval), highlight=False, markup=False, soft_wrap=True)

    _run(main())


@app.command()
def audit(config: ConfigOption = None) -> None:
    """List every long-term memory by visibility scope.

    The corpus is read directly instead of trusting the committed manifest: an
    audit whose input can be stale is exactly where a newly added private file
    could disappear from view.
    """
    cfg = _load(config)
    if not cfg.ltm.configured:
        err.print("[red]error[/red]: no memory repo configured; run `kasa init`")
        raise typer.Exit(1)

    root = cfg.ltm.resolved_clone_path()
    manifest, problems = Manifest.rebuild(root)
    table = Table(show_header=True)
    table.add_column("scope", no_wrap=True)
    table.add_column("id", no_wrap=True)
    table.add_column("path", overflow="fold")
    table.add_column("title", overflow="fold")
    for memory_id, entry in sorted(
        manifest.memories.items(), key=lambda item: (item[1].visibility, item[1].path)
    ):
        table.add_row(entry.visibility, memory_id, entry.path, entry.title)
    console.print(table)
    scope_count = len({entry.visibility for entry in manifest.memories.values()})
    console.print(f"[dim]{len(manifest)} memory(s) across {scope_count} scope(s)[/dim]")
    for problem in problems:
        err.print(f"[yellow]![/yellow] {problem.path}: {problem.reason}")
    if problems:
        raise typer.Exit(1)


@app.command()
def doctor(config: ConfigOption = None) -> None:
    """Check config, tokens, repo privacy, and the local clone.

    Exits non-zero if anything failed, so it is usable as a health check.
    """

    async def main() -> None:
        report = await diagnose(_load(config), path=config or config_path())
        _print_report(report)
        if not report.ok:
            raise typer.Exit(1)

    _run(main())


@app.command()
def cost(config: ConfigOption = None) -> None:
    """Show token and spend totals recorded so far."""

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            rows = await store.cost_summary()
            if not rows:
                console.print("[dim]no calls recorded yet[/dim]")
                return
            table = Table(show_header=True)
            table.add_column("day", no_wrap=True)
            table.add_column("role", no_wrap=True)
            table.add_column("job", no_wrap=True)
            # Folded, not truncated, for the same reason `doctor` folds its
            # detail: this column is the row's identity, and a provider's
            # canonical id is long. Truncated at rich's 80-column fallback,
            # two models from one provider became the same row as soon as the
            # output was piped anywhere (#80).
            table.add_column("model", overflow="fold")
            # The figures are what the table is for, so they do not wrap while
            # a long name is being folded beside them.
            for column in ("calls", "in", "out", "cached", "usd"):
                table.add_column(column, no_wrap=True)
            for row in rows:
                usd = row["cost_usd"]
                table.add_row(
                    row["day"],
                    row["role"],
                    row["job_kind"],
                    row["model"],
                    str(row["calls"]),
                    str(row["input_tokens"] or 0),
                    str(row["output_tokens"] or 0),
                    str(row["cache_read_tokens"] or 0),
                    f"{usd:.4f}" if usd else "[dim]unpriced[/dim]",
                )
            console.print(table)

    _run(main())


#: Printed in this order whatever the database returns, so a state with no rows
#: reads as zero rather than as a missing line.
INBOX_STATES = ("pending", "leased", "done", "failed")


@inbox_app.command("status")
def inbox_status(config: ConfigOption = None) -> None:
    """Show what is queued, and what stopped being retried.

    `failed` is the one to read. Those are messages somebody sent that Kasa
    gave up on, and nothing retries them until `kasa inbox retry` says so.
    """

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            inbox = Inbox(store)
            counts = await inbox.counts()
            failed = await inbox.dead_letters()
        if not counts:
            console.print("[dim]nothing has arrived yet[/dim]")
            return
        table = Table(show_header=True)
        table.add_column("state", no_wrap=True)
        table.add_column("events", no_wrap=True)
        for state in INBOX_STATES:
            table.add_row(state, str(counts.get(state, 0)))
        console.print(table)
        for row in failed:
            err.print(
                f"[red]![/red] {row['id']} {row['source']}:{row['external_id']}"
                f" after {row['attempts']} attempt(s) — {row['last_error']}"
            )

    _run(main())


@inbox_app.command("retry")
def inbox_retry(config: ConfigOption = None) -> None:
    """Put every dead-lettered event back in the queue.

    Dead-lettering is a pause for a human, not a delete: the message is still
    there, with its payload, and this is how it gets another five attempts.
    """

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            revived = await Inbox(store).revive()
        if revived:
            console.print(f"requeued {revived} event(s)")
        else:
            console.print("[dim]no dead letters[/dim]")

    _run(main())


JOB_STATES = ("pending", "leased", "done", "failed")


@job_app.command("run")
def job_run(
    kind: Annotated[str, typer.Argument(help="Which job to run.")],
    config: ConfigOption = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run one job now, in this process.

    It is still a row, so a job run this way leases, retries and dead-letters
    exactly like one the daemon picked up — the difference is only who is
    waiting for it.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            scheduler = Scheduler(store, default_specs(cfg, store))
            try:
                job = await scheduler.run_now(kind)
            except UnknownJob as exc:
                err.print(f"[red]error[/red]: {exc}")
                raise typer.Exit(1) from exc
        if job["state"] == "done":
            console.print(f"{kind} finished")
            return
        # A pending row that has been attempted is waiting out its backoff, and
        # "pending" reads as though nothing happened. A row with no recorded
        # error did not fail — it never ran — and `None` is not a reason.
        state = "retrying" if job["state"] == "pending" else job["state"]
        err.print(f"[red]{kind} {state}[/red]: {job['last_error'] or 'it never ran'}")
        raise typer.Exit(1)

    _run(main())


@job_app.command("list")
def job_list(config: ConfigOption = None) -> None:
    """Show what each job is doing, and when it last ran."""

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            rows = await store.job_overview()
            failed = await store.failed_jobs()
            known = Scheduler(store, default_specs(cfg, store)).kinds
        if not rows:
            console.print(
                f"[dim]nothing queued yet; this build knows: {', '.join(known) or 'no jobs'}[/dim]"
            )
            return
        table = Table(show_header=True)
        table.add_column("kind", no_wrap=True)
        for column in (*JOB_STATES, "last run"):
            table.add_column(column, no_wrap=True)
        for kind, states in _by_kind(rows).items():
            table.add_row(
                kind,
                *(str(states.get(state, ("0", ""))[0]) for state in JOB_STATES),
                max((last for _, last in states.values() if last), default="[dim]never[/dim]"),
            )
        console.print(table)
        for row in failed:
            err.print(
                f"[red]![/red] {row['kind']} {row['id']} after {row['attempts']} attempt(s)"
                f" — {row['last_error']}"
            )

    _run(main())


@review_app.command("list")
def review_list(config: ConfigOption = None) -> None:
    """Show what is waiting on a person.

    One thing raises these today: a claim already in long-term memory whose
    source message was edited or deleted afterwards (#25). Kasa will not
    rewrite the corpus over a retraction — the memory may have been merged,
    superseded or built on since — so it says what it noticed and stops.
    """

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            rows = await store.open_reviews()
        if not rows:
            console.print("[dim]nothing waiting[/dim]")
            return
        table = Table(show_header=True)
        for column in ("id", "raised", "why", "subject"):
            table.add_column(column, no_wrap=column != "subject", overflow="fold")
        for row in rows:
            table.add_row(
                str(row["id"]), str(row["created_at"])[:16], str(row["kind"]), str(row["subject"])
            )
        console.print(table)
        for row in rows:
            # The detail carries the claim, and the claim may have come from a
            # DM. Printed here on the operator's own terminal, with the scope
            # beside it, and never anywhere a channel can read.
            console.print(f"\n[bold]{row['id']}[/bold] [dim]({row['scope']})[/dim]")
            console.print(f"  {row['detail']}")

    _run(main())


@review_app.command("done")
def review_done(
    review_id: Annotated[str, typer.Argument(help="The review to close.")],
    config: ConfigOption = None,
) -> None:
    """Mark a review as dealt with. Kasa does not check that it was."""

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            closed = await store.resolve_review(review_id)
        if not closed:
            err.print(f"[yellow]![/yellow] no open review {review_id}")
            raise typer.Exit(1)
        console.print(f"[green]closed[/green] {review_id}")

    _run(main())


@job_app.command("retry")
def job_retry(config: ConfigOption = None) -> None:
    """Put every dead-lettered job back in the queue, due now."""

    async def main() -> None:
        cfg = _load(config)
        async with await Store.open(cfg.store.resolved()) as store:
            revived = await store.revive_failed_jobs()
        console.print(f"requeued {revived} job(s)" if revived else "[dim]no dead letters[/dim]")

    _run(main())


def _by_kind(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, tuple[str, str]]]:
    """`{kind: {state: (count, last run)}}`, from one row per (kind, state)."""
    grouped: dict[str, dict[str, tuple[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["kind"]), {})[str(row["state"])] = (
            str(row["n"]),
            str(row["last_run"] or ""),
        )
    return grouped


@db_app.command("migrate")
def db_migrate(config: ConfigOption = None) -> None:
    """Apply pending migrations."""

    async def main() -> None:
        cfg = _load(config)
        path = cfg.store.resolved()
        async with await Store.open(path) as store:
            applied = await store.migrate()
        if applied:
            console.print(f"applied {len(applied)} migration(s): {', '.join(applied)}")
        else:
            console.print(f"[dim]{path} is up to date[/dim]", soft_wrap=True)

    _run(main())


@db_app.command("path")
def db_path(config: ConfigOption = None) -> None:
    """Print the database location."""
    emit(str(_load(config).store.resolved()))


def _one_per_file(*groups: Sequence[Problem]) -> list[Problem]:
    """Merge problem lists by path, keeping the first reason given for each."""
    seen: dict[str, Problem] = {}
    for problem in chain.from_iterable(groups):
        seen.setdefault(problem.path, problem)
    return list(seen.values())


def emit(value: str) -> None:
    """Print one value for something else to read.

    A command whose whole output is a value — `kasa db path`, `kasa version` —
    must hand it over unchanged, and rich's defaults do three things to it.
    `soft_wrap` because rich falls back to 80 columns when stdout is not a
    terminal and hard-wraps there, which is how `$(kasa db path)` came back
    with a newline in the middle of it. `markup=False` because a path may
    legitimately contain square brackets, and rich reads those as style tags
    and deletes them. `highlight=False` because ANSI colour in a captured
    value is no more use than a newline in it (#68).
    """
    console.print(value, soft_wrap=True, markup=False, highlight=False)


# -- wiring ------------------------------------------------------------------


class ConsolePrompter:
    """`kasa.init.Prompter`, backed by a terminal."""

    def ask(self, question: str, *, default: str | None = None) -> str:
        # An empty default is a real answer ("no base URL"), so it is offered as
        # one rather than becoming a required question.
        return str(typer.prompt(question, default=default if default is not None else ""))

    def choose(self, question: str, choices: tuple[str, ...], *, default: str) -> str:
        options = "/".join(choices)
        while True:
            answer = str(typer.prompt(f"{question} [{options}]", default=default)).strip()
            if answer in choices:
                return answer
            self.warn(f"Pick one of: {options}")

    def select(self, question: str, options: tuple[str, ...], *, default: str) -> str:
        console.print(f"{question}:")
        for number, option in enumerate(options, start=1):
            console.print(f"  {number}. {option}")
        answer = str(typer.prompt("Number or model name", default=default)).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        return answer

    def confirm(self, question: str, *, default: bool = False) -> bool:
        return bool(typer.confirm(question, default=default))

    def say(self, text: str) -> None:
        console.print(text, soft_wrap=True)

    def warn(self, text: str) -> None:
        err.print(f"[yellow]![/yellow] {text}")


def _load(path: Path | None) -> Config:
    try:
        return load_config(path)
    except KasaError as exc:
        err.print(f"[red]error[/red]: {exc}")
        raise typer.Exit(1) from exc


_STATUS_STYLE = {
    Status.OK: "[green]ok[/green]",
    Status.WARN: "[yellow]warn[/yellow]",
    Status.FAIL: "[red]FAIL[/red]",
    Status.SKIP: "[dim]skip[/dim]",
}


def _print_report(report: Report) -> None:
    table = Table(show_header=False, box=None)
    table.add_column("status", no_wrap=True)
    table.add_column("check", no_wrap=True)
    # Folded, not truncated: the detail is usually a path, and a path with its
    # middle replaced by an ellipsis is not something you can act on.
    table.add_column("detail", overflow="fold")
    for check in report.checks:
        table.add_row(_STATUS_STYLE[check.status], check.name, check.detail)
    console.print(table)
    if not report.ok:
        console.print(f"\n[red]{len(report.failed)} check(s) failed.[/red]")


@asynccontextmanager
async def _agent(cfg: Config) -> AsyncIterator[Agent]:
    """Everything a surface talks to, built once and torn down on every path.

    Shared by the terminal and by Slack, so a conversation held in one is
    indistinguishable in the database from one held in the other.
    """
    # A repo that silently became public is a serious incident, so visibility is
    # re-checked on every start rather than trusted from setup time.
    await verify_repo_visibility(cfg)

    # The store must be closed on every path, including a failure to build the
    # registry: aiosqlite holds a non-daemon thread per connection, so leaking
    # one leaves the process alive after the error has already been printed.
    tokenizer = default_tokenizer()
    # One redactor for the whole session: it reads the environment once, and
    # both places that send text onwards — recalled memory and tool results —
    # have to agree about what counts as a secret.
    scrub = Redactor.from_config(cfg).scrub
    async with await Store.open(cfg.store.resolved()) as store:
        registry = cfg.build_registry(store=store)
        tools = builtin_tools()
        retriever = None

        if cfg.ltm.configured:
            try:
                memory = await MemoryStore.open(cfg, store)
            except MemoryStoreError as exc:
                # Running without memory beats not running: the conversation
                # still works, and `kasa doctor` says what is wrong.
                err.print(f"[yellow]![/yellow] long-term memory unavailable — {exc}")
            else:
                embedding = cfg.llm.get("embedding")
                retriever = Retriever(
                    store,
                    tokenizer=tokenizer,
                    budget_tokens=cfg.context.tokens_for_retrieval(),
                    scrub=scrub,
                    # The one retriever that serves conversations, so the one
                    # that counts a recall. `kasa why` traces what *would* be
                    # recalled and `promote` reads competition for a plan;
                    # counting either would let a debugging session decide what
                    # stays in long-term memory.
                    record_hits=True,
                    embedder=registry.embed if embedding else None,
                    embedding_model=embedding.model if embedding else None,
                )
                tools += memory_tools(retriever=retriever, memory=memory, store=store)

        try:
            yield Agent(
                registry=registry,
                store=store,
                tools=ToolRegistry(tools, scrub=scrub),
                packer=ContextPacker(cfg.context.to_budget(), tokenizer=tokenizer),
                config=cfg.agent_config(),
                retriever=retriever,
            )
        finally:
            await registry.aclose()


async def _repl(cfg: Config) -> None:
    async with _agent(cfg) as agent:
        await run_repl(agent, console, scrub=Redactor.from_config(cfg).scrub)


async def _serve_slack(cfg: Config) -> None:
    # Imported here because it is the `slack` extra. `kasa run` with no flags
    # must keep working on an install that never asked for Slack.
    from kasa.adapters.slack import SlackAdapter

    async with _agent(cfg) as agent:
        adapter = await SlackAdapter.connect(agent, cfg.slack)
        # The daemon is where background work belongs: it is the process that
        # stays up, and `kasa run` on a terminal is not.
        scheduler = Scheduler(
            agent.store,
            default_specs(cfg, agent.store, agent.registry),
            pause_when=agent.registry.meter.daily_ceiling_reached,
        )
        console.print(
            f"[green]Connected[/green] to Slack as {adapter.context.bot_user_id}"
            f" in {adapter.context.team_id}. Ctrl-C to stop."
        )

        def stop() -> None:
            # A daemon is stopped by a signal, and `stop()` drains rather than
            # dropping: in-flight turns and jobs finish, and their rows
            # complete. Without this, systemd restarting Kasa replays
            # everything that was running at the time.
            adapter.runtime.stop()
            scheduler.stop()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop)
        try:
            await adapter.start()
            await asyncio.gather(adapter.runtime.run(), scheduler.run())
        finally:
            await adapter.aclose()


def _run(coro: object) -> None:
    try:
        asyncio.run(coro)  # type: ignore[arg-type]
    except KasaError as exc:
        err.print(f"[red]error[/red]: {exc}")
        raise typer.Exit(1) from exc
    except KeyboardInterrupt:
        raise typer.Exit(130) from None


def main() -> None:
    app()


if __name__ == "__main__":
    main()

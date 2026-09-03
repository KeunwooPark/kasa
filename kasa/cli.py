"""Command-line entry point."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from kasa import __version__
from kasa.adapters.cli import run_repl
from kasa.config import Config, config_path, load_config
from kasa.core.agent import Agent
from kasa.core.context import ContextPacker
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
from kasa.memory.retrieve import Retriever
from kasa.redact import Redactor
from kasa.store import Store

app = typer.Typer(
    name="kasa",
    help="A long-running, memory-native agent server.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
app.add_typer(db_app, name="db")

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
    cli: Annotated[bool, typer.Option("--cli", help="Run the terminal adapter.")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start Kasa.

    The terminal is the only surface today; Slack arrives in v2.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not cli:
        raise typer.BadParameter("the terminal is the only surface today; Slack arrives in v2")
    cfg = _load(config)
    Redactor.from_config(cfg).install()
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

    Safe at any time: the index is derived, and the repo is the source of truth.
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
            result = await MemoryIndex(store, cfg.ltm.resolved_clone_path()).reindex(full=full)
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
            retriever = Retriever(
                store,
                tokenizer=default_tokenizer(),
                budget_tokens=cfg.context.tokens_for_retrieval(),
                # Scrubbed here too, even though this prints to a terminal
                # rather than to a model. `kasa why` output is what gets pasted
                # into a bug report; the file itself is one `cat` away for an
                # operator who actually needs the raw value.
                scrub=Redactor.from_config(cfg).scrub,
            )
            retrieval = await retriever.retrieve(question, scope=scope, explain=True)
        # markup=False because a memory id arrives as `[[mem_01...]]`, and rich
        # reads square brackets as style tags — it renders the ids away entirely.
        # soft_wrap because the score table is meant to be read in columns.
        console.print(render_trace(retrieval), highlight=False, markup=False, soft_wrap=True)

    _run(main())


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
            table = Table("role", "model", "calls", "in", "out", "cached", "usd")
            for row in rows:
                usd = row["cost_usd"]
                table.add_row(
                    row["role"],
                    row["model"],
                    str(row["calls"]),
                    str(row["input_tokens"] or 0),
                    str(row["output_tokens"] or 0),
                    str(row["cache_read_tokens"] or 0),
                    f"{usd:.4f}" if usd else "[dim]unpriced[/dim]",
                )
            console.print(table)

    _run(main())


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


async def _repl(cfg: Config) -> None:
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
                retriever = Retriever(
                    store,
                    tokenizer=tokenizer,
                    budget_tokens=cfg.context.tokens_for_retrieval(),
                    scrub=scrub,
                )
                tools += memory_tools(retriever=retriever, memory=memory, store=store)

        try:
            await run_repl(
                Agent(
                    registry=registry,
                    store=store,
                    tools=ToolRegistry(tools, scrub=scrub),
                    packer=ContextPacker(cfg.context.to_budget(), tokenizer=tokenizer),
                    config=cfg.agent_config(),
                    retriever=retriever,
                ),
                console,
            )
        finally:
            await registry.aclose()


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

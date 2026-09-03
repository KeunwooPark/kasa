"""Command-line entry point."""

from __future__ import annotations

import asyncio
import json
import logging
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
from kasa.core.tools import ToolRegistry, builtin_tools
from kasa.errors import KasaError
from kasa.init import run_init
from kasa.llm.tokens import default_tokenizer
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
err = Console(stderr=True)

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Path to config.toml.")]


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


@app.command()
def init(config: ConfigOption = None) -> None:
    """Interactive setup: the memory repo, the models, and the config file."""

    async def main() -> None:
        result = await run_init(ConsolePrompter(), path=config)
        console.print()
        console.print(f"[green]Ready.[/green] Memory repo: {result.repo_path}")
        console.print("Run [bold]kasa run[/bold] to start talking to it.")

    _run(main())


@app.command()
def run(
    config: ConfigOption = None,
    cli: Annotated[bool, typer.Option("--cli", help="Run the terminal adapter.")] = True,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start Kasa.

    v0 ships the terminal adapter only; Slack arrives in v2.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not cli:
        raise typer.BadParameter("only --cli is supported at v0")
    _run(_repl(_load(config)))


@app.command("config")
def show_config(config: ConfigOption = None) -> None:
    """Print the resolved configuration.

    Safe to paste: the file holds environment-variable *names*, never secrets.
    """
    cfg = _load(config)
    console.print(f"[dim]{config or config_path()}[/dim]")
    console.print_json(json.dumps(cfg.redacted(), indent=2))


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
            console.print(f"[dim]{path} is up to date[/dim]")

    _run(main())


@db_app.command("path")
def db_path(config: ConfigOption = None) -> None:
    """Print the database location."""
    console.print(str(_load(config).store.resolved()))


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
        console.print(text)

    def warn(self, text: str) -> None:
        err.print(f"[yellow]![/yellow] {text}")


def _load(path: Path | None) -> Config:
    try:
        return load_config(path)
    except KasaError as exc:
        err.print(f"[red]error[/red]: {exc}")
        raise typer.Exit(1) from exc


async def _repl(cfg: Config) -> None:
    # The store must be closed on every path, including a failure to build the
    # registry: aiosqlite holds a non-daemon thread per connection, so leaking
    # one leaves the process alive after the error has already been printed.
    async with await Store.open(cfg.store.resolved()) as store:
        registry = cfg.build_registry(store=store)
        try:
            await run_repl(
                Agent(
                    registry=registry,
                    store=store,
                    tools=ToolRegistry(builtin_tools()),
                    packer=ContextPacker(cfg.context.to_budget(), tokenizer=default_tokenizer()),
                    config=cfg.agent_config(),
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

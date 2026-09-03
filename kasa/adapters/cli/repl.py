"""Interactive terminal adapter.

Exists so the agent loop can be built and debugged without Slack in the way, but
it is a real adapter, not a harness: it goes through the same store, session and
agent path that the Slack adapter will use in v2, so a conversation held here is
indistinguishable in the database from one held in a channel.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from ulid import ULID

from kasa.core.agent import Agent, AgentResult
from kasa.llm.types import Delta, TextDelta

PROMPT = "\n› "  # noqa: RUF001 - a prompt glyph, not punctuation

HELP = """\
[bold]Commands[/bold]
  /reset     start a fresh session
  /session   show the current session id and message count
  /tokens    show token and cost totals for this process
  /trace     show the context packing trace from the last turn
  /help      this list
  /quit      exit (ctrl-d also works)
"""


@dataclass
class Repl:
    agent: Agent
    console: Console
    session_id: str
    last_result: AgentResult | None = None

    async def run(self) -> None:
        self.console.print(
            Panel.fit(
                "[bold]kasa[/bold] — v0, conversation only. No memory yet.\n"
                "[dim]/help for commands[/dim]",
                border_style="cyan",
            )
        )
        while True:
            try:
                line = (await asyncio.to_thread(input, PROMPT)).strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                return

            if not line:
                continue
            if line.startswith("/"):
                if await self._meta(line):
                    return
                continue

            await self._turn(line)

    async def _turn(self, text: str) -> None:
        first = True

        async def on_delta(delta: Delta) -> None:
            nonlocal first
            if isinstance(delta, TextDelta):
                if first:
                    self.console.print()
                    first = False
                # Written raw: model output must never be parsed as markup.
                self.console.file.write(delta.text)
                self.console.file.flush()

        try:
            self.last_result = await self.agent.respond(
                self.session_id, text, surface="cli", on_delta=on_delta
            )
        except asyncio.CancelledError:
            self.console.print("\n[yellow]interrupted[/yellow]")
            raise
        except Exception as exc:
            self.console.print(f"\n[red]{type(exc).__name__}[/red]: {exc}")
            return

        self.console.print()
        result = self.last_result
        if (note := result.note) is not None:
            self.console.print(f"[yellow]…[/yellow] {note}")
        if result.tool_calls:
            self.console.print(
                f"[dim]{result.tool_calls} tool call(s), {result.iterations} iteration(s)[/dim]"
            )

    async def _meta(self, line: str) -> bool:
        """Handle a slash command. Returns True if the REPL should exit."""
        command = line.split()[0].lower()
        match command:
            case "/quit" | "/exit":
                return True
            case "/help":
                self.console.print(HELP)
            case "/reset":
                self.session_id = f"cli:{ULID()}"
                self.console.print(f"[dim]new session {self.session_id}[/dim]")
            case "/session":
                count = await self.agent.store.message_count(self.session_id)
                self.console.print(f"[dim]{self.session_id} — {count} messages[/dim]")
            case "/tokens":
                meter = self.agent.registry.meter
                usage = meter.total
                cost = f"${meter.total_usd:.4f}" if meter.total_usd else "unpriced"
                self.console.print(
                    f"[dim]in {usage.input_tokens} · out {usage.output_tokens} · "
                    f"cached {usage.cache_read_tokens} · {cost}[/dim]"
                )
            case "/trace":
                if self.last_result and self.last_result.trace:
                    self.console.print(f"[dim]{self.last_result.trace.render()}[/dim]")
                else:
                    self.console.print("[dim]no turn yet[/dim]")
            case _:
                self.console.print(f"[dim]unknown command {command}; /help for the list[/dim]")
        return False


async def run_repl(agent: Agent, console: Console | None = None) -> None:
    repl = Repl(
        agent=agent,
        console=console or Console(),
        session_id=f"cli:{ULID()}",
    )
    await repl.run()

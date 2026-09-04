from __future__ import annotations

from pathlib import Path

from kasa.core.agent import AgentResult
from kasa.core.events import InboundEvent
from kasa.core.runtime import Runtime, one_message
from kasa.redact import Redactor
from kasa.store import Store


class FakeAgent:
    def __init__(self, store: Store) -> None:
        self.store = store


async def test_durable_inbox_contains_only_the_scrubbed_event(tmp_path: Path) -> None:
    secret = "sk-ant-this-must-not-enter-the-inbox"
    async with await Store.open(tmp_path / "inbox.db") as store:
        agent = FakeAgent(store)

        async def sink(event: InboundEvent, result: AgentResult) -> None:
            pass

        runtime = Runtime(agent, one_message(sink), scrub=Redactor().scrub)  # type: ignore[arg-type]
        await runtime.submit(
            InboundEvent(source="slack", external_id="E1", session_id="slack:T:C:1", text=secret)
        )
        rows = await store.raw("SELECT payload FROM inbox WHERE external_id = ?", ("E1",))

    assert secret not in rows[0]["payload"]
    event = InboundEvent.from_json(rows[0]["payload"])
    assert event.text == "[redacted]"
    assert event.credential_scrubbed

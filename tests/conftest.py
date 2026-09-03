from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from kasa.llm.tokens import HeuristicTokenizer, Tokenizer
from kasa.store import Store


@pytest.fixture
def tokenizer() -> Tokenizer:
    # The heuristic tokenizer, always: tests must not depend on whether the
    # optional tiktoken extra happens to be installed.
    return HeuristicTokenizer()


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[Store]:
    opened = await Store.open(tmp_path / "kasa.db")
    try:
        yield opened
    finally:
        await opened.close()


def mock_client(
    handler: Callable[[httpx.Request], httpx.Response], base_url: str = "https://api.test/v1"
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=base_url)


def sse(events: list[tuple[str | None, dict[str, object]]], *, done: bool = False) -> bytes:
    """Render server-sent events the way the real APIs do."""
    chunks = []
    for name, data in events:
        prefix = f"event: {name}\n" if name else ""
        chunks.append(f"{prefix}data: {json.dumps(data)}\n\n")
    if done:
        chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode()

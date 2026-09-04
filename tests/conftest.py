from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import httpx
import pytest

from kasa.llm.tokens import HeuristicTokenizer, Tokenizer
from kasa.store import Store
from kasa.vault import VAULT_ENV, clear_cache


@pytest.fixture(autouse=True)
def isolated_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point every test at an empty vault under `tmp_path`.

    Autouse and not optional. Without it the default path is the real
    `user_data_dir`, so a developer who has run `kasa vault set` would have
    their own credentials resolved inside the suite — which would make tests
    pass on their machine and fail in CI, and is a live secret in a test
    process either way.
    """
    monkeypatch.setenv(VAULT_ENV, str(tmp_path / "vault" / "vault.json"))
    clear_cache()
    yield
    clear_cache()


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


async def until(predicate: Callable[[], bool], *, within: float = 10.0) -> None:
    """Wait for a background loop to get somewhere, without sleeping blind.

    The loops under test are driven by tasks rather than by the test, so there
    is nothing to await directly. A deadline is what keeps a broken one from
    hanging the suite instead of failing it.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("timed out waiting for the loop under test")

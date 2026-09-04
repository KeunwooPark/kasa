"""Brave Search.

Chosen as the first backend because it returns snippets rather than pages: the
result of a call is a few hundred words of somebody else's text, not a document
to be fetched, rendered, and stripped. That keeps both the token cost and the
injection surface small, which is what makes web search shippable ahead of a
`web_fetch` tool rather than alongside one.
"""

from __future__ import annotations

from typing import Any

import httpx

from kasa.errors import SearchError
from kasa.search.base import SearchResult

DEFAULT_BASE_URL = "https://api.search.brave.com"
SEARCH_PATH = "/res/v1/web/search"

#: Brave's own ceiling for one request. Asking for more is a 422, not a
#: truncated list.
MAX_COUNT = 20


class BraveSearch:
    name = "brave"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=(base_url or DEFAULT_BASE_URL).rstrip("/"),
            timeout=timeout,
        )
        # Set on the client rather than passed to its constructor, so an
        # injected client is authenticated too. A transport supplied by a test
        # that silently sent no key would make every auth test vacuous.
        self._client.headers.update(
            {
                "accept": "application/json",
                "accept-encoding": "gzip",
                "x-subscription-token": api_key,
            }
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str, *, count: int) -> list[SearchResult]:
        params: dict[str, str | int] = {"q": query, "count": min(max(count, 1), MAX_COUNT)}
        try:
            resp = await self._client.get(SEARCH_PATH, params=params)
        except httpx.TimeoutException as exc:
            raise SearchError(f"[{self.name}] search timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise SearchError(f"[{self.name}] search failed: {exc}") from exc

        if resp.status_code >= 400:
            raise SearchError(_message_for(self.name, resp))
        try:
            body: dict[str, Any] = resp.json()
        except ValueError as exc:
            raise SearchError(f"[{self.name}] response was not JSON") from exc
        return _results(body)


def _message_for(name: str, resp: httpx.Response) -> str:
    """A failure the model can act on, without quoting the response body.

    The body is the provider's, not the searcher's, and echoing it into a tool
    result would put unvetted text on the trusted side of the boundary that the
    rest of this package exists to maintain. The status code is enough to
    choose between rephrasing, waiting, and giving up.
    """
    if resp.status_code in (401, 403):
        return (
            f"[{name}] the search API key was rejected. Searching will not work until it is fixed."
        )
    if resp.status_code == 429:
        return f"[{name}] rate limited. Try again shortly, or answer without searching."
    if resp.status_code == 422:
        return f"[{name}] the query was rejected as malformed. Try rephrasing it."
    return f"[{name}] search failed with HTTP {resp.status_code}."


def _results(body: dict[str, Any]) -> list[SearchResult]:
    """Read the `web.results` list, skipping anything without a url.

    Tolerant on purpose. Brave returns several result families — news, videos,
    discussions — under keys that come and go, and a `KeyError` from an unseen
    shape would fail a turn that had perfectly good web results in hand.
    """
    web = body.get("web")
    raw = web.get("results", []) if isinstance(web, dict) else []
    results = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            SearchResult(
                title=str(item.get("title") or "").strip(),
                url=url,
                snippet=str(item.get("description") or "").strip(),
                published=_age(item),
            )
        )
    return results


def _age(item: dict[str, Any]) -> str | None:
    value = item.get("page_age") or item.get("age")
    return str(value) if value else None

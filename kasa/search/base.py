"""What a search provider is, in terms that name no vendor.

Search is deliberately its own small protocol rather than another `ProviderKind`
on the LLM side. The two have nothing in common but HTTP: no roles, no token
accounting, no streaming, no fallback chain. Folding them together would put a
`search` branch inside every method of a class that exists to talk to models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One hit, reduced to the fields every provider actually agrees on.

    Every string here was written by a stranger. Nothing that renders a result
    may treat any of it as trusted, including the url.
    """

    title: str
    url: str
    snippet: str
    #: Whatever the provider says about age, verbatim and unparsed. Providers
    #: disagree about the format and about whether it is a publication date or
    #: a crawl date, and a normalized-looking value would imply a precision
    #: that is not there.
    published: str | None = None


class SearchProvider(Protocol):
    """A web search backend."""

    name: str

    async def search(self, query: str, *, count: int) -> list[SearchResult]: ...

    async def aclose(self) -> None: ...

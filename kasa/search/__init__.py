"""Outbound web search.

The first capability that reaches outside the machine for something other than
a model, and the first that brings back text nobody in the conversation wrote.
`tool.py` is where that second fact is handled.
"""

from kasa.search.base import SearchProvider, SearchResult
from kasa.search.brave import BraveSearch
from kasa.search.tool import web_search_tool

__all__ = ["BraveSearch", "SearchProvider", "SearchResult", "web_search_tool"]

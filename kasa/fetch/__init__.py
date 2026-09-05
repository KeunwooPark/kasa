"""Reading a web page, and the boundary that makes it safe to.

`guard` decides where a request may go, `client` sends it and bounds what comes
back, `readable` turns a page into words, and `tool` hands those words to the
model behind `kasa/untrusted.py`'s delimiter. `browser` is the other way of
getting the HTML, for pages that draw themselves — same guard, same extraction,
same boundary, and a much larger bill. Nothing here trusts anything it reads,
including the URL it was given.
"""

from kasa.fetch.browser import BrowserRenderer, Rendered
from kasa.fetch.client import MAX_BYTES, MAX_CHARS, MAX_REDIRECTS, Page, WebFetcher
from kasa.fetch.guard import Target, approve
from kasa.fetch.tool import web_fetch_tool

__all__ = [
    "MAX_BYTES",
    "MAX_CHARS",
    "MAX_REDIRECTS",
    "BrowserRenderer",
    "Page",
    "Rendered",
    "Target",
    "WebFetcher",
    "approve",
    "web_fetch_tool",
]

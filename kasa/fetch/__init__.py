"""Reading a web page, and the boundary that makes it safe to.

`guard` decides where a request may go, `client` sends it and bounds what comes
back, `readable` turns a page into words, and `tool` hands those words to the
model behind `kasa/untrusted.py`'s delimiter. Nothing here trusts anything it
reads, including the URL it was given.
"""

from kasa.fetch.client import MAX_BYTES, MAX_CHARS, MAX_REDIRECTS, Page, WebFetcher
from kasa.fetch.guard import Target, approve
from kasa.fetch.tool import web_fetch_tool

__all__ = [
    "MAX_BYTES",
    "MAX_CHARS",
    "MAX_REDIRECTS",
    "Page",
    "Target",
    "WebFetcher",
    "approve",
    "web_fetch_tool",
]

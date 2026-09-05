"""HTML to the words a reader would have seen.

Not a renderer and not trying to be. What a turn needs from a page is the text
in reading order, small enough to sit in a context window beside everything
else — so this drops what a reader never sees (scripts, styles, metadata),
keeps the block structure that tells a heading from a paragraph, and stops at a
character budget rather than letting one long page evict the conversation.

`html.parser` rather than a parser dependency, deliberately. The input is
hostile by assumption, and the standard library's parser has one job here:
turn tags into events. Nothing is executed, no resource is loaded, and no
element's content is trusted — the output goes behind `kasa/untrusted.py`
either way.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

#: Elements whose text is markup, styling, or metadata rather than reading
#: matter. Everything between the open and close tag is dropped.
#:
#: `head` is not among them, because `<title>` is in it and the title is often
#: the most useful line on the page. Nothing else in a head carries text: the
#: two that do, `script` and `style`, are silenced by name.
SILENT = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "math",
        "iframe",
        "object",
        "canvas",
        "form",
    }
)

#: Elements that end a line. A page whose every block ran together would cost
#: the same tokens and read as one paragraph.
BLOCK = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl",
        "dt", "fieldset", "figcaption", "figure", "footer", "h1", "h2", "h3",
        "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p",
        "pre", "section", "table", "tbody", "td", "tfoot", "th", "thead",
        "tr", "ul",
    }
)  # fmt: skip

#: Elements that end a paragraph rather than a line, so a wall of `<div>`s does
#: not become a wall of blank lines but a heading still stands apart.
PARAGRAPH = frozenset({"p", "article", "section", "h1", "h2", "h3", "h4", "h5", "h6"})

_SPACES = re.compile(r"[^\S\n]+")
_BLANKS = re.compile(r"\n{3,}")

#: Appended when the budget ran out, so the model can tell a short page from a
#: long one it has only the beginning of.
CUT = "\n[…the rest of the page was not read]"


class _Reader(HTMLParser):
    def __init__(self) -> None:
        # `convert_charrefs` does the entity decoding, which is the one part of
        # this that would otherwise be a table nobody maintains.
        super().__init__(convert_charrefs=True)
        self.title: str | None = None
        self._parts: list[str] = []
        self._silent = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SILENT:
            self._silent += 1
            return
        if self._silent:
            return
        if tag == "title":
            self._in_title = True
        else:
            self._break(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Void elements never close, so `handle_starttag` must not be allowed
        # to open a silent region one of them named.
        if tag not in SILENT and not self._silent:
            self._break(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in SILENT:
            self._silent = max(0, self._silent - 1)
            return
        if self._silent:
            return
        if tag == "title":
            self._in_title = False
        else:
            self._break(tag)

    def _break(self, tag: str) -> None:
        """End the line, or the paragraph, without stacking the two.

        A list item both closes and opens a block between `a` and `b`, and
        appending a newline for each would put a blank line between every pair
        of items. Adjacent breaks merge into the strongest one asked for.
        """
        size = 2 if tag in PARAGRAPH else 1 if tag in BLOCK else 0
        if not size:
            return
        while self._parts and self._parts[-1] in ("\n", "\n\n"):
            size = max(size, len(self._parts.pop()))
        self._parts.append("\n" * size)

    def handle_data(self, data: str) -> None:
        if self._silent:
            return
        if self._in_title:
            # First one wins: a page with two `<title>`s is telling a browser
            # about the first, and the rest is somebody's mistake or bait.
            if self.title is None and data.strip():
                self.title = _SPACES.sub(" ", data).strip()
            return
        self._parts.append(data)

    def text(self) -> str:
        return _tidy("".join(self._parts))


def readable(html: str, *, limit: int) -> tuple[str | None, str, bool]:
    """`(title, text, truncated)` for a page of HTML.

    Never raises on bad markup: `html.parser` is permissive by design, and a
    page that fails to parse cleanly is still a page whose visible words are
    worth having. The budget is applied last, to the text, because it is the
    text that costs tokens — a megabyte of minified `<div>` attributes is free
    by the time it gets here.
    """
    reader = _Reader()
    reader.feed(html)
    reader.close()
    return (reader.title, *clamp(reader.text(), limit=limit))


def clamp(text: str, *, limit: int) -> tuple[str, bool]:
    """Cut `text` to `limit` characters, on a line boundary where there is one."""
    tidied = _tidy(text)
    if len(tidied) <= limit:
        return tidied, False
    cut = tidied[:limit]
    # Back up to the last line break in the final tenth, so the cut lands
    # between two things rather than inside a sentence.
    if (edge := cut.rfind("\n", int(limit * 0.9))) > 0:
        cut = cut[:edge]
    return cut.rstrip() + CUT, True


def _tidy(text: str) -> str:
    """Collapse the whitespace HTML is written with, keeping the line breaks
    the block elements earned."""
    # `\xa0` is a space to a reader and a word character to `\s`, so a page
    # laid out with them would come back as one unbreakable run.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()

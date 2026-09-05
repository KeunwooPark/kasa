"""The half that needs a real browser.

Marked `browser` and deselected by default, because CI does not install 650MB
of Chromium to prove that Chromium works. What it does prove is the two claims
`kasa/fetch/browser.py` rests on and cannot check against a fake:

- `--host-resolver-rules` is obeyed, so the pin is real rather than decorative;
- a certificate is still verified against the *name* under that pin, so the pin
  costs no TLS.

Run with `uv run pytest -m browser` on a machine where
`uv run playwright install chromium` has been run.
"""

from __future__ import annotations

import socket

import pytest

from kasa.errors import FetchError
from kasa.fetch.browser import BrowserRenderer

pytestmark = [pytest.mark.browser, pytest.mark.external]


async def address_of(host: str) -> str:
    infos = (
        await __import__("asyncio")
        .get_running_loop()
        .getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    )
    return sorted({i[4][0] for i in infos if ":" not in i[4][0]})[0]


async def test_a_real_page_renders() -> None:
    renderer = BrowserRenderer(timeout=45.0, settle_ms=1_000)
    try:
        page = await renderer.render("https://example.com/")
    finally:
        await renderer.aclose()

    assert "Example Domain" in page.html
    assert page.requests >= 1


async def test_the_pin_is_obeyed_rather_than_ignored() -> None:
    """The claim the whole design rests on. A name pinned at an address that
    answers nothing must fail to connect — if Chromium quietly resolved the
    name itself, this would load."""

    async def to_a_black_hole(host: str, port: int) -> list[str]:
        return ["192.0.2.1"]  # TEST-NET-1, routable nowhere

    renderer = BrowserRenderer(timeout=12.0, settle_ms=0, resolver=to_a_black_hole)
    try:
        with pytest.raises(FetchError):
            await renderer.render("https://example.com/")
    finally:
        await renderer.aclose()


async def test_the_certificate_is_still_checked_against_the_name() -> None:
    """Pinning must not become a way to accept anybody's certificate. Pointing
    one name at an unrelated host has to fail, and fail on the certificate."""
    elsewhere = await address_of("github.com")

    async def to_the_wrong_host(host: str, port: int) -> list[str]:
        return [elsewhere]

    renderer = BrowserRenderer(timeout=20.0, settle_ms=0, resolver=to_the_wrong_host)
    try:
        with pytest.raises(FetchError):
            await renderer.render("https://example.com/")
    finally:
        await renderer.aclose()


async def test_a_page_that_draws_itself_comes_back_with_what_it_drew() -> None:
    """#195's actual errand: a timetable that is not in the served HTML."""
    import re

    renderer = BrowserRenderer(timeout=60.0, settle_ms=4_000)
    try:
        page = await renderer.render("https://www.megabox.co.kr/theater/time?brchNo=3392")
    finally:
        await renderer.aclose()

    assert re.search(r"\b\d{1,2}:\d{2}\b", page.html), "showtimes reached the HTML"

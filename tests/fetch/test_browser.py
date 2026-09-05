"""Rendering: which requests are made, and which are refused.

No browser. What is worth asserting about this module is its policy — the
resource types it never fetches, the addresses it aborts, the caps that stop a
page spending a turn — and none of that needs 650MB of Chromium. `Launcher` is
a protocol for exactly this reason.

`test_browser_live.py` is the other half, marked `browser`, and runs a real one.
"""

from __future__ import annotations

from typing import Any

import pytest

from kasa.errors import Blocked, FetchError
from kasa.fetch.browser import HARDENING, SKIPPED_TYPES, BrowserRenderer

PUBLIC = "93.184.216.34"
PAGE = "<html><head><title>Deploys</title></head><body><p>Tuesday.</p></body></html>"


def resolving(*addresses: str) -> Any:
    async def resolve(host: str, port: int) -> list[str]:
        return list(addresses or (PUBLIC,))

    return resolve


class FakeRoute:
    """One intercepted request, and what the handler decided about it."""

    def __init__(self, url: str, resource_type: str = "document") -> None:
        self.request = type("Req", (), {"url": url, "resource_type": resource_type})()
        self.outcome: str | None = None

    async def abort(self) -> None:
        self.outcome = "abort"

    async def continue_(self) -> None:
        self.outcome = "continue"


class FakePage:
    def __init__(self, browser: FakeBrowser) -> None:
        self._browser = browser
        self.url = browser.final_url
        self.handler: Any = None

    async def route(self, pattern: str, handler: Any) -> None:
        self.handler = handler

    async def goto(self, url: str, **kwargs: Any) -> None:
        if self._browser.goto_raises:
            raise self._browser.goto_raises
        self._browser.navigated.append((url, kwargs.get("wait_until")))
        # A real page fetches the document and then its subresources, all of
        # them through `page.route`, so the tally is exercised where it
        # actually is: during the render rather than after it.
        for sub, kind in [(url, "document"), *self._browser.subrequests]:
            route = FakeRoute(sub, kind)
            await self.handler(route)
            self._browser.outcomes.append(route.outcome)

    async def wait_for_timeout(self, ms: float) -> None:
        self._browser.settled = ms

    async def content(self) -> str:
        return self._browser.html


class FakeContext:
    def __init__(self, browser: FakeBrowser) -> None:
        self._browser = browser

    async def new_page(self) -> FakePage:
        page = FakePage(self._browser)
        self._browser.pages.append(page)
        return page


class FakeBrowser:
    def __init__(self, html: str = PAGE, final_url: str = "https://example.invalid/a") -> None:
        self.html = html
        self.final_url = final_url
        self.pages: list[FakePage] = []
        self.contexts: list[dict[str, Any]] = []
        self.navigated: list[tuple[str, Any]] = []
        self.settled: float | None = None
        self.closed = False
        self.goto_raises: Exception | None = None
        #: What the page asks for while it loads, and what it was told.
        self.subrequests: list[tuple[str, str]] = []
        self.outcomes: list[str | None] = []

    async def new_context(self, **kwargs: Any) -> FakeContext:
        self.contexts.append(kwargs)
        return FakeContext(self)

    async def close(self) -> None:
        self.closed = True


def renderer(browser: FakeBrowser, **kwargs: Any) -> tuple[BrowserRenderer, list[list[str]]]:
    """A renderer wired to `browser`, plus the args it launched with."""
    launches: list[list[str]] = []

    async def launcher(*, args: list[str], timeout_ms: float) -> Any:
        launches.append(args)
        return browser

    return (
        BrowserRenderer(launcher=launcher, resolver=resolving(), **kwargs),
        launches,
    )


# -- the ordinary case --------------------------------------------------------


async def test_a_rendered_page_comes_back_as_its_html() -> None:
    browser = FakeBrowser()
    r, _ = renderer(browser)

    page = await r.render("https://example.invalid/a")

    assert page.html == PAGE
    assert page.url == "https://example.invalid/a"
    assert browser.closed, "the browser does not outlive the render"


async def test_the_document_is_pinned_to_the_approved_address() -> None:
    """The same property a static fetch gets by connecting to the address it
    resolved. Chromium obeys the rule and still checks the certificate against
    the name, so the pin costs no TLS."""
    r, launches = renderer(FakeBrowser())

    await r.render("https://example.invalid/a")

    assert f"--host-resolver-rules=MAP example.invalid {PUBLIC}" in launches[0]


async def test_background_networking_is_turned_off() -> None:
    """Traffic `page.route` never sees is traffic worth not generating."""
    r, launches = renderer(FakeBrowser())

    await r.render("https://example.invalid/a")

    assert set(HARDENING) <= set(launches[0])


async def test_the_context_refuses_downloads_and_bad_certificates() -> None:
    browser = FakeBrowser()
    r, _ = renderer(browser)

    await r.render("https://example.invalid/a")

    assert browser.contexts[0]["accept_downloads"] is False
    assert browser.contexts[0]["ignore_https_errors"] is False


async def test_the_page_is_given_a_fixed_settle_not_network_idle() -> None:
    """Waiting for the network to fall idle is what ordinary automation does,
    and it does not survive a page that polls — the one this was measured
    against never went idle at all."""
    browser = FakeBrowser()
    r, _ = renderer(browser, settle_ms=1234)

    await r.render("https://example.invalid/a")

    assert browser.navigated == [("https://example.invalid/a", "domcontentloaded")]
    assert browser.settled == 1234


async def test_a_url_the_guard_refuses_never_launches_a_browser() -> None:
    launched = []

    async def launcher(*, args: list[str], timeout_ms: float) -> Any:
        launched.append(args)
        raise AssertionError("a blocked URL launched a browser")

    r = BrowserRenderer(launcher=launcher, resolver=resolving("127.0.0.1"))

    with pytest.raises(Blocked, match="loopback"):
        await r.render("https://example.invalid/a")
    assert launched == []


async def test_a_page_that_will_not_load_says_so_without_the_stack() -> None:
    browser = FakeBrowser()
    browser.goto_raises = RuntimeError("net::ERR_SOMETHING at https://93.184.216.34/a")
    r, _ = renderer(browser)

    with pytest.raises(FetchError, match="did not load") as caught:
        await r.render("https://example.invalid/a")

    assert PUBLIC not in str(caught.value), "the pin is this machine's business"
    assert browser.closed


# -- what the page is allowed to fetch ----------------------------------------


async def routed(browser: FakeBrowser, *requests: tuple[str, str], **kwargs: Any) -> list[str]:
    """Render a page that asks for `requests`, and report what each was told.

    The document itself is routed first, as it is in a real render, so the
    outcomes here are the subresources' — which is what the tests are about.
    """
    browser.subrequests = list(requests)
    r, _ = renderer(browser, **kwargs)
    await r.render("https://example.invalid/a")
    return browser.outcomes[1:]


@pytest.mark.parametrize("kind", sorted(SKIPPED_TYPES))
async def test_what_a_reader_never_sees_is_never_fetched(kind: str) -> None:
    """Aborted on type alone, before any resolution. On the page this was
    measured against they were 5,265 of 5,341 requests."""
    assert await routed(FakeBrowser(), ("https://cdn.example.invalid/x", kind)) == ["abort"]


async def test_a_script_the_page_needs_is_allowed_through() -> None:
    assert await routed(FakeBrowser(), ("https://cdn.example.invalid/x.js", "script")) == [
        "continue"
    ]


async def test_an_xhr_to_a_private_address_is_aborted() -> None:
    """The whole reason every request is judged. A bare headless browser is a
    live SSRF — it reached a loopback server when this was measured."""
    launches: list[list[str]] = []

    async def launcher(*, args: list[str], timeout_ms: float) -> Any:
        launches.append(args)
        return browser

    browser = FakeBrowser()

    async def resolve(host: str, port: int) -> list[str]:
        return ["10.0.0.7"] if host == "internal.invalid" else [PUBLIC]

    r = BrowserRenderer(launcher=launcher, resolver=resolve)
    await r.render("https://example.invalid/a")
    handler = browser.pages[0].handler

    route = FakeRoute("https://internal.invalid/secrets", "xhr")
    await handler(route)

    assert route.outcome == "abort"


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:80/x", "http://169.254.169.254/latest/", "file:///etc/passwd"],
)
async def test_a_request_at_this_machine_is_aborted(url: str) -> None:
    assert await routed(FakeBrowser(), (url, "xhr")) == ["abort"]


async def test_a_host_is_judged_once_per_render_not_once_per_request() -> None:
    """A page making a thousand requests to one CDN would otherwise be a
    thousand resolutions of one name, and would time out before the guard
    finished."""
    asked: list[str] = []

    async def resolve(host: str, port: int) -> list[str]:
        asked.append(host)
        return [PUBLIC]

    browser = FakeBrowser()

    async def launcher(*, args: list[str], timeout_ms: float) -> Any:
        return browser

    r = BrowserRenderer(launcher=launcher, resolver=resolve)
    await r.render("https://example.invalid/a")
    handler = browser.pages[0].handler
    for n in range(50):
        await handler(FakeRoute(f"https://cdn.example.invalid/{n}.js", "script"))

    assert asked.count("cdn.example.invalid") == 1


# -- the caps -----------------------------------------------------------------


async def test_a_page_that_will_not_stop_fetching_is_cut_off() -> None:
    browser = FakeBrowser()
    requests = [(f"https://example.invalid/{n}.js", "script") for n in range(10)]

    # Four fetches in total, and the document is the first of them.
    outcomes = await routed(browser, *requests, max_requests=4)

    assert outcomes[:3] == ["continue"] * 3
    assert outcomes[3:] == ["abort"] * 7


async def test_what_is_never_fetched_does_not_consume_the_budget() -> None:
    """#197. A measured page intercepted 4,381 requests of which 4,311 were
    images; counting those against the cap stopped a render at 600 that had
    fetched about 70, and reported a complete page as cut off."""
    browser = FakeBrowser()
    requests = [(f"https://example.invalid/{n}.png", "image") for n in range(200)]
    requests.append(("https://example.invalid/late.js", "script"))

    outcomes = await routed(browser, *requests, max_requests=4)

    assert outcomes[:200] == ["abort"] * 200, "every image aborted"
    assert outcomes[200] == "continue", "and the script that came after still went"


async def test_a_render_that_only_aborted_images_is_not_incomplete() -> None:
    """The bug as the model saw it: a page that lost nothing, described as
    though it had."""
    browser = FakeBrowser()
    browser.subrequests = [(f"https://example.invalid/{n}.png", "image") for n in range(500)]
    r, _ = renderer(browser, max_requests=10)

    page = await r.render("https://example.invalid/a")

    assert not page.incomplete
    assert page.blocked == 500
    assert page.fetched == 1, "the document, and nothing else"


async def test_a_page_that_asks_pathologically_often_is_still_stopped() -> None:
    """The cap on fetches is the budget; this is the runaway guard, and it is
    absolute rather than a multiple of that budget. Free is not unlimited."""
    browser = FakeBrowser()
    browser.subrequests = [("https://example.invalid/x.png", "image")] * 300
    r, _ = renderer(browser, max_requests=2, max_intercepts=100)

    page = await r.render("https://example.invalid/a")

    assert page.incomplete
    assert page.fetched == 1, "the document, and nothing after the ceiling"
    assert page.requests == 301, "it keeps counting how often it was asked"
    assert set(browser.outcomes) == {"abort", "continue"}


async def test_a_render_stopped_by_the_fetch_cap_says_it_was_incomplete() -> None:
    """So the model can tell a page that finished from one stopped part-way,
    and weigh what it read accordingly."""
    browser = FakeBrowser()
    browser.subrequests = [(f"https://example.invalid/{n}.js", "script") for n in range(10)]
    r, _ = renderer(browser, max_requests=3)

    page = await r.render("https://example.invalid/a")

    assert page.incomplete
    assert page.fetched == 3
    assert browser.outcomes[3:] == ["abort"] * 8, "everything past the cap"


async def test_a_render_that_stayed_inside_its_caps_is_not_marked_incomplete() -> None:
    browser = FakeBrowser()
    browser.subrequests = [("https://example.invalid/a.js", "script")]
    r, _ = renderer(browser, max_requests=50)

    page = await r.render("https://example.invalid/a")

    assert not page.incomplete
    assert page.blocked == 0
    assert page.fetched == 2, "the document and its script"


async def test_the_count_of_what_was_refused_comes_back() -> None:
    """Worth reporting: a page that had two thirds of its requests aborted is
    a page whose text should be read with that in mind."""
    browser = FakeBrowser()
    browser.subrequests = [
        ("https://example.invalid/a.png", "image"),
        ("https://example.invalid/b.woff", "font"),
        ("https://example.invalid/c.js", "script"),
    ]
    r, _ = renderer(browser)

    page = await r.render("https://example.invalid/a")

    assert page.requests == 4, "the document and its three subresources"
    assert page.blocked == 2, "the image and the font, never the script"


async def test_only_one_render_runs_at_a_time() -> None:
    """A browser is the most expensive thing this daemon can be asked for, and
    two at once is a gigabyte of RSS for two answers nobody is reading yet."""
    import asyncio

    live = 0
    peak = 0

    class Slow(FakeBrowser):
        async def new_context(self, **kwargs: Any) -> FakeContext:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.02)
            live -= 1
            return await super().new_context(**kwargs)

    browser = Slow()

    async def launcher(*, args: list[str], timeout_ms: float) -> Any:
        return browser

    r = BrowserRenderer(launcher=launcher, resolver=resolving())
    await asyncio.gather(*(r.render("https://example.invalid/a") for _ in range(4)))

    assert peak == 1

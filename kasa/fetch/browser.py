"""Rendering a page that draws itself in the browser.

`web_fetch` reads the HTML a site serves. A growing share of the web does not
put its content there — a cinema timetable arrives over XHR after the document
has loaded, and the document says nothing about what is playing. This module is
the other way of getting it: run the page, then read what it drew.

It is a much larger capability than a GET, and the difference is not the
rendering. It is that a browser makes hundreds of requests nobody chose. A
static fetch has one URL per hop to judge; one measured render of a cinema page
made 147, and 5,341 with interception on. So the design is about that number:

- **Every request is judged**, by the same `guard.approve` a static fetch uses,
  and anything it refuses is aborted before it is sent. A bare headless browser
  is a live SSRF — measured, not assumed: it reached a loopback server.
- **Most requests are never made.** Images, media and fonts are aborted on
  resource type alone, before any resolution. They are not reading matter, and
  on that same page they were 5,265 of the 5,341.
- **The document's address is pinned**, with `--host-resolver-rules`, to the one
  the guard approved — the same property `kasa/fetch/client.py` gets by
  connecting to the address it resolved. Chromium obeys the rule and still
  verifies the certificate against the *name*, so the pin costs no TLS.
- **Nothing is operated.** Navigate, wait, read. No clicking, typing, form
  submission, or downloads: a rendered page is full of controls, and the text
  next to them was written by a stranger who would like them pressed.

The context is thrown away after every render, so nothing — cookie, storage,
service worker — survives from one page to the next.

`playwright` is an optional extra and is imported inside the launch, so an
install without it is an install where `[browser] enabled` simply cannot be
turned on, rather than one that fails at import time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from kasa.errors import FetchError
from kasa.fetch.guard import Resolver, approve

log = logging.getLogger(__name__)

#: Resource types a reader never sees the point of. Aborted on type alone,
#: which is cheaper than judging them and is most of the traffic.
SKIPPED_TYPES = frozenset({"image", "media", "font"})

#: The whole render, including navigation and the settle below it.
DEFAULT_TIMEOUT = 30.0

#: How long to let script keep working after the document is ready. Waiting for
#: the network to fall idle is what an ordinary automation does and it does not
#: survive contact with a page that polls: the measured cinema page never went
#: idle at all. A fixed settle does.
DEFAULT_SETTLE_MS = 3_000

#: A page that wants more than this is not a page. Both are bounds on what one
#: turn can cost, not on what a site may legitimately do.
DEFAULT_MAX_REQUESTS = 600
DEFAULT_MAX_BYTES = 20_000_000

#: Turned off because none of it is this page: update pings, sync, safe-browsing
#: lists, network prediction. Traffic `page.route` never sees is traffic worth
#: not generating.
HARDENING = (
    "--disable-background-networking",
    "--disable-sync",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--no-default-browser-check",
    "--no-first-run",
    "--disable-features=NetworkPrediction,OptimizationHints,MediaRouter",
    "--disable-dev-shm-usage",
)


@dataclass(frozen=True, slots=True)
class Rendered:
    """The HTML a page produced, and what it cost to get it."""

    url: str
    html: str
    requests: int
    blocked: int
    truncated: bool
    """Whether a cap stopped the render before the page was finished."""


class Launcher(Protocol):
    """`playwright.chromium.launch`, narrowed to what this module asks of it.

    A protocol so the policy above is testable without 650MB of browser: what
    is worth asserting is which requests are aborted and which caps fire, and
    none of that needs a real renderer.
    """

    async def __call__(self, *, args: list[str], timeout_ms: float) -> Any: ...


class BrowserRenderer:
    """One browser, launched on demand and closed when it goes quiet."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        settle_ms: int = DEFAULT_SETTLE_MS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_bytes: int = DEFAULT_MAX_BYTES,
        resolver: Resolver | None = None,
        launcher: Launcher | None = None,
    ) -> None:
        self._timeout = timeout
        self._settle_ms = settle_ms
        self._max_requests = max_requests
        self._max_bytes = max_bytes
        self._resolver = resolver
        self._launcher = launcher
        self._playwright: Any = None
        self._lock = asyncio.Lock()

    async def warm(self) -> None:
        """Prove a browser can actually start, now rather than mid-turn.

        `import playwright` succeeding says the wheel is installed. It says
        nothing about `playwright install chromium` ever having been run, and
        the gap between the two is a render that fails on the first question
        somebody asks it.
        """
        browser = await self._launch("localhost", "127.0.0.1")
        await _quietly(browser.close())

    async def aclose(self) -> None:
        if self._playwright is not None:
            playwright, self._playwright = self._playwright, None
            await playwright.stop()

    async def render(self, url: str) -> Rendered:
        """Run `url` and hand back what it drew, or raise saying why not."""
        target = await approve(url, resolve=self._resolver)
        # One render at a time. A browser is the most expensive thing this
        # daemon can be asked for, and two turns rendering at once is a
        # gigabyte of RSS for two answers nobody is reading yet.
        async with self._lock:
            started = time.monotonic()
            browser = await self._launch(target.host, target.address)
            try:
                return await asyncio.wait_for(
                    self._run(browser, target.url),
                    timeout=max(self._timeout - (time.monotonic() - started), 1.0),
                )
            except TimeoutError as exc:
                raise FetchError(
                    f"{target.host} did not finish rendering in {self._timeout:.0f}s."
                ) from exc
            finally:
                await _quietly(browser.close())

    # -- internals -----------------------------------------------------------

    async def _launch(self, host: str, address: str) -> Any:
        args = [
            # The pin. Chromium obeys it — a name mapped to a black hole fails
            # to connect — and still verifies the certificate against the name,
            # so the document is fetched from the address the guard approved
            # without weakening TLS to do it.
            f"--host-resolver-rules=MAP {host} {address}",
            *HARDENING,
        ]
        launcher = self._launcher or await self._chromium()
        try:
            return await launcher(args=args, timeout_ms=self._timeout * 1000)
        except Exception as exc:
            raise FetchError(f"the browser would not start ({type(exc).__name__}).") from exc

    async def _chromium(self) -> Launcher:
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise FetchError(
                "rendering needs the browser extra: install it with "
                "`uv sync --extra browser && uv run playwright install chromium`."
            ) from exc
        if self._playwright is None:
            self._playwright = await async_playwright().start()

        async def launch(*, args: list[str], timeout_ms: float) -> Any:
            return await self._playwright.chromium.launch(
                headless=True, args=args, timeout=timeout_ms
            )

        return launch

    async def _run(self, browser: Any, url: str) -> Rendered:
        # Everything the context is given is a refusal to remember anything:
        # no storage state, and it is discarded with the browser either way.
        context = await browser.new_context(
            ignore_https_errors=False,
            java_script_enabled=True,
            accept_downloads=False,
        )
        page = await context.new_page()
        tally = _Tally(self._max_requests, self._max_bytes)
        await page.route("**/*", _guarding(tally, self._resolver))

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=self._timeout * 1000)
        except Exception as exc:
            raise FetchError(f"the page did not load ({type(exc).__name__}).") from exc
        # Fixed rather than `networkidle`: the page this was measured against
        # polls and never goes idle, and a render that waits for that waits
        # until the turn's timeout.
        await page.wait_for_timeout(self._settle_ms)

        html = str(await page.content())
        final = str(page.url) or url
        return Rendered(
            url=final,
            html=html,
            requests=tally.seen,
            blocked=tally.blocked,
            truncated=tally.capped,
        )


class _Tally:
    """What a render has spent, and whether it may spend more."""

    def __init__(self, max_requests: int, max_bytes: int) -> None:
        self.seen = 0
        self.blocked = 0
        self.capped = False
        self._max_requests = max_requests
        self._max_bytes = max_bytes

    def allow(self) -> bool:
        self.seen += 1
        if self.seen > self._max_requests:
            self.capped = True
            return False
        return True

    def refuse(self) -> None:
        self.blocked += 1


def _guarding(tally: _Tally, resolver: Resolver | None) -> Any:
    """The route handler: judge, or abort, every request the page makes."""
    approved: dict[str, bool] = {}

    async def handle(route: Any) -> None:
        request = route.request
        if not tally.allow():
            tally.refuse()
            await _quietly(route.abort())
            return
        if request.resource_type in SKIPPED_TYPES:
            # Not judged, because not fetched. Cheaper than resolving a name
            # for a photograph nobody will read.
            tally.refuse()
            await _quietly(route.abort())
            return

        url = str(request.url)
        host = url.split("/")[2] if "//" in url else ""
        # Cached per render, not across renders. Judging every one of a
        # thousand requests to the same CDN would be a thousand resolutions of
        # one name, and the page would time out before the guard finished.
        if host not in approved:
            try:
                await approve(url, resolve=resolver)
            except Exception as exc:
                log.debug("render refused %s: %s", host or url, exc)
                approved[host] = False
            else:
                approved[host] = True
        if not approved[host]:
            tally.refuse()
            await _quietly(route.abort())
            return
        await _quietly(route.continue_())

    return handle


async def _quietly(awaitable: Any) -> None:
    """Await something whose failure changes nothing.

    Aborting a request the page has already given up on, or closing a browser
    that has already died, both raise — and neither is a reason to fail a
    render that otherwise worked.
    """
    try:
        await awaitable
    except Exception as exc:
        log.debug("ignored while tidying up a render: %s", exc)

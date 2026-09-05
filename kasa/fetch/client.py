"""One GET, bounded in every dimension that can be made to grow.

A fetch is the only thing in Kasa that opens a connection to an address a model
chose. Everything here is a bound on what that can cost or reach:

- where — every hop re-judged by `guard.approve`, and connected to the address
  the guard approved rather than to a second resolution of the same name;
- how far — a small redirect limit, since a chain is a chain of chances to be
  sent somewhere else;
- how long — one timeout for the whole thing, not per-hop, so a chain of slow
  redirects cannot outlast the turn that is waiting for it;
- how big — the body is read in chunks against a byte cap and abandoned when it
  is reached, so `Content-Length: 12` in front of a gigabyte costs one buffer;
- what — a content-type allowlist, because an executable is not reading matter
  and a 200 that returns one is not a page.

Nothing this daemon knows travels outbound. No cookies are kept between hops,
no `Authorization` is ever set, and the model cannot name a header — a fetch
carries a `User-Agent`, an `Accept`, and nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from kasa.errors import FetchError
from kasa.fetch.browser import BrowserRenderer
from kasa.fetch.guard import Resolver, Target, approve
from kasa.fetch.readable import clamp, readable

log = logging.getLogger(__name__)

#: Chains beyond this are a redirector, a login wall, or a loop, and each hop
#: is another address to be talked out of refusing.
MAX_REDIRECTS = 4

#: Read from the wire, before any text extraction. Generous enough for a real
#: page — the front page of a news site is a megabyte of markup — and far short
#: of a video.
MAX_BYTES = 2_000_000

#: Handed to the model, after extraction. What actually costs tokens.
MAX_CHARS = 20_000

DEFAULT_TIMEOUT = 15.0

#: Sent so that a site can identify and refuse this. A crawler that will not say
#: what it is has decided its convenience outranks the operator's wishes.
USER_AGENT = "KasaBot/1.0 (+https://github.com/KeunwooPark/kasa)"

#: What can be read as text. Anything else is a download, and a download is not
#: something a turn has a use for.
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
TEXT_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/csv",
        "text/xml",
        "application/xml",
        "application/json",
        "application/ld+json",
    }
)


@dataclass(frozen=True, slots=True)
class Page:
    """What a fetch produced. Every string in it was written by a stranger."""

    url: str
    """Where it ended up, which is not always where it was sent."""
    status: int
    content_type: str
    title: str | None
    text: str
    truncated: bool
    redirects: int
    #: Whether a browser ran the page, rather than the served HTML being read.
    rendered: bool = False
    #: Set when the served HTML looks like a shell that fills itself in later —
    #: little text for its size, plenty of script. A fact about the page, not
    #: advice: what to *do* about it depends on whether this install has a
    #: browser, and only the tool knows that. It is what turns "there is
    #: nothing here" into "there is nothing here *yet*", which is the
    #: difference between the model giving up and asking for a render (#195).
    scripted: bool = False


class WebFetcher:
    """The `web_fetch` backend. One per process; holds one connection pool."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = MAX_BYTES,
        max_chars: int = MAX_CHARS,
        max_redirects: int = MAX_REDIRECTS,
        resolver: Resolver | None = None,
        client: httpx.AsyncClient | None = None,
        renderer: BrowserRenderer | None = None,
    ) -> None:
        self._timeout = timeout
        self._max_bytes = max_bytes
        self._max_chars = max_chars
        self._max_redirects = max_redirects
        self._resolver = resolver
        #: None when the browser extra is absent or `[browser]` is off, which
        #: is also when the tool stops offering `render` at all.
        self._renderer = renderer
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            # Redirects are followed here, one judged hop at a time. Letting
            # httpx follow them would mean the guard saw the first URL and the
            # socket went wherever the last `Location` said.
            follow_redirects=False,
            # A pool this small is a rate limit as much as a resource bound: a
            # model in a loop cannot turn one turn into fifty sockets.
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    @property
    def can_render(self) -> bool:
        return self._renderer is not None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
        if self._renderer is not None:
            await self._renderer.aclose()

    async def fetch(self, url: str, *, render: bool = False) -> Page:
        """Read `url`, or raise `Blocked` / `FetchError` saying why not.

        `render` runs the page in a browser instead of reading what the server
        sent. It is the expensive path — seconds and hundreds of megabytes —
        so it is asked for rather than guessed at.
        """
        if render:
            return await self._rendered(url)
        seen: list[str] = []
        current = url
        for hop in range(self._max_redirects + 1):
            target = await approve(current, resolve=self._resolver)
            if target.url in seen:
                raise FetchError("that URL redirects in a loop.")
            seen.append(target.url)

            response = await self._get(target)
            location = response.headers.get("location")
            if response.status_code in (301, 302, 303, 307, 308) and location:
                await response.aclose()
                if hop == self._max_redirects:
                    raise FetchError(f"that URL redirected more than {self._max_redirects} times.")
                # Resolved against the URL that answered, so a relative
                # `Location` cannot become a scheme-relative one by accident —
                # and the result goes back through `approve` like any other.
                current = urljoin(target.url, location)
                continue
            return await self._read(target, response, redirects=hop)
        raise FetchError("that URL redirected more than expected.")  # pragma: no cover

    # -- internals -----------------------------------------------------------

    async def _rendered(self, url: str) -> Page:
        """The same page, by way of a browser.

        Only the acquisition differs. Extraction is `readable` exactly as it is
        for served HTML, so what reaches the model — and the boundary it
        arrives behind — does not depend on how the bytes were got.
        """
        if self._renderer is None:
            raise FetchError(
                "rendering is not available on this install; reading the page as served instead "
                "is the only option. Ask again without render."
            )
        page = await self._renderer.render(url)
        title, text, cut = readable(page.html, limit=self._max_chars)
        if not text:
            raise FetchError("that page rendered with no readable text in it.")
        return Page(
            url=page.url,
            status=200,
            content_type="text/html",
            title=title,
            text=text,
            truncated=cut or page.truncated,
            redirects=0,
            rendered=True,
        )

    async def _get(self, target: Target) -> httpx.Response:
        """Send one request, to the address the guard approved.

        The URL's host is replaced by that address and the name is put back in
        `Host` and in TLS SNI, so the connection goes where the check went and
        the certificate is still checked against the name. Resolving again here
        — which is what any ordinary client does — is exactly the second lookup
        a rebinding attack is waiting for.
        """
        pinned = httpx.URL(target.url).copy_with(host=target.address, port=target.port)
        # Before the request is built, because that is where httpx attaches the
        # jar. Nothing a previous hop set may travel to the next one: a
        # `Set-Cookie` on a redirect to a different host is a way to make this
        # daemon carry a stranger's state to a stranger.
        self._client.cookies.clear()
        request = self._client.build_request(
            "GET",
            pinned,
            headers={
                "Host": target.authority,
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
                "Accept-Encoding": "gzip, deflate",
            },
            extensions={"sni_hostname": target.host} if target.scheme == "https" else {},
        )
        try:
            return await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise FetchError(f"{target.host} did not answer in {self._timeout:.0f}s.") from exc
        except httpx.TooManyRedirects as exc:  # pragma: no cover - redirects are ours
            raise FetchError("that URL redirects in a loop.") from exc
        except httpx.HTTPError as exc:
            # The class, not the message: httpx puts the URL in it, and the URL
            # at this point is the pinned one, which would say more about this
            # machine's resolver than the model needs to know.
            raise FetchError(f"could not reach {target.host} ({type(exc).__name__}).") from exc

    async def _read(self, target: Target, response: httpx.Response, *, redirects: int) -> Page:
        try:
            if response.status_code >= 400:
                raise FetchError(_status(target.host, response.status_code))
            content_type = _media_type(response.headers.get("content-type", ""))
            if content_type not in HTML_TYPES | TEXT_TYPES:
                named = content_type or "no content type"
                raise FetchError(
                    f"that URL returned {named}, which is not a page I can read as text."
                )
            body, oversize = await self._body(response)
        finally:
            await response.aclose()

        charset = response.charset_encoding or "utf-8"
        try:
            decoded = body.decode(charset, errors="replace")
        except LookupError:
            # A charset nobody has heard of. UTF-8 with replacement is still
            # more readable than refusing the page.
            decoded = body.decode("utf-8", errors="replace")

        if content_type in HTML_TYPES:
            title, text, cut = readable(decoded, limit=self._max_chars)
        else:
            title, (text, cut) = None, clamp(decoded, limit=self._max_chars)
        if not text and not _is_shell(decoded, content_type):
            raise FetchError(f"that URL returned {content_type} with no readable text in it.")
        return Page(
            url=target.url,
            status=response.status_code,
            content_type=content_type,
            title=title,
            text=text,
            truncated=cut or oversize,
            redirects=redirects,
            scripted=_is_shell(decoded, content_type),
        )

    async def _body(self, response: httpx.Response) -> tuple[bytes, bool]:
        """Read up to the byte cap, then stop reading.

        Streamed rather than `response.aread()`, so the cap is on what crosses
        the wire and not on what a `Content-Length` promised. A body that keeps
        coming is abandoned mid-download, which is the point.
        """
        chunks: list[bytes] = []
        size = 0
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            size += len(chunk)
            if size >= self._max_bytes:
                log.debug("stopped reading %s at %d bytes", response.url, size)
                return b"".join(chunks)[: self._max_bytes], True
        return b"".join(chunks), False


#: What share of a document has to survive as readable text before it counts as
#: a page rather than a shell waiting to be filled in. Measured rather than
#: guessed, against pages of both kinds:
#:
#:     megabox timetable (the shell)   2.4%   28 <script
#:     github repo page                5.1%   14
#:     wikipedia article               7.9%    5
#:     hacker news                    10.7%    1
#:     python docs                    15.4%   13
#:     example.com                    23.1%    0
#:
#: 4% sits in the gap, with the nearest content-bearing page nearly a third
#: clear of it.
SHELL_RATIO = 0.04

#: And enough script to explain where the content went. A short, script-free
#: page is just short.
SHELL_SCRIPTS = 5

#: Below this a document is too small for the ratio to mean anything.
SHELL_FLOOR = 2_000


def _is_shell(html: str, content_type: str) -> bool:
    """Whether this looks like a page whose content has not arrived yet.

    Deliberately crude, and deliberately only ever used to *offer* a render.
    Getting it wrong costs one sentence of advice the model is free to ignore;
    saying nothing is what left an agent reporting that a timetable did not
    exist when it was one XHR away.
    """
    if content_type not in HTML_TYPES or len(html) < SHELL_FLOOR:
        return False
    if html.lower().count("<script") < SHELL_SCRIPTS:
        return False
    _, text, _ = readable(html, limit=len(html))
    return len(text) < SHELL_RATIO * len(html)


def _media_type(header: str) -> str:
    return header.split(";", 1)[0].strip().lower()


def _status(host: str, status: int) -> str:
    """A failure the model can act on, without a word of the body in it.

    An error page is a page, written by whoever runs the site, and quoting one
    into a tool result would put a stranger's text on the trusted side of the
    boundary this package exists to hold.
    """
    if status in (401, 403):
        return f"{host} refused to serve that page without signing in (HTTP {status})."
    if status == 404:
        return f"{host} says that page does not exist (HTTP 404)."
    if status == 429:
        return f"{host} is rate limiting this (HTTP 429). Try again later, or read something else."
    if status >= 500:
        return f"{host} is failing on its own end (HTTP {status})."
    return f"{host} returned HTTP {status}."

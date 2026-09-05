"""Where a fetch is allowed to go, decided before a byte is sent.

`web_fetch` takes a URL from a model that read it in somebody else's web page.
That makes the address itself attacker-influenced, and the machine Kasa runs on
is a long-running daemon on somebody's network with a cloud metadata service, a
Slack token in its environment, and an SQLite file next to it. The interesting
target of an SSRF is never the internet; it is `169.254.169.254`, `localhost`,
and the 10/8 nobody firewalled from itself.

So the guard is not a blocklist of hostnames. It resolves the name and judges
the *addresses*, which is the only thing that survives a redirect chain, a
`0x7f.1`, a `[::ffff:127.0.0.1]`, and a domain whose A record is public on the
first lookup and `127.0.0.1` on the second. And because the check and the
connection would otherwise be two different resolutions, `approve` hands back
the address it approved and the fetcher connects to that — the name is never
resolved twice.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from kasa.errors import Blocked

ALLOWED_SCHEMES = frozenset({"http", "https"})

#: The two ports the web is on. Everything else — 22, 25, 6379, 11211, 8080 on
#: a colleague's laptop — is a service that was never meant to be read by a
#: model, and a public host is as capable of running one as a private host is.
#: Narrow on purpose: a site on a non-standard port is a URL somebody can paste
#: into a browser instead, which is a smaller loss than the alternative.
ALLOWED_PORTS = frozenset({80, 443})

DEFAULT_PORTS = {"http": 80, "https": 443}


class Resolver(Protocol):
    """`getaddrinfo`, narrowed to what the guard asks of it."""

    async def __call__(self, host: str, port: int) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class Target:
    """A URL that may be fetched, and the address it may be fetched from."""

    url: str
    scheme: str
    #: The hostname as written. Still what goes in `Host` and in TLS SNI, so
    #: the certificate is checked against the name and not against the pin.
    host: str
    port: int
    #: The literal address `approve` resolved and accepted. Connecting to
    #: anything else would be resolving the name a second time, which is the
    #: window a rebinding attack lives in.
    address: str

    @property
    def authority(self) -> str:
        """What the `Host` header says: the name, and the port if it is not the
        scheme's own."""
        if self.port == DEFAULT_PORTS[self.scheme]:
            return self.host
        return f"{self.host}:{self.port}"


async def approve(url: str, *, resolve: Resolver | None = None) -> Target:
    """Judge `url`, or raise `Blocked` saying which rule it broke."""
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise Blocked(f"that is not a URL I can parse: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        named = scheme or "no scheme"
        raise Blocked(f"only http and https can be fetched, and that URL has {named}.")
    if parts.username or parts.password:
        # A credential in a URL is either one the model invented or one it read
        # off a page, and neither is a credential this daemon should present.
        raise Blocked("a URL with a username or password in it will not be fetched.")

    try:
        host = parts.hostname
        port = parts.port or DEFAULT_PORTS[scheme]
    except ValueError as exc:
        raise Blocked(f"that URL has no usable port: {exc}") from exc
    if not host:
        raise Blocked("that URL has no host in it.")
    if port not in ALLOWED_PORTS:
        raise Blocked(f"only ports 80 and 443 can be fetched, and that URL asks for {port}.")

    addresses = await _addresses(host, port, resolve)
    # Every address, not the one that will be used. A name that answers with a
    # public address and a private one is answering with the private one to
    # whoever asks next, and that is the whole of a rebinding attack.
    for address in addresses:
        if reason := _forbidden(address):
            raise Blocked(f"{host} resolves to {address}, which is {reason}.")
    return Target(url=url, scheme=scheme, host=host, port=port, address=addresses[0])


async def _addresses(host: str, port: int, resolve: Resolver | None) -> list[str]:
    if literal := _literal(host):
        # An address written as an address needs no resolver, and asking one
        # about it would let a hosts file or a resolver plugin answer.
        return [literal]
    try:
        found = await (resolve or _getaddrinfo)(host, port)
    except OSError as exc:
        raise Blocked(f"{host} does not resolve: {exc}") from exc
    if not found:
        raise Blocked(f"{host} does not resolve to anything.")
    return found


def _literal(host: str) -> str | None:
    """The host as an IP, if it is written as one. `[::1]` arrives unbracketed."""
    try:
        return str(ipaddress.ip_address(host.strip("[]")))
    except ValueError:
        return None


async def _getaddrinfo(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    # Order preserved: the resolver put its preference first and so does this.
    seen: dict[str, None] = {}
    for info in infos:
        seen.setdefault(str(info[4][0]).split("%", 1)[0], None)
    return list(seen)


def _forbidden(address: str) -> str | None:
    """Why this address may not be fetched, or `None` if it may."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return "not an address at all"

    # The address inside first, where there is one. `::ffff:169.254.169.254` is
    # the metadata service wearing a hat, and the v6 predicates below say
    # nothing about where a tunnelled packet actually comes out.
    #
    # Judged as well as the outer address, never instead of it: 6to4 and Teredo
    # are themselves non-global, and unwrapping them *in place of* the outer
    # check would turn a blocked spelling into an allowed one.
    inner = _tunnelled(ip) if isinstance(ip, ipaddress.IPv6Address) else None
    if inner is not None and (reason := _forbidden(str(inner))):
        return f"{reason}, once the v4 address tunnelled inside it is unwrapped"
    # An inner address that is fine falls through to the outer checks below,
    # which is where 6to4 and Teredo are caught for being what they are.

    if ip.is_loopback:
        return "the loopback address — this machine itself"
    if ip.is_link_local:
        # 169.254.169.254 lives here, which is the address this whole module is
        # really about.
        return "link-local, where cloud metadata services live"
    if ip.is_multicast:
        return "a multicast address"
    if ip.is_unspecified:
        return "the unspecified address"
    if ip.is_private:
        return "a private address, inside this machine's own network"
    if ip.is_reserved:
        return "a reserved address"
    if not ip.is_global:
        return "not a globally routable address"
    return None


def _tunnelled(ip: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The v4 address a v6 address is carrying, for the three ways it can.

    Mapped, 6to4, and Teredo all reach a v4 destination through a v6 spelling,
    and the v6 predicates say nothing useful about where that destination is.
    """
    if ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    if ip.sixtofour is not None:
        return ip.sixtofour
    teredo = ip.teredo
    return teredo[1] if teredo else None

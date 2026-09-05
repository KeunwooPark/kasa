"""Where a fetch may go.

Every test here is an address somebody would like Kasa to connect to. The guard
is the only thing between a URL a model read off a web page and a socket on the
network this daemon runs on, so the cases are written as the attack rather than
as the branch: `localhost`, the metadata service, the same thing spelled in
IPv6, and a name that answers with one of each.
"""

from __future__ import annotations

import pytest

from kasa.errors import Blocked
from kasa.fetch.guard import approve


def answering(*addresses: str) -> object:
    async def resolve(host: str, port: int) -> list[str]:
        return list(addresses)

    return resolve


async def approved(url: str, *addresses: str) -> object:
    return await approve(url, resolve=answering(*addresses or ("93.184.216.34",)))


# -- what may be asked for ----------------------------------------------------


async def test_a_public_page_is_approved() -> None:
    target = await approved("https://example.invalid/a?b=c")

    assert target.host == "example.invalid"
    assert target.port == 443
    assert target.address == "93.184.216.34"
    assert target.authority == "example.invalid", "no port when it is the scheme's own"


async def test_a_non_default_port_stays_in_the_host_header() -> None:
    target = await approved("http://example.invalid:80/a")

    assert target.port == 80
    assert target.authority == "example.invalid"


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.invalid/x",
        "gopher://example.invalid/x",
        "data:text/html,hello",
        "//example.invalid/x",
        "example.invalid/x",
    ],
)
async def test_only_http_and_https_are_fetchable(url: str) -> None:
    with pytest.raises(Blocked, match="http and https"):
        await approved(url)


async def test_a_url_with_a_credential_in_it_is_refused() -> None:
    """Either the model invented it or it read it off a page, and this daemon
    presents neither."""
    with pytest.raises(Blocked, match="username or password"):
        await approved("https://user:secret@example.invalid/")


@pytest.mark.parametrize("port", [22, 25, 3306, 6379, 8080, 8443, 11211])
async def test_only_the_web_s_two_ports_are_fetchable(port: int) -> None:
    with pytest.raises(Blocked, match="ports 80 and 443"):
        await approved(f"https://example.invalid:{port}/")


async def test_a_url_with_no_host_is_refused() -> None:
    with pytest.raises(Blocked, match="no host"):
        await approved("http:///just-a-path")


# -- where it may go ----------------------------------------------------------


@pytest.mark.parametrize(
    ("address", "why"),
    [
        ("127.0.0.1", "loopback"),
        ("127.5.5.5", "loopback"),
        ("0.0.0.0", "unspecified"),
        ("10.0.0.7", "private"),
        ("172.16.4.4", "private"),
        ("192.168.1.1", "private"),
        ("169.254.169.254", "link-local"),
        ("100.64.0.1", "globally routable"),
        ("224.0.0.1", "multicast"),
        ("::1", "loopback"),
        ("fe80::1", "link-local"),
        ("fc00::1", "private"),
    ],
)
async def test_an_address_off_the_public_internet_is_refused(address: str, why: str) -> None:
    with pytest.raises(Blocked, match=why):
        await approved("https://example.invalid/", address)


async def test_the_metadata_service_is_refused_by_address_not_by_name() -> None:
    """169.254.169.254 is the address this module is really about, and a
    hostname blocklist would miss every name that points at it."""
    with pytest.raises(Blocked, match="cloud metadata"):
        await approved("https://totally-normal.invalid/", "169.254.169.254")


@pytest.mark.parametrize(
    "spelling",
    [
        "::ffff:127.0.0.1",  # v4-mapped
        "::ffff:169.254.169.254",
        "2002:7f00:1::",  # 6to4 carrying 127.0.0.1
        "2001:0:0:0:0:0:5601:5601",  # teredo carrying the metadata service
    ],
)
async def test_a_v4_address_wearing_a_v6_hat_is_still_that_address(spelling: str) -> None:
    with pytest.raises(Blocked):
        await approved("https://example.invalid/", spelling)


async def test_an_address_literal_is_judged_without_asking_a_resolver() -> None:
    """A resolver could otherwise be persuaded — by a hosts file, by a plugin —
    that 127.0.0.1 is something else."""

    async def never(host: str, port: int) -> list[str]:
        raise AssertionError("the resolver was asked about an address")

    with pytest.raises(Blocked, match="loopback"):
        await approve("http://127.0.0.1/", resolve=never)


async def test_every_answer_must_be_public_not_merely_the_first() -> None:
    """The rebinding shape. A name that answers with a public address and a
    private one is answering with the private one to whoever asks next, and the
    fetcher would be the one asking."""
    with pytest.raises(Blocked, match=r"10\.0\.0\.7"):
        await approved("https://example.invalid/", "93.184.216.34", "10.0.0.7")


async def test_the_approved_address_is_the_one_that_comes_back() -> None:
    """So the fetcher connects to what was judged. Resolving the name a second
    time at connect is the window the whole check exists to close."""
    target = await approved("https://example.invalid/", "93.184.216.34", "93.184.216.35")

    assert target.address == "93.184.216.34"


async def test_a_name_that_does_not_resolve_says_so() -> None:
    async def nothing(host: str, port: int) -> list[str]:
        return []

    with pytest.raises(Blocked, match="does not resolve"):
        await approve("https://example.invalid/", resolve=nothing)


async def test_a_resolver_failure_is_a_block_not_a_crash() -> None:
    async def broken(host: str, port: int) -> list[str]:
        raise OSError("no such host")

    with pytest.raises(Blocked, match="does not resolve"):
        await approve("https://example.invalid/", resolve=broken)

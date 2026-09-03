from __future__ import annotations

import httpx
import pytest

from kasa.errors import GitHubError
from kasa.github import GitHubClient
from tests.conftest import mock_client

REPO = {
    "full_name": "someone/kasa-memory",
    "private": True,
    "default_branch": "main",
    "clone_url": "https://github.com/someone/kasa-memory.git",
    "ssh_url": "git@github.com:someone/kasa-memory.git",
    "html_url": "https://github.com/someone/kasa-memory",
    "permissions": {"push": True, "admin": False},
    "size": 0,
}


def client(handler) -> GitHubClient:  # type: ignore[no-untyped-def]
    return GitHubClient("tok", client=mock_client(handler, base_url="https://api.github.test"))


async def test_get_repo_parses_what_matters() -> None:
    async with client(lambda r: httpx.Response(200, json=REPO)) as gh:
        info = await gh.get_repo("someone/kasa-memory")

    assert info is not None
    assert info.private and info.can_push and info.empty
    assert info.default_branch == "main"


async def test_a_missing_repo_is_none_not_an_error() -> None:
    async with client(lambda r: httpx.Response(404, json={"message": "Not Found"})) as gh:
        assert await gh.get_repo("someone/nope") is None


async def test_read_only_access_is_visible() -> None:
    payload = {**REPO, "permissions": {"push": False, "admin": False}}
    async with client(lambda r: httpx.Response(200, json=payload)) as gh:
        info = await gh.get_repo("someone/kasa-memory")
    assert info is not None and not info.can_push


async def test_absent_permissions_are_not_assumed_to_be_write() -> None:
    payload = {k: v for k, v in REPO.items() if k != "permissions"}
    async with client(lambda r: httpx.Response(200, json=payload)) as gh:
        info = await gh.get_repo("someone/kasa-memory")
    assert info is not None and not info.can_push


async def test_a_repo_under_your_own_login_is_a_user_repo() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "someone"})
        return httpx.Response(201, json=REPO)

    async with client(handler) as gh:
        await gh.create_repo("someone/kasa-memory")
    assert "/user/repos" in seen


async def test_a_repo_under_another_namespace_is_an_org_repo() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "someone"})
        return httpx.Response(201, json={**REPO, "full_name": "acme/kasa-memory"})

    async with client(handler) as gh:
        await gh.create_repo("acme/kasa-memory")
    assert "/orgs/acme/repos" in seen


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (401, "invalid or expired"),
        (403, "contents: write"),
        (422, "already exist"),
        (500, "GitHub 500"),
    ],
)
async def test_errors_explain_what_to_do(status: int, expected: str) -> None:
    handler = lambda r: httpx.Response(status, json={"message": "boom"})  # noqa: E731
    async with client(handler) as gh:
        with pytest.raises(GitHubError, match=expected):
            await gh.login()


async def test_a_network_failure_is_a_kasa_error_not_an_httpx_one() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async with client(handler) as gh:
        with pytest.raises(GitHubError, match="could not reach GitHub"):
            await gh.login()


async def test_a_non_json_body_is_reported_rather_than_crashing() -> None:
    handler = lambda r: httpx.Response(200, text="<html>maintenance</html>")  # noqa: E731
    async with client(handler) as gh:
        with pytest.raises(GitHubError, match="non-JSON"):
            await gh.login()


async def test_the_token_is_sent_as_a_bearer_header() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"login": "someone"})

    async with client(handler) as gh:
        await gh.login()
    assert seen["authorization"] == "Bearer tok"
    assert seen["x-github-api-version"]

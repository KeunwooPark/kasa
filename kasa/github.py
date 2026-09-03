"""The slice of the GitHub REST API that Kasa needs.

Only three questions are ever asked of it: who am I, does this repo exist and is
it private, and please create it. A dedicated SDK for that would be a dependency
carrying a hundred endpoints to serve three.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import httpx

from kasa.errors import GitHubError

DEFAULT_API_URL = "https://api.github.com"
_ACCEPT = "application/vnd.github+json"

#: `owner/name`, as opposed to anything git would recognize as a URL.
_FULL_NAME = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def is_full_name(spec: str) -> bool:
    """True when `spec` names a GitHub repo the API can be asked about."""
    return bool(_FULL_NAME.match(spec))


@dataclass(frozen=True)
class RepoInfo:
    full_name: str
    private: bool
    default_branch: str
    clone_url: str
    ssh_url: str
    html_url: str
    #: False when the token can read the repo but not write to it. GitHub omits
    #: `permissions` for unauthenticated reads, in which case this is False too.
    can_push: bool
    empty: bool

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> RepoInfo:
        permissions = payload.get("permissions") or {}
        return cls(
            full_name=str(payload["full_name"]),
            private=bool(payload["private"]),
            default_branch=str(payload.get("default_branch") or "main"),
            clone_url=str(payload["clone_url"]),
            ssh_url=str(payload["ssh_url"]),
            html_url=str(payload["html_url"]),
            can_push=bool(permissions.get("push") or permissions.get("admin")),
            empty=payload.get("size", 1) == 0,
        )


class GitHubClient:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = DEFAULT_API_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owned = client is None
        self._client = client or httpx.AsyncClient(base_url=base_url, timeout=30.0)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": _ACCEPT,
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owned:
            await self._client.aclose()

    # -- endpoints -----------------------------------------------------------

    async def login(self) -> str:
        """The authenticated user's login. Doubles as a token check."""
        return str((await self._request("GET", "/user"))["login"])

    async def get_repo(self, full_name: str) -> RepoInfo | None:
        """Look up `owner/name`. None when it does not exist or is invisible."""
        payload = await self._request("GET", f"/repos/{full_name}", allow_404=True)
        return RepoInfo.parse(payload) if payload is not None else None

    async def create_repo(
        self, full_name: str, *, private: bool = True, description: str = ""
    ) -> RepoInfo:
        owner, _, name = full_name.partition("/")
        body = {"name": name, "private": private, "description": description, "auto_init": False}
        # A repo under someone else's namespace is an org repo; under our own it
        # is a user repo, and the two live at different endpoints.
        path = "/user/repos" if owner == await self.login() else f"/orgs/{owner}/repos"
        return RepoInfo.parse(await self._request("POST", path, json=body))

    # -- plumbing ------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> Any:
        try:
            response = await self._client.request(
                method, path, headers=self._headers, json=json, follow_redirects=True
            )
        except httpx.HTTPError as exc:
            raise GitHubError(f"could not reach GitHub: {exc}") from exc

        if response.status_code == 404 and allow_404:
            return None
        if response.status_code >= 400:
            raise GitHubError(_explain(response), status=response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubError(f"GitHub returned a non-JSON body for {path}") from exc


def _explain(response: httpx.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("message") or ""
        for error in payload.get("errors") or []:
            if message := error.get("message"):
                detail = f"{detail}: {message}"
    except ValueError:
        detail = response.text[:200]

    hint = {
        401: " — the token is invalid or expired",
        403: " — the token lacks the required scope (needs `contents: write` on the repo)",
        404: " — no such repository, or the token cannot see it",
        422: " — GitHub rejected the request as invalid (a repo by that name may already exist)",
    }.get(response.status_code, "")
    return f"GitHub {response.status_code}: {detail}{hint}"

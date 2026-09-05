"""Typed errors.

Providers raise these instead of leaking SDK or HTTP exceptions, so the retry
and fallback policy in `kasa.llm.registry` can make decisions without knowing
which vendor it is talking to.
"""

from __future__ import annotations


class KasaError(Exception):
    """Base class for every error Kasa raises deliberately."""


class ConfigError(KasaError):
    """Configuration is missing, malformed, or internally inconsistent."""


class LLMError(KasaError):
    """Something went wrong talking to a model provider."""

    #: Whether retrying the identical request could plausibly succeed.
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status

    def __str__(self) -> str:
        prefix = f"[{self.provider}] " if self.provider else ""
        return f"{prefix}{super().__str__()}"


class AuthError(LLMError):
    """Credentials were rejected. Retrying will not help."""


class RateLimitError(LLMError):
    """Provider asked us to slow down."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, status=status)
        self.retry_after = retry_after


class TransientError(LLMError):
    """Network blip, 5xx, or a timeout. Worth retrying."""

    retryable = True


class ContextOverflowError(LLMError):
    """The request exceeded the model's context window.

    Not retryable as-is: the caller has to shrink the request. The packer is
    supposed to prevent this, so raising it means the packer's token estimate
    drifted from the provider's.
    """


class ContentFilterError(LLMError):
    """The provider refused to return the completion on policy grounds."""


class ProviderProtocolError(LLMError):
    """The provider returned something we cannot parse.

    Most often an "OpenAI-compatible" server that is compatible only in
    aspiration.
    """


class BudgetExceededError(LLMError):
    """A non-interactive model call was stopped by the daily spend ceiling."""


class ToolError(KasaError):
    """A tool could not be dispatched, or failed in a way the agent should see."""


class SearchError(KasaError):
    """A web search could not be performed.

    Not an `LLMError`: nothing retries or falls back over it. It travels up to
    the tool dispatcher, which turns it into an `is_error` tool result for the
    model to read and work around.
    """


class FetchError(KasaError):
    """A URL could not be read. Travels the same route as `SearchError`.

    Its message is shown to the model, so it says what went wrong and never
    quotes the response that went wrong — an error page is somebody else's
    text, and the whole point of `kasa/fetch` is that such text arrives on the
    untrusted side of a delimiter or not at all.
    """


class Blocked(FetchError):
    """The URL was refused before anything was sent to it.

    Its own class because it is the one fetch failure that is a decision rather
    than an accident: rephrasing will not help and retrying will not either,
    and `kasa/fetch/guard.py` is the only thing that raises it.
    """


class StoreError(KasaError):
    """The database could not be opened or read.

    Almost always a file that is not a database, or one truncated by a full
    disk or a kill mid-write. The database is derived — the memory repo is the
    source of truth — so the message says so, because the recovery is to delete
    it and reindex rather than to try to repair anything.
    """


class GitError(KasaError):
    """A git command failed.

    Carries the command's own stderr: git's diagnostics are better than
    anything we would write over the top of them.
    """

    def __init__(self, message: str, *, command: str | None = None, stderr: str | None = None):
        super().__init__(message)
        self.command = command
        self.stderr = stderr

    def __str__(self) -> str:
        detail = (self.stderr or "").strip()
        return f"{super().__str__()}\n{detail}" if detail else super().__str__()


class GitHubError(KasaError):
    """The GitHub API rejected a request or returned something unusable."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

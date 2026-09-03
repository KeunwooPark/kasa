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


class ToolError(KasaError):
    """A tool could not be dispatched, or failed in a way the agent should see."""

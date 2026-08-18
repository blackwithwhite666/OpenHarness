"""API error types for OpenHarness."""

from __future__ import annotations


class OpenHarnessApiError(RuntimeError):
    """Base class for upstream API failures."""


class AuthenticationFailure(OpenHarnessApiError):
    """Raised when the upstream service rejects the provided credentials."""


class RateLimitFailure(OpenHarnessApiError):
    """Raised when the upstream service rejects the request due to rate limits."""


class QuotaExceededError(OpenHarnessApiError):
    """Raised when the provider rejects the request because the subscription
    quota/usage limit is exhausted (billing-window scoped, not transient).

    Unlike RateLimitFailure these errors never clear within a retry backoff
    window, so clients must fail fast instead of retrying."""


class RequestFailure(OpenHarnessApiError):
    """Raised for generic request or transport failures."""


# Provider messages that mean "subscription quota exhausted" (case-insensitive
# substring match). Covers Codex ("The usage limit has been reached"), Kimi
# ("You've reached your usage limit for this billing cycle. Your quota will be
# refreshed..."), and OpenAI ("insufficient_quota" / "You exceeded your current
# quota"). Deliberately excludes generic transient "rate limit" wording.
_QUOTA_MESSAGE_MARKERS: tuple[str, ...] = (
    "usage limit",
    "usage_limit",
    "usage_limit_reached",
    "insufficient_quota",
    "exceeded your current quota",
    "your quota",
    "quota will be refreshed",
    "billing cycle",
    "plan_limit",
)


def is_quota_error_message(message: str) -> bool:
    """Return True when a provider error message indicates an exhausted
    subscription quota rather than a transient rate limit."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _QUOTA_MESSAGE_MARKERS)

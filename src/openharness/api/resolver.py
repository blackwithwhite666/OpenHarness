"""Runtime API client resolver shared by CLI, UI, and eval runners."""

from __future__ import annotations

from openharness.api.client import AnthropicApiClient, SupportsStreamingMessages
from openharness.api.codex_client import CodexApiClient
from openharness.api.copilot_client import CopilotClient
from openharness.api.openai_client import OpenAICompatibleClient
from openharness.config.settings import Settings


class ApiClientResolutionError(ValueError):
    """Raised when settings cannot produce a runtime API client."""


def resolve_api_client_from_settings(settings: Settings) -> SupportsStreamingMessages:
    """Build the appropriate API client for the resolved settings."""
    # Ensure profile fields (base_url, model, api_format) are projected to settings.
    settings = settings.materialize_active_profile()

    def _safe_resolve_auth():
        try:
            return settings.resolve_auth()
        except ValueError as exc:
            raise ApiClientResolutionError(str(exc)) from exc

    if settings.api_format == "copilot":
        from openharness.api.copilot_client import COPILOT_DEFAULT_MODEL

        copilot_model = (
            COPILOT_DEFAULT_MODEL
            if settings.model
            in {
                "claude-sonnet-4-20250514",
                "claude-sonnet-4-6",
                "sonnet",
                "default",
            }
            else settings.model
        )
        return CopilotClient(model=copilot_model)
    if settings.provider == "openai_codex":
        auth = _safe_resolve_auth()
        return CodexApiClient(
            auth_token=auth.value,
            base_url=settings.base_url,
            # Re-resolve before each request so a long-running gateway picks up a
            # refreshed/rotated codex token (resolve_auth refreshes on expiry) -
            # otherwise it 401s on the captured token until restart.
            auth_token_resolver=lambda: settings.resolve_auth().value,
        )
    if settings.provider == "anthropic_claude":
        return AnthropicApiClient(
            auth_token=_safe_resolve_auth().value,
            base_url=settings.base_url,
            claude_oauth=True,
            auth_token_resolver=lambda: settings.resolve_auth().value,
        )
    if settings.api_format in ("openai", "openai_compat"):
        auth = _safe_resolve_auth()
        if settings.provider == "kimi_coding":
            from openharness.auth.external import kimi_api_headers

            return OpenAICompatibleClient(
                api_key=auth.value,
                base_url=settings.base_url,
                timeout=settings.timeout,
                default_headers=kimi_api_headers(),
                # Re-resolve before each request so a long-running gateway picks
                # up a refreshed/rotated kimi token (resolve_auth refreshes on
                # expiry) instead of 401-ing on the captured token until restart.
                api_key_resolver=lambda: settings.resolve_auth().value,
            )
        if settings.provider == "openrouter":
            # OpenRouter accepts OpenAI's reasoning_effort hint; the flag keeps
            # the parameter off requests to Kimi and other strict gateways.
            return OpenAICompatibleClient(
                api_key=auth.value,
                base_url=settings.base_url,
                timeout=settings.timeout,
                supports_reasoning_effort=True,
            )
        return OpenAICompatibleClient(
            api_key=auth.value,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )
    auth = _safe_resolve_auth()
    return AnthropicApiClient(
        api_key=auth.value,
        base_url=settings.base_url,
    )

from __future__ import annotations

from types import SimpleNamespace

import pytest

from openharness.api.client import AnthropicApiClient
from openharness.api.codex_client import CodexApiClient
from openharness.api.openai_client import OpenAICompatibleClient
from openharness.api.resolver import (
    ApiClientResolutionError,
    resolve_api_client_from_settings,
)
from openharness.config.settings import Settings
from openharness.ui import runtime as ui_runtime


def test_resolve_api_client_from_settings_builds_openai_compatible_client():
    client = resolve_api_client_from_settings(
        Settings(active_profile="openai-compatible", api_key="sk-test")
    )

    assert isinstance(client, OpenAICompatibleClient)


def test_ui_private_resolver_delegates_to_shared_resolver(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        ui_runtime,
        "resolve_api_client_from_settings",
        lambda settings: sentinel,
    )

    assert ui_runtime._resolve_api_client_from_settings(Settings(api_key="sk-test")) is sentinel


def test_resolve_api_client_from_settings_raises_typed_error_for_auth_failures():
    class _SettingsWithoutAuth:
        api_format = "openai"
        provider = "openai"
        base_url = None
        timeout = 60.0

        def materialize_active_profile(self):
            return self

        def resolve_auth(self):
            raise ValueError("missing credentials")

    with pytest.raises(ApiClientResolutionError, match="missing credentials"):
        resolve_api_client_from_settings(_SettingsWithoutAuth())  # type: ignore[arg-type]


def test_resolve_api_client_from_settings_preserves_codex_token_resolver():
    settings = _RotatingAuthSettings(
        provider="openai_codex",
        api_format="openai",
        tokens=["initial-token", "fresh-token"],
    )

    client = resolve_api_client_from_settings(settings)  # type: ignore[arg-type]

    assert isinstance(client, CodexApiClient)
    assert client._auth_token == "initial-token"
    assert client._auth_token_resolver is not None
    assert client._auth_token_resolver() == "fresh-token"


def test_resolve_api_client_from_settings_preserves_claude_oauth_token_resolver():
    settings = _RotatingAuthSettings(
        provider="anthropic_claude",
        api_format="anthropic",
        tokens=["initial-token", "fresh-token"],
    )

    client = resolve_api_client_from_settings(settings)  # type: ignore[arg-type]

    assert isinstance(client, AnthropicApiClient)
    assert client._auth_token == "initial-token"
    assert client._auth_token_resolver is not None
    assert client._auth_token_resolver() == "fresh-token"


class _RotatingAuthSettings:
    model = "test-model"
    base_url = None
    timeout = 60.0

    def __init__(self, *, provider: str, api_format: str, tokens: list[str]) -> None:
        self.provider = provider
        self.api_format = api_format
        self._tokens = tokens
        self._index = 0

    def materialize_active_profile(self):
        return self

    def resolve_auth(self):
        value = self._tokens[min(self._index, len(self._tokens) - 1)]
        self._index += 1
        return SimpleNamespace(value=value)


def test_resolve_api_client_from_settings_kimi_uses_headers_and_resolver(monkeypatch):
    from openharness.auth import external as external_mod

    monkeypatch.setattr(
        external_mod,
        "kimi_api_headers",
        lambda: {"User-Agent": "KimiCLI/1.41.0", "X-Msh-Platform": "kimi_cli"},
    )
    settings = _RotatingAuthSettings(
        provider="kimi_coding",
        api_format="openai",
        tokens=["initial-token", "fresh-token"],
    )
    settings.base_url = "https://api.kimi.com/coding/v1"

    client = resolve_api_client_from_settings(settings)  # type: ignore[arg-type]

    assert isinstance(client, OpenAICompatibleClient)
    assert client._client.api_key == "initial-token"
    assert client._custom_headers["X-Msh-Platform"] == "kimi_cli"
    assert client._api_key_resolver is not None
    assert client._api_key_resolver() == "fresh-token"

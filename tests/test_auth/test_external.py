from __future__ import annotations

import base64
import io
import json
import urllib.error
from pathlib import Path

import pytest
from typer.testing import CliRunner

from openharness.auth.external import (
    CLAUDE_PROVIDER,
    CODEX_PROVIDER,
    ExternalAuthState,
    describe_external_binding,
    default_binding_for_provider,
    get_claude_code_version,
    _load_codex_credential,
    load_external_credential,
    refresh_claude_oauth_credential,
    refresh_codex_oauth_credential,
)
from openharness.auth.storage import ExternalAuthBinding, load_external_binding, store_external_binding
from openharness.cli import app
from openharness.config.settings import Settings, load_settings


def _b64url(data: dict[str, object]) -> str:
    raw = json.dumps(data, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _fake_jwt(payload: dict[str, object]) -> str:
    return f"{_b64url({'alg': 'none', 'typ': 'JWT'})}.{_b64url(payload)}.sig"


def test_load_codex_external_credential(monkeypatch, tmp_path: Path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    token = _fake_jwt(
        {
            "exp": 4_102_444_800,
            "https://api.openai.com/profile": {"email": "dev@example.com"},
        }
    )
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": token,
                    "refresh_token": "refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    binding = default_binding_for_provider(CODEX_PROVIDER)
    credential = load_external_credential(binding)

    assert credential.provider == CODEX_PROVIDER
    assert credential.auth_kind == "api_key"
    assert credential.value == token
    assert credential.refresh_token == "refresh-token"
    assert credential.profile_label == "dev@example.com"
    assert credential.expires_at_ms == 4_102_444_800_000


def test_refresh_codex_oauth_credential_builds_form_request(monkeypatch):
    captured: dict[str, str] = {}
    new_token = _fake_jwt({"exp": 5_000_000_000})

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"access_token": new_token, "refresh_token": "rt2", "id_token": "id2", "expires_in": 864000}
            ).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data.decode()
        return _Resp()

    monkeypatch.setattr("openharness.auth.external.urllib.request.urlopen", fake_urlopen)
    out = refresh_codex_oauth_credential("rt1")

    import urllib.parse

    form = dict(urllib.parse.parse_qsl(captured["data"]))
    assert "auth.openai.com" in captured["url"]
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "rt1"
    assert form["client_id"]  # codex app client_id present
    assert out["access_token"] == new_token
    assert out["refresh_token"] == "rt2"  # rotation surfaced
    assert out["id_token"] == "id2"
    assert out["expires_at_ms"] == 5_000_000_000_000  # from the JWT exp


def test_refresh_codex_invalid_grant_raises(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"invalid_grant"}')
        )

    monkeypatch.setattr("openharness.auth.external.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="codex login"):
        refresh_codex_oauth_credential("rt-dead")


def test_load_codex_credential_refreshes_and_persists_when_expired(monkeypatch, tmp_path: Path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    expired = _fake_jwt({"exp": 1})  # 1970 → expired
    fresh = _fake_jwt({"exp": 5_000_000_000})
    auth_path = codex_home / "auth.json"
    auth_path.write_text(
        json.dumps(
            {"auth_mode": "chatgpt", "tokens": {"access_token": expired, "refresh_token": "rt1", "id_token": "id1"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(
        "openharness.auth.external.refresh_codex_oauth_credential",
        lambda rt: {
            "access_token": fresh,
            "refresh_token": "rt2",
            "id_token": "id2",
            "expires_at_ms": 5_000_000_000_000,
        },
    )
    binding = default_binding_for_provider(CODEX_PROVIDER)
    credential = load_external_credential(binding, refresh_if_needed=True)
    assert credential.value == fresh
    assert credential.refresh_token == "rt2"
    # rotation persisted back to auth.json (incl. id_token + last_refresh)
    saved = json.loads(auth_path.read_text())
    assert saved["tokens"]["access_token"] == fresh
    assert saved["tokens"]["refresh_token"] == "rt2"
    assert saved["tokens"]["id_token"] == "id2"
    assert saved["last_refresh"]


def test_load_codex_credential_does_not_refresh_when_not_requested(monkeypatch, tmp_path: Path):
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    expired = _fake_jwt({"exp": 1})
    (codex_home / "auth.json").write_text(
        json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": expired, "refresh_token": "rt1"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    calls = {"n": 0}

    def _should_not_run(rt):
        calls["n"] += 1
        return {}

    monkeypatch.setattr("openharness.auth.external.refresh_codex_oauth_credential", _should_not_run)
    binding = default_binding_for_provider(CODEX_PROVIDER)
    credential = load_external_credential(binding, refresh_if_needed=False)
    assert credential.value == expired and calls["n"] == 0


def test_load_codex_double_check_skips_refresh_when_already_rotated(monkeypatch, tmp_path: Path):
    # Race guard: caller read a stale token, but another task already refreshed the
    # file before we took the lock → re-read inside the lock returns the fresh token
    # WITHOUT consuming the (now-rotated) refresh_token again.
    auth_path = tmp_path / "auth.json"
    fresh = _fake_jwt({"exp": 4_102_444_800})
    auth_path.write_text(
        json.dumps({"tokens": {"access_token": fresh, "refresh_token": "rt-new"}}), encoding="utf-8"
    )
    stale = _fake_jwt({"exp": 1})
    stale_payload = {"tokens": {"access_token": stale, "refresh_token": "rt-old"}}
    binding = ExternalAuthBinding(
        provider=CODEX_PROVIDER,
        source_path=str(auth_path),
        source_kind="codex_auth_json",
        managed_by="codex-cli",
        profile_label="Codex CLI",
    )
    calls = {"n": 0}

    def _should_not_run(rt):
        calls["n"] += 1
        return {}

    monkeypatch.setattr("openharness.auth.external.refresh_codex_oauth_credential", _should_not_run)
    credential = _load_codex_credential(stale_payload, auth_path, binding, refresh_if_needed=True)
    assert credential.value == fresh and calls["n"] == 0


def test_load_claude_external_credential(monkeypatch, tmp_path: Path):
    claude_home = tmp_path / "claude-home"
    claude_home.mkdir()
    (claude_home / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "claude-access-token",
                    "refreshToken": "claude-refresh-token",
                    "expiresAt": 4_102_444_800_000,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_HOME", str(claude_home))
    monkeypatch.setattr("openharness.auth.external.platform.system", lambda: "Linux")

    binding = default_binding_for_provider(CLAUDE_PROVIDER)
    credential = load_external_credential(binding)

    assert credential.provider == CLAUDE_PROVIDER
    assert credential.auth_kind == "auth_token"
    assert credential.value == "claude-access-token"
    assert credential.refresh_token == "claude-refresh-token"
    assert credential.expires_at_ms == 4_102_444_800_000


def test_default_claude_binding_uses_keychain_on_macos(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_HOME", raising=False)
    monkeypatch.setattr("openharness.auth.external.platform.system", lambda: "Darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    binding = default_binding_for_provider(CLAUDE_PROVIDER)

    assert binding.source_kind == "claude_credentials_keychain"
    assert binding.source_path == "keychain:Claude Code-credentials"


def test_default_claude_binding_prefers_config_dir_on_macos(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr("openharness.auth.external.platform.system", lambda: "Darwin")

    binding = default_binding_for_provider(CLAUDE_PROVIDER)

    assert binding.source_kind == "claude_credentials_json"
    assert Path(binding.source_path) == config_dir / ".credentials.json"


def test_load_claude_external_credential_from_keychain(monkeypatch, tmp_path: Path):
    login_keychain = tmp_path / "login.keychain-db"

    def _fake_check_output(args, text=True):
        if args == ["security", "find-generic-password", "-w", "-s", "Claude Code-credentials"]:
            return json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "claude-access-token",
                        "refreshToken": "claude-refresh-token",
                        "expiresAt": 4_102_444_800_000,
                    }
                }
            )
        if args == ["security", "find-generic-password", "-s", "Claude Code-credentials"]:
            return (
                f'keychain: "{login_keychain}"\n'
                'attributes:\n'
                '    "acct"<blob>="yanchundong"\n'
                '    "svce"<blob>="Claude Code-credentials"\n'
            )
        raise AssertionError(args)

    monkeypatch.setattr("openharness.auth.external.subprocess.check_output", _fake_check_output)

    credential = load_external_credential(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path="keychain:Claude Code-credentials",
            source_kind="claude_credentials_keychain",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    )

    assert credential.provider == CLAUDE_PROVIDER
    assert credential.auth_kind == "auth_token"
    assert credential.value == "claude-access-token"
    assert credential.refresh_token == "claude-refresh-token"
    assert credential.expires_at_ms == 4_102_444_800_000
    assert credential.source_path == login_keychain
    assert credential.profile_label == "yanchundong"


def test_settings_resolve_auth_uses_external_binding(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    source = tmp_path / "claude-credentials.json"
    source.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "bound-claude-token",
                    "refreshToken": "bound-claude-refresh",
                    "expiresAt": 4_102_444_800_000,
                }
            }
        ),
        encoding="utf-8",
    )
    store_external_binding(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path=str(source),
            source_kind="claude_credentials_json",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    )

    resolved = Settings(active_profile="claude-subscription").resolve_auth()

    assert resolved.auth_kind == "auth_token"
    assert resolved.value == "bound-claude-token"
    assert str(source) in resolved.source


def test_settings_resolve_auth_refreshes_expired_external_binding(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    source = tmp_path / "claude-credentials.json"
    source.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "openharness.auth.external.refresh_claude_oauth_credential",
        lambda refresh_token: {
            "access_token": "fresh-token",
            "refresh_token": refresh_token,
            "expires_at_ms": 4_102_444_800_000,
        },
    )
    store_external_binding(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path=str(source),
            source_kind="claude_credentials_json",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    )

    resolved = Settings(active_profile="claude-subscription").resolve_auth()

    assert resolved.value == "fresh-token"
    persisted = json.loads(source.read_text(encoding="utf-8"))
    assert persisted["claudeAiOauth"]["accessToken"] == "fresh-token"
    assert persisted["claudeAiOauth"]["refreshToken"] == "refresh-token"


def test_cli_codex_login_binds_without_switching(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    codex_home = tmp_path / "codex-home"
    config_dir.mkdir()
    codex_home.mkdir()
    token = _fake_jwt({"exp": 4_102_444_800})
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": token,
                    "refresh_token": "refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    # Prevent env var leakage from overriding the configured api_key
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    (config_dir / "settings.json").write_text(
        json.dumps(
            {
                "api_format": "openai",
                "provider": "openai",
                "model": "kimi-k2.5",
                "base_url": "https://api.moonshot.cn/anthropic",
                "api_key": "stale-key",
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "codex-login"])

    assert result.exit_code == 0
    settings = load_settings()
    assert settings.active_profile != "codex"
    assert settings.provider == "openai"
    assert settings.base_url == "https://api.moonshot.cn/anthropic"
    assert settings.api_key == "stale-key"
    assert "Use `oh provider use codex` to activate it." in result.stdout
    binding = load_external_binding(CODEX_PROVIDER)
    assert binding is not None
    assert Path(binding.source_path) == codex_home / "auth.json"


def test_cli_claude_login_binds_without_switching(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    claude_home = tmp_path / "claude-home"
    claude_home.mkdir()
    (claude_home / ".credentials.json").write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "claude-access-token",
                    "refreshToken": "claude-refresh-token",
                    "expiresAt": 4_102_444_800_000,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("CLAUDE_HOME", str(claude_home))
    monkeypatch.setattr("openharness.auth.external.platform.system", lambda: "Linux")

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "claude-login"])

    assert result.exit_code == 0
    settings = load_settings()
    assert settings.provider == "anthropic"
    assert settings.api_format == "anthropic"
    assert settings.active_profile == "claude-api"
    assert "Use `oh provider use claude-subscription` to activate it." in result.stdout
    binding = load_external_binding(CLAUDE_PROVIDER)
    assert binding is not None
    assert Path(binding.source_path) == claude_home / ".credentials.json"


def test_cli_claude_login_refreshes_expired_credentials(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    claude_home = tmp_path / "claude-home"
    claude_home.mkdir()
    source = claude_home / ".credentials.json"
    source.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-token",
                    "refreshToken": "claude-refresh-token",
                    "expiresAt": 1,
                    "scopes": ["user:inference"],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("CLAUDE_HOME", str(claude_home))
    monkeypatch.setattr("openharness.auth.external.platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "openharness.auth.external.refresh_claude_oauth_credential",
        lambda refresh_token: {
            "access_token": "fresh-token",
            "refresh_token": refresh_token,
            "expires_at_ms": 4_102_444_800_000,
        },
    )

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "claude-login"])

    assert result.exit_code == 0
    persisted = json.loads(source.read_text(encoding="utf-8"))
    assert persisted["claudeAiOauth"]["accessToken"] == "fresh-token"
    assert persisted["claudeAiOauth"]["scopes"] == ["user:inference"]


def test_load_claude_external_credential_refreshes_expired_keychain(monkeypatch, tmp_path: Path):
    login_keychain = tmp_path / "login.keychain-db"
    writes: list[list[str]] = []

    def _fake_check_output(args, text=True):
        if args == ["security", "find-generic-password", "-w", "-s", "Claude Code-credentials"]:
            return json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "expired-token",
                        "refreshToken": "refresh-token",
                        "expiresAt": 1,
                        "scopes": ["user:inference"],
                    }
                }
            )
        if args == ["security", "find-generic-password", "-s", "Claude Code-credentials"]:
            return (
                f'keychain: "{login_keychain}"\n'
                'attributes:\n'
                '    "acct"<blob>="yanchundong"\n'
                '    "svce"<blob>="Claude Code-credentials"\n'
            )
        raise AssertionError(args)

    def _fake_run(args, check=True, capture_output=True, text=True):
        writes.append(args)
        return None

    monkeypatch.setattr("openharness.auth.external.subprocess.check_output", _fake_check_output)
    monkeypatch.setattr("openharness.auth.external.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "openharness.auth.external.refresh_claude_oauth_credential",
        lambda refresh_token: {
            "access_token": "fresh-token",
            "refresh_token": refresh_token,
            "expires_at_ms": 4_102_444_800_000,
        },
    )

    credential = load_external_credential(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path="keychain:Claude Code-credentials",
            source_kind="claude_credentials_keychain",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        ),
        refresh_if_needed=True,
    )

    assert credential.value == "fresh-token"
    assert writes
    assert writes[0][:6] == [
        "security",
        "add-generic-password",
        "-U",
        "-s",
        "Claude Code-credentials",
        "-a",
    ]
    assert "yanchundong" in writes[0]


def test_cli_provider_use_activates_codex_profile(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    codex_home = tmp_path / "codex-home"
    config_dir.mkdir()
    codex_home.mkdir()
    token = _fake_jwt({"exp": 4_102_444_800})
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": token,
                    "refresh_token": "refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    runner = CliRunner()
    assert runner.invoke(app, ["auth", "codex-login"]).exit_code == 0

    result = runner.invoke(app, ["provider", "use", "codex"])

    assert result.exit_code == 0
    settings = load_settings()
    assert settings.active_profile == "codex"
    assert settings.provider == CODEX_PROVIDER
    assert settings.api_format == "openai"
    assert settings.base_url is None
    assert settings.model == "gpt-5.4"


def test_settings_resolve_auth_rejects_third_party_base_url_for_claude_subscription(
    monkeypatch,
    tmp_path: Path,
):
    config_dir = tmp_path / "config"
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    source = tmp_path / "claude-credentials.json"
    source.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "valid-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 4_102_444_800_000,
                }
            }
        ),
        encoding="utf-8",
    )
    store_external_binding(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path=str(source),
            source_kind="claude_credentials_json",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    )
    settings = Settings(active_profile="claude-subscription").model_copy(
        update={"base_url": "https://api.moonshot.cn/anthropic"}
    ).sync_active_profile_from_flat_fields()

    with pytest.raises(ValueError, match="third-party"):
        settings.resolve_auth()


def test_describe_external_binding_reports_refreshable_claude_token(tmp_path: Path):
    source = tmp_path / "claude-credentials.json"
    source.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    state = describe_external_binding(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path=str(source),
            source_kind="claude_credentials_json",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    )

    assert state == ExternalAuthState(
        configured=True,
        state="refreshable",
        source="external",
        detail=f"expired token can be refreshed from {source}",
    )


def test_describe_external_binding_reports_configured_claude_keychain(
    monkeypatch, tmp_path: Path
):
    login_keychain = tmp_path / "login.keychain-db"

    def _fake_check_output(args, text=True):
        if args == ["security", "find-generic-password", "-w", "-s", "Claude Code-credentials"]:
            return json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "claude-access-token",
                        "refreshToken": "claude-refresh-token",
                        "expiresAt": 4_102_444_800_000,
                    }
                }
            )
        if args == ["security", "find-generic-password", "-s", "Claude Code-credentials"]:
            return (
                f'keychain: "{login_keychain}"\n'
                'attributes:\n'
                '    "acct"<blob>="yanchundong"\n'
                '    "svce"<blob>="Claude Code-credentials"\n'
            )
        raise AssertionError(args)

    monkeypatch.setattr("openharness.auth.external.subprocess.check_output", _fake_check_output)

    state = describe_external_binding(
        ExternalAuthBinding(
            provider=CLAUDE_PROVIDER,
            source_path="keychain:Claude Code-credentials",
            source_kind="claude_credentials_keychain",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    )

    assert state == ExternalAuthState(
        configured=True,
        state="configured",
        source="external",
        detail=str(login_keychain),
    )


def test_refresh_claude_oauth_credential(monkeypatch):
    seen: dict[str, object] = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "access_token": "fresh-token",
                    "refresh_token": "fresh-refresh",
                    "expires_in": 7200,
                }
            ).encode("utf-8")

    def _fake_urlopen(request, timeout=10):
        seen["timeout"] = timeout
        seen["headers"] = dict(request.header_items())
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("openharness.auth.external.urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("openharness.auth.external.time.time", lambda: 1000)

    refreshed = refresh_claude_oauth_credential("refresh-token")

    assert refreshed["access_token"] == "fresh-token"
    assert refreshed["refresh_token"] == "fresh-refresh"
    assert refreshed["expires_at_ms"] == (1000 * 1000) + (7200 * 1000)
    assert seen["timeout"] == 10
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["body"]["grant_type"] == "refresh_token"
    assert seen["body"]["refresh_token"] == "refresh-token"
    assert "user:inference" in seen["body"]["scope"]


def test_refresh_claude_oauth_credential_reports_invalid_grant(monkeypatch):
    class _FakeResponse:
        def read(self):
            return b'{"error":"invalid_grant","error_description":"Refresh token not found or invalid"}'

        def close(self):
            return None

    error = urllib.error.HTTPError(
        "https://platform.claude.com/v1/oauth/token",
        400,
        "Bad Request",
        hdrs=None,
        fp=_FakeResponse(),
    )

    monkeypatch.setattr(
        "openharness.auth.external.urllib.request.urlopen",
        lambda request, timeout=10: (_ for _ in ()).throw(error),
    )

    with pytest.raises(ValueError, match="claude auth login"):
        refresh_claude_oauth_credential("refresh-token")


def test_get_claude_code_version_uses_fallback(monkeypatch):
    class _Result:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(
        "openharness.auth.external.subprocess.run",
        lambda *args, **kwargs: _Result(),
    )
    monkeypatch.setattr("openharness.auth.external._claude_code_version_cache", None)

    assert get_claude_code_version() == "2.1.92"

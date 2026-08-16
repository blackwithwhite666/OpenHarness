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


# ---- Kimi For Coding (OAuth) ----


def _kimi_binding(tmp_path: Path) -> ExternalAuthBinding:
    return ExternalAuthBinding(
        provider="kimi_coding",
        source_path=str(tmp_path / "kimi-home" / "openharness_auth.json"),
        source_kind="kimi_oauth_json",
        managed_by="openharness",
        profile_label="Kimi For Coding",
    )


def _write_kimi_auth(path: Path, access: str, refresh: str, expires_at_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"access_token": access, "refresh_token": refresh, "expires_at_ms": expires_at_ms}),
        encoding="utf-8",
    )


def test_default_kimi_binding_uses_kimi_home(monkeypatch, tmp_path: Path):
    from openharness.auth.external import KIMI_PROVIDER

    monkeypatch.setenv("KIMI_HOME", str(tmp_path / "kimi-home"))

    binding = default_binding_for_provider(KIMI_PROVIDER)

    assert binding.source_kind == "kimi_oauth_json"
    assert Path(binding.source_path) == tmp_path / "kimi-home" / "openharness_auth.json"
    assert binding.managed_by == "openharness"


def test_load_kimi_external_credential(tmp_path: Path):
    from openharness.auth.external import load_external_credential

    binding = _kimi_binding(tmp_path)
    source = Path(binding.source_path)
    _write_kimi_auth(source, "kimi-access", "kimi-refresh", 4_102_444_800_000)

    credential = load_external_credential(binding)

    assert credential.provider == "kimi_coding"
    assert credential.value == "kimi-access"
    assert credential.refresh_token == "kimi-refresh"
    assert credential.expires_at_ms == 4_102_444_800_000


def test_load_kimi_credential_does_not_refresh_fresh_token(monkeypatch, tmp_path: Path):
    from openharness.auth.external import load_external_credential

    binding = _kimi_binding(tmp_path)
    source = Path(binding.source_path)
    _write_kimi_auth(source, "kimi-access", "kimi-refresh", 4_102_444_800_000)
    calls = {"n": 0}

    def _should_not_run(rt):
        calls["n"] += 1
        return {}

    monkeypatch.setattr("openharness.auth.external.refresh_kimi_oauth_credential", _should_not_run)
    credential = load_external_credential(binding, refresh_if_needed=True)
    assert credential.value == "kimi-access" and calls["n"] == 0


def test_load_kimi_credential_refreshes_and_persists_when_expired(monkeypatch, tmp_path: Path):
    from openharness.auth.external import load_external_credential

    binding = _kimi_binding(tmp_path)
    source = Path(binding.source_path)
    _write_kimi_auth(source, "stale-access", "rt1", 1)
    monkeypatch.setattr(
        "openharness.auth.external.refresh_kimi_oauth_credential",
        lambda rt: {"access_token": "fresh-access", "refresh_token": "rt2", "expires_at_ms": 4_102_444_800_000},
    )

    credential = load_external_credential(binding, refresh_if_needed=True)

    assert credential.value == "fresh-access"
    assert credential.refresh_token == "rt2"
    saved = json.loads(source.read_text())
    assert saved["access_token"] == "fresh-access"
    assert saved["refresh_token"] == "rt2"  # rotation persisted
    assert saved["expires_at_ms"] == 4_102_444_800_000


def test_load_kimi_double_check_skips_refresh_when_already_rotated(monkeypatch, tmp_path: Path):
    # Race guard: caller saw an expired token, but another task already refreshed
    # the file before we took the lock -> no second refresh with the consumed token.
    from openharness.auth.external import _load_kimi_credential

    source = tmp_path / "auth.json"
    _write_kimi_auth(source, "fresh-access", "rt-new", 4_102_444_800_000)
    stale_payload = {"access_token": "stale-access", "refresh_token": "rt-old", "expires_at_ms": 1}
    binding = _kimi_binding(tmp_path)
    calls = {"n": 0}

    def _should_not_run(rt):
        calls["n"] += 1
        return {}

    monkeypatch.setattr("openharness.auth.external.refresh_kimi_oauth_credential", _should_not_run)
    credential = _load_kimi_credential(stale_payload, source, binding, refresh_if_needed=True)
    assert credential.value == "fresh-access" and calls["n"] == 0


def test_refresh_kimi_oauth_credential_builds_form_request(monkeypatch):
    from openharness.auth.external import refresh_kimi_oauth_credential

    captured: dict[str, object] = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(
                {"access_token": "new-access", "refresh_token": "rt2", "expires_in": 3600}
            ).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data.decode()
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _Resp()

    monkeypatch.setattr("openharness.auth.external.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("openharness.auth.external.time.time", lambda: 1000)
    out = refresh_kimi_oauth_credential("rt1")

    import urllib.parse

    form = dict(urllib.parse.parse_qsl(captured["data"]))
    assert captured["url"] == "https://auth.kimi.com/api/oauth/token"
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "rt1"
    assert form["client_id"] == "17e5f671-d194-4dfb-9706-5516cb48c098"
    headers = captured["headers"]
    assert headers["user-agent"] == "KimiCLI/1.41.0"
    assert headers["x-msh-platform"] == "kimi_cli"
    assert out["access_token"] == "new-access"
    assert out["refresh_token"] == "rt2"  # rotation surfaced
    assert out["expires_at_ms"] == 1000 * 1000 + 3600 * 1000


def test_refresh_kimi_invalid_grant_raises(monkeypatch):
    from openharness.auth.external import refresh_kimi_oauth_credential

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error":"invalid_grant"}')
        )

    monkeypatch.setattr("openharness.auth.external.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="kimi-login"):
        refresh_kimi_oauth_credential("rt-dead")


def test_kimi_device_flow_pending_then_success(monkeypatch):
    from openharness.auth.external import kimi_poll_device_token

    responses = iter(
        [
            {"error": "authorization_pending"},
            {"access_token": "kimi-access", "refresh_token": "kimi-refresh", "expires_in": 3600},
        ]
    )
    statuses = iter([400, 200])
    captured: list[dict[str, str]] = []

    class _Resp:
        def __init__(self, status, payload):
            self.status = status
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(self._payload).encode()

    def fake_urlopen(req, timeout=None):
        captured.append(dict(urllib.parse.parse_qsl(req.data.decode())))
        status = next(statuses)
        payload = next(responses)
        if status >= 400:
            raise urllib.error.HTTPError(
                req.full_url, status, "err", {}, io.BytesIO(json.dumps(payload).encode())
            )
        return _Resp(status, payload)

    import urllib.parse

    monkeypatch.setattr("openharness.auth.external.urllib.request.urlopen", fake_urlopen)
    sleeps: list[float] = []
    tokens = kimi_poll_device_token(
        {"device_code": "dc", "interval": 1, "expires_in": 60},
        sleep_fn=sleeps.append,
    )
    assert tokens["access_token"] == "kimi-access"
    assert sleeps == [1.0, 1.0]
    assert captured[-1]["device_code"] == "dc"
    assert captured[-1]["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"


def test_kimi_device_flow_slow_down_increases_interval(monkeypatch):
    from openharness.auth.external import kimi_poll_device_token

    calls = {"n": 0}

    def fake_post(url, params, *, timeout=30.0):
        calls["n"] += 1
        if calls["n"] == 1:
            err = ValueError("slow down")
            err.kimi_error = "slow_down"
            err.kimi_status = 400
            raise err
        return {"access_token": "a", "refresh_token": "r", "expires_in": 3600}

    monkeypatch.setattr("openharness.auth.external._kimi_post_form", fake_post)
    sleeps: list[float] = []
    tokens = kimi_poll_device_token(
        {"device_code": "dc", "interval": 2, "expires_in": 60},
        sleep_fn=sleeps.append,
    )
    assert tokens["access_token"] == "a"
    assert sleeps == [2.0, 7.0]


def test_kimi_device_flow_expired_raises(monkeypatch):
    from openharness.auth.external import kimi_poll_device_token

    def fake_post(url, params, *, timeout=30.0):
        err = ValueError("expired")
        err.kimi_error = "expired_token"
        err.kimi_status = 400
        raise err

    monkeypatch.setattr("openharness.auth.external._kimi_post_form", fake_post)
    with pytest.raises(ValueError, match="kimi-login"):
        kimi_poll_device_token({"device_code": "dc", "interval": 1, "expires_in": 60}, sleep_fn=lambda s: None)


def test_kimi_api_headers_and_device_id(monkeypatch, tmp_path: Path):
    from openharness.auth.external import kimi_api_headers

    monkeypatch.setenv("KIMI_HOME", str(tmp_path / "kimi-home"))
    headers = kimi_api_headers()
    assert headers["User-Agent"] == "KimiCLI/1.41.0"
    assert headers["X-Msh-Platform"] == "kimi_cli"
    assert headers["X-Msh-Version"] == "1.41.0"
    assert headers["X-Msh-Device-Id"]
    assert "-" not in headers["X-Msh-Device-Id"]
    assert headers["X-Msh-Device-Name"]
    assert headers["X-Msh-Device-Model"]
    assert headers["X-Msh-Os-Version"]
    # device id persisted and reused
    assert kimi_api_headers()["X-Msh-Device-Id"] == headers["X-Msh-Device-Id"]
    device_id_file = tmp_path / "kimi-home" / "device_id"
    assert device_id_file.read_text() == headers["X-Msh-Device-Id"]
    assert (device_id_file.stat().st_mode & 0o777) == 0o600


def test_settings_resolve_auth_uses_kimi_binding(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    source = tmp_path / "kimi-auth.json"
    _write_kimi_auth(source, "bound-kimi-token", "kimi-refresh", 4_102_444_800_000)
    store_external_binding(
        ExternalAuthBinding(
            provider="kimi_coding",
            source_path=str(source),
            source_kind="kimi_oauth_json",
            managed_by="openharness",
            profile_label="Kimi For Coding",
        )
    )

    resolved = Settings(active_profile="kimi").resolve_auth()

    assert resolved.value == "bound-kimi-token"
    assert str(source) in resolved.source


def test_cli_kimi_login_device_flow(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    kimi_home = tmp_path / "kimi-home"
    monkeypatch.setenv("OPENHARNESS_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KIMI_HOME", str(kimi_home))

    monkeypatch.setattr(
        "openharness.auth.external.kimi_start_device_auth",
        lambda: {
            "device_code": "dc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://kimi.com/device",
            "verification_uri_complete": "https://kimi.com/device?code=ABCD-1234",
            "expires_in": 600,
            "interval": 1,
        },
    )
    monkeypatch.setattr(
        "openharness.auth.external.kimi_poll_device_token",
        lambda device: {"access_token": "kimi-access", "refresh_token": "kimi-refresh", "expires_in": 3600},
    )

    runner = CliRunner()
    result = runner.invoke(app, ["auth", "kimi-login"])

    assert result.exit_code == 0
    assert "https://kimi.com/device?code=ABCD-1234" in result.stdout
    assert "Use `oh provider use kimi` to activate it." in result.stdout
    saved = json.loads((kimi_home / "openharness_auth.json").read_text())
    assert saved["access_token"] == "kimi-access"
    assert saved["refresh_token"] == "kimi-refresh"
    assert saved["expires_at_ms"] > 0
    binding = load_external_binding("kimi_coding")
    assert binding is not None
    assert Path(binding.source_path) == kimi_home / "openharness_auth.json"

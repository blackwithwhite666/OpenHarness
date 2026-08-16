"""Integration with external CLI-managed subscription credentials."""

from __future__ import annotations

import base64
import json
import os
import platform
import re
import subprocess
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.auth.storage import ExternalAuthBinding
from openharness.utils.fs import atomic_write_text

CODEX_PROVIDER = "openai_codex"
CLAUDE_PROVIDER = "anthropic_claude"
KIMI_PROVIDER = "kimi_coding"
# Kimi For Coding subscription OAuth (device flow + rotating refresh_token).
# Values mirror the official kimi-cli v1.41.0 (client_id is a public constant
# shipped inside that CLI, not a secret). The api.kimi.com backend 403s any
# request without the KimiCLI user agent and the X-Msh-* device headers
# ("access_terminated_error: only available for Coding Agents").
KIMI_CLI_VERSION = "1.41.0"
KIMI_USER_AGENT = f"KimiCLI/{KIMI_CLI_VERSION}"
KIMI_OAUTH_DEVICE_AUTH_URL = "https://auth.kimi.com/api/oauth/device_authorization"
KIMI_OAUTH_TOKEN_URL = "https://auth.kimi.com/api/oauth/token"
KIMI_OAUTH_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
KIMI_REFRESH_GRANT = "refresh_token"
KIMI_API_BASE_URL = "https://api.kimi.com/coding/v1"
# Same rationale as the codex lock: kimi refresh tokens rotate, so two
# concurrent refreshes would consume the same token and brick the chain.
_KIMI_REFRESH_LOCK = threading.Lock()
_KIMI_REFRESH_SKEW_MS = 60_000
_KIMI_REFRESH_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_KIMI_REFRESH_MAX_RETRIES = 3
# ChatGPT/Codex subscription OAuth: refresh the short-lived access token in
# ~/.codex/auth.json with its (rotating) refresh_token, so a long-running gateway
# self-heals instead of 401-ing until restart. client_id = the codex app audience
# from the id_token; endpoint = the OpenAI OAuth token endpoint.
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_TOKEN_ENDPOINT = "https://auth.openai.com/oauth/token"
# Codex refresh tokens are single-use/rotating: two concurrent refreshes would
# consume the same token and brick the chain (the loser 400s). Serialize the
# whole check→refresh→persist section in-process, with a double-checked re-read.
_CODEX_REFRESH_LOCK = threading.Lock()
# Refresh slightly BEFORE expiry so a request never goes out on a just-expired
# token (codex 401s aren't retried) — proactive instead of after a visible 401.
_CODEX_REFRESH_SKEW_MS = 60_000
CLAUDE_CODE_VERSION_FALLBACK = "2.1.92"
CLAUDE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_OAUTH_TOKEN_ENDPOINTS = (
    "https://platform.claude.com/v1/oauth/token",
    "https://console.anthropic.com/v1/oauth/token",
)
CLAUDE_COMMON_BETAS = (
    "interleaved-thinking-2025-05-14",
    "fine-grained-tool-streaming-2025-05-14",
)
CLAUDE_AI_OAUTH_SCOPES = (
    "user:profile",
    "user:inference",
    "user:sessions:claude_code",
    "user:mcp_servers",
    "user:file_upload",
)
CLAUDE_OAUTH_ONLY_BETAS = (
    "claude-code-20250219",
    "oauth-2025-04-20",
)
CLAUDE_KEYCHAIN_SERVICE = "Claude Code-credentials"
_KEYCHAIN_BINDING_PREFIX = "keychain:"

_claude_code_version_cache: str | None = None
_claude_code_session_id: str | None = None


@dataclass(frozen=True)
class ExternalAuthCredential:
    """Normalized external credential used at runtime."""

    provider: str
    value: str
    auth_kind: str
    source_path: Path
    managed_by: str
    profile_label: str = ""
    refresh_token: str = ""
    expires_at_ms: int | None = None


@dataclass(frozen=True)
class ExternalAuthState:
    """Human-readable state for an external auth source."""

    configured: bool
    state: str
    source: str
    detail: str = ""


def default_binding_for_provider(provider: str) -> ExternalAuthBinding:
    """Return the default external auth source for *provider*."""
    if provider == CODEX_PROVIDER:
        codex_home = Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
        return ExternalAuthBinding(
            provider=provider,
            source_path=str(codex_home / "auth.json"),
            source_kind="codex_auth_json",
            managed_by="codex-cli",
            profile_label="Codex CLI",
        )
    if provider == CLAUDE_PROVIDER:
        configured_dir = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
        if configured_dir:
            return ExternalAuthBinding(
                provider=provider,
                source_path=str(Path(configured_dir).expanduser() / ".credentials.json"),
                source_kind="claude_credentials_json",
                managed_by="claude-cli",
                profile_label="Claude CLI",
            )
        if platform.system() == "Darwin":
            return ExternalAuthBinding(
                provider=provider,
                source_path=f"{_KEYCHAIN_BINDING_PREFIX}{CLAUDE_KEYCHAIN_SERVICE}",
                source_kind="claude_credentials_keychain",
                managed_by="claude-cli",
                profile_label="Claude CLI",
            )
        claude_home = Path(os.environ.get("CLAUDE_HOME", "~/.claude")).expanduser()
        return ExternalAuthBinding(
            provider=provider,
            source_path=str(claude_home / ".credentials.json"),
            source_kind="claude_credentials_json",
            managed_by="claude-cli",
            profile_label="Claude CLI",
        )
    if provider == KIMI_PROVIDER:
        kimi_home = Path(os.environ.get("KIMI_HOME", "~/.kimi")).expanduser()
        return ExternalAuthBinding(
            provider=provider,
            source_path=str(kimi_home / "openharness_auth.json"),
            source_kind="kimi_oauth_json",
            managed_by="openharness",
            profile_label="Kimi For Coding",
        )
    raise ValueError(f"Unsupported external auth provider: {provider}")


def load_external_credential(
    binding: ExternalAuthBinding,
    *,
    refresh_if_needed: bool = False,
) -> ExternalAuthCredential:
    """Read a runtime credential from an external auth binding."""
    if binding.provider == CODEX_PROVIDER:
        source_path = Path(binding.source_path).expanduser()
        if not source_path.exists():
            raise ValueError(f"External auth source not found: {source_path}")
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in external auth source: {source_path}") from exc
        return _load_codex_credential(
            payload, source_path, binding, refresh_if_needed=refresh_if_needed
        )
    if binding.provider == CLAUDE_PROVIDER:
        payload, source_path, keychain_service, keychain_account = _load_claude_payload(binding)
        return _load_claude_credential(
            payload,
            source_path,
            binding,
            refresh_if_needed=refresh_if_needed,
            keychain_service=keychain_service,
            keychain_account=keychain_account,
        )
    if binding.provider == KIMI_PROVIDER:
        source_path = Path(binding.source_path).expanduser()
        if not source_path.exists():
            raise ValueError(f"External auth source not found: {source_path}")
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in external auth source: {source_path}") from exc
        return _load_kimi_credential(
            payload, source_path, binding, refresh_if_needed=refresh_if_needed
        )
    raise ValueError(f"Unsupported external auth provider: {binding.provider}")


def _load_codex_credential(
    payload: dict[str, Any],
    source_path: Path,
    binding: ExternalAuthBinding,
    *,
    refresh_if_needed: bool = False,
) -> ExternalAuthCredential:
    tokens = payload.get("tokens")
    access_token = ""
    refresh_token = ""
    if isinstance(tokens, dict):
        access_token = str(tokens.get("access_token", "") or "")
        refresh_token = str(tokens.get("refresh_token", "") or "")
    if not access_token:
        access_token = str(payload.get("OPENAI_API_KEY", "") or "")
    if not access_token:
        raise ValueError("Codex auth source does not contain an access token.")

    def _build(token: str, rtoken: str) -> ExternalAuthCredential:
        return ExternalAuthCredential(
            provider=CODEX_PROVIDER,
            value=token,
            auth_kind="api_key",
            source_path=source_path,
            managed_by=binding.managed_by,
            profile_label=(
                _decode_json_web_token_claim(token, ["https://api.openai.com/profile", "email"])
                or binding.profile_label
            ),
            refresh_token=rtoken,
            expires_at_ms=_decode_jwt_expiry(token),
        )

    credential = _build(access_token, refresh_token)
    # Self-heal an expired ChatGPT/Codex access token: a long-running gateway
    # captures the token once and the codex CLI only refreshes auth.json when it is
    # run, so without this the gateway 401s until a manual restart. Refresh with the
    # rotating refresh_token and persist (incl. the rotated refresh_token) so the
    # chain survives across runs.
    if refresh_if_needed and _codex_should_refresh(credential):
        if not refresh_token:
            raise ValueError(
                f"Codex credentials at {source_path} are expired and cannot be refreshed "
                "(no refresh_token). Run `codex login` to re-authenticate."
            )
        with _CODEX_REFRESH_LOCK:
            # Re-read inside the lock: a concurrent task (e.g. another teammate, or
            # the codex CLI) may have rotated the single-use refresh_token while we
            # waited. Refreshing again with the now-consumed token would 400 and
            # brick the chain — so if it's already fresh, just use it.
            try:
                cur_payload = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cur_payload = payload
            cur_tokens = cur_payload.get("tokens") if isinstance(cur_payload.get("tokens"), dict) else {}
            cur_access = str((cur_tokens or {}).get("access_token", "") or "")
            cur_refresh = str((cur_tokens or {}).get("refresh_token", "") or refresh_token)
            if cur_access:
                cur_cred = _build(cur_access, cur_refresh)
                if not _codex_should_refresh(cur_cred):
                    return cur_cred  # another task already refreshed
            refreshed = refresh_codex_oauth_credential(cur_refresh)
            _write_codex_auth_json(source_path, cur_payload, refreshed)
            credential = _build(
                str(refreshed["access_token"]),
                str(refreshed.get("refresh_token", cur_refresh) or cur_refresh),
            )
    return credential


def _codex_should_refresh(credential: ExternalAuthCredential, *, now_ms: int | None = None) -> bool:
    """True when the codex access token is expired or within the refresh skew."""
    if credential.expires_at_ms is None:
        return False
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    return credential.expires_at_ms - now_ms <= _CODEX_REFRESH_SKEW_MS


def refresh_codex_oauth_credential(refresh_token: str) -> dict[str, Any]:
    """Refresh a ChatGPT/Codex OAuth access token via its rotating refresh_token.

    Form-encoded ``refresh_token`` grant against the OpenAI token endpoint (verified
    request shape). Returns the new ``access_token``, the (rotated) ``refresh_token``,
    the fresh ``id_token`` if present, and ``expires_at_ms``. Does not write files.
    """
    if not refresh_token:
        raise ValueError("refresh_token is required")
    data = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CODEX_OAUTH_CLIENT_ID,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        CODEX_OAUTH_TOKEN_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        if "invalid_grant" in body:
            raise ValueError(
                "Codex OAuth refresh token is invalid or expired. "
                "Run `codex login` to re-authenticate the Codex CLI."
            ) from exc
        detail = f"{exc.code} {exc.reason}" + (f": {body}" if body else "")
        raise ValueError(f"Codex OAuth refresh failed: {detail}") from exc
    access_token = str(result.get("access_token", "") or "")
    if not access_token:
        raise ValueError("Codex OAuth refresh response missing access_token")
    expires_at_ms = _decode_jwt_expiry(access_token)
    if expires_at_ms is None:
        expires_at_ms = int(time.time() * 1000) + int(result.get("expires_in", 3600) or 3600) * 1000
    return {
        "access_token": access_token,
        "refresh_token": str(result.get("refresh_token", refresh_token) or refresh_token),
        "id_token": str(result.get("id_token", "") or ""),
        "expires_at_ms": expires_at_ms,
    }


def _write_codex_auth_json(
    source_path: Path, payload: dict[str, Any], refreshed: dict[str, Any]
) -> None:
    """Persist refreshed codex tokens back to auth.json (preserving other fields)."""
    updated = dict(payload)
    tokens = dict(updated.get("tokens") or {})
    tokens["access_token"] = str(refreshed["access_token"])
    if refreshed.get("refresh_token"):
        tokens["refresh_token"] = str(refreshed["refresh_token"])
    if refreshed.get("id_token"):
        tokens["id_token"] = str(refreshed["id_token"])
    updated["tokens"] = tokens
    updated["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S.000000000Z", time.gmtime())
    atomic_write_text(source_path, json.dumps(updated, indent=2))


def _ascii_header_value(value: str, fallback: str = "unknown") -> str:
    """Strip non-ASCII/control chars so the value is a legal HTTP header."""
    sanitized = re.sub(r"[^\x20-\x7e]", "", value).strip()
    return sanitized or fallback


def _kimi_home() -> Path:
    return Path(os.environ.get("KIMI_HOME", "~/.kimi")).expanduser()


def _kimi_device_id() -> str:
    """Persistent device fingerprint, shared with the official kimi-cli."""
    kimi_home = _kimi_home()
    kimi_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    device_id_path = kimi_home / "device_id"
    try:
        existing = device_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing
    device_id = uuid.uuid4().hex
    atomic_write_text(device_id_path, device_id)
    os.chmod(device_id_path, 0o600)
    return device_id


def _kimi_device_model() -> str:
    system = platform.system()
    release = platform.release()
    machine = platform.machine()
    if system == "Darwin":
        mac_version = platform.mac_ver()[0] or release
        if mac_version and machine:
            return f"macOS {mac_version} {machine}"
        if mac_version:
            return f"macOS {mac_version}"
        return f"macOS {machine}".strip()
    if system and release and machine:
        return f"{system} {release} {machine}"
    if system and release:
        return f"{system} {release}"
    if system:
        return f"{system} {machine}".strip()
    return "Unknown"


def kimi_api_headers() -> dict[str, str]:
    """Headers the kimi-for-coding backend requires on every request.

    Missing or deviating values make api.kimi.com 403 with
    "access_terminated_error: Kimi For Coding is currently only available for
    Coding Agents". Values mirror kimi-cli's `_common_headers`.
    """
    return {
        "User-Agent": KIMI_USER_AGENT,
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": KIMI_CLI_VERSION,
        "X-Msh-Device-Name": _ascii_header_value(platform.node() or "unknown"),
        "X-Msh-Device-Model": _ascii_header_value(_kimi_device_model()),
        "X-Msh-Device-Id": _kimi_device_id(),
        "X-Msh-Os-Version": _ascii_header_value(platform.version() or f"{platform.system()} {platform.release()}"),
    }


def _kimi_post_form(url: str, params: dict[str, str], *, timeout: float = 30.0) -> dict[str, Any]:
    """Form-POST to a kimi OAuth endpoint with the mandatory CLI headers."""
    data = urllib.parse.urlencode(params).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            **kimi_api_headers(),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        status = exc.code
    try:
        result = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Kimi OAuth: non-JSON response from {url} (status {status}): {body[:200]}") from exc
    if status >= 400:
        error = result.get("error", status)
        description = result.get("error_description", body[:200])
        err = ValueError(f"Kimi OAuth {error}: {description}")
        err.kimi_error = error  # type: ignore[attr-defined]
        err.kimi_status = status  # type: ignore[attr-defined]
        raise err
    return result


def kimi_start_device_auth() -> dict[str, Any]:
    """Begin the RFC 8628 device flow for a Kimi For Coding subscription."""
    return _kimi_post_form(KIMI_OAUTH_DEVICE_AUTH_URL, {"client_id": KIMI_OAUTH_CLIENT_ID})


def kimi_poll_device_token(
    device: dict[str, Any],
    *,
    sleep_fn: Any = time.sleep,
    now_fn: Any = time.monotonic,
) -> dict[str, Any]:
    """Poll the token endpoint until the user approves the device code."""
    interval = max(1.0, float(device.get("interval", 5) or 5))
    deadline = now_fn() + float(device.get("expires_in", 600) or 600)
    while now_fn() < deadline:
        sleep_fn(interval)
        try:
            return _kimi_post_form(
                KIMI_OAUTH_TOKEN_URL,
                {
                    "client_id": KIMI_OAUTH_CLIENT_ID,
                    "device_code": str(device.get("device_code", "")),
                    "grant_type": KIMI_DEVICE_GRANT,
                },
            )
        except ValueError as exc:
            code = getattr(exc, "kimi_error", None)
            if code == "authorization_pending":
                continue
            if code == "slow_down":
                interval += 5.0
                continue
            if code == "expired_token":
                raise ValueError("Kimi OAuth: device code expired — run `oh auth kimi-login` again.") from exc
            raise
    raise ValueError("Kimi OAuth: device code expired before it was approved — run `oh auth kimi-login` again.")


def refresh_kimi_oauth_credential(refresh_token: str) -> dict[str, Any]:
    """Refresh a Kimi For Coding access token via its rotating refresh_token.

    Returns the new ``access_token``, the (rotated) ``refresh_token``, and
    ``expires_at_ms``. Does not write files. Transient 429/5xx responses are
    retried with backoff; ``invalid_grant`` tells the user to re-login.
    """
    if not refresh_token:
        raise ValueError("refresh_token is required")
    last_error: Exception | None = None
    for attempt in range(_KIMI_REFRESH_MAX_RETRIES):
        try:
            result = _kimi_post_form(
                KIMI_OAUTH_TOKEN_URL,
                {
                    "client_id": KIMI_OAUTH_CLIENT_ID,
                    "refresh_token": refresh_token,
                    "grant_type": KIMI_REFRESH_GRANT,
                },
            )
        except ValueError as exc:
            status = getattr(exc, "kimi_status", None)
            if getattr(exc, "kimi_error", None) == "invalid_grant":
                raise ValueError(
                    "Kimi OAuth refresh token is invalid or expired. "
                    "Run `oh auth kimi-login` to re-authenticate."
                ) from exc
            last_error = exc
            if status is not None and status not in _KIMI_REFRESH_RETRYABLE_STATUSES:
                raise
            if attempt < _KIMI_REFRESH_MAX_RETRIES - 1:
                time.sleep(2**attempt)
            continue
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < _KIMI_REFRESH_MAX_RETRIES - 1:
                time.sleep(2**attempt)
            continue
        access_token = str(result.get("access_token", "") or "")
        if not access_token:
            raise ValueError("Kimi OAuth refresh response missing access_token")
        expires_at_ms = int(time.time() * 1000) + int(result.get("expires_in", 3600) or 3600) * 1000
        return {
            "access_token": access_token,
            "refresh_token": str(result.get("refresh_token", refresh_token) or refresh_token),
            "expires_at_ms": expires_at_ms,
        }
    assert last_error is not None
    raise last_error


def store_kimi_oauth_tokens(source_path: Path, tokens: dict[str, Any]) -> None:
    """Persist kimi OAuth tokens (login or rotated refresh) to the auth file."""
    source_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write_text(source_path, json.dumps(tokens, indent=2))
    os.chmod(source_path, 0o600)


def _write_kimi_auth_json(source_path: Path, payload: dict[str, Any], refreshed: dict[str, Any]) -> None:
    """Persist refreshed kimi tokens back to the auth file (preserving extras)."""
    updated = dict(payload)
    updated["access_token"] = str(refreshed["access_token"])
    if refreshed.get("refresh_token"):
        updated["refresh_token"] = str(refreshed["refresh_token"])
    if refreshed.get("expires_at_ms"):
        updated["expires_at_ms"] = int(refreshed["expires_at_ms"])
    store_kimi_oauth_tokens(source_path, updated)


def _kimi_should_refresh(credential: ExternalAuthCredential, *, now_ms: int | None = None) -> bool:
    """True when the kimi access token is expired or within the refresh skew."""
    if credential.expires_at_ms is None:
        return False
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    return credential.expires_at_ms - now_ms <= _KIMI_REFRESH_SKEW_MS


def _load_kimi_credential(
    payload: dict[str, Any],
    source_path: Path,
    binding: ExternalAuthBinding,
    *,
    refresh_if_needed: bool = False,
) -> ExternalAuthCredential:
    access_token = str(payload.get("access_token", "") or "")
    refresh_token = str(payload.get("refresh_token", "") or "")
    if not access_token:
        raise ValueError("Kimi auth source does not contain an access token.")
    expires_at_raw = payload.get("expires_at_ms")
    expires_at_ms = int(expires_at_raw) if isinstance(expires_at_raw, (int, float)) else None

    def _build(token: str, rtoken: str, expiry: int | None) -> ExternalAuthCredential:
        return ExternalAuthCredential(
            provider=KIMI_PROVIDER,
            value=token,
            auth_kind="oauth",
            source_path=source_path,
            managed_by=binding.managed_by,
            profile_label=binding.profile_label,
            refresh_token=rtoken,
            expires_at_ms=expiry,
        )

    credential = _build(access_token, refresh_token, expires_at_ms)
    # Self-heal an expired Kimi For Coding access token the same way the codex
    # loader does: the refresh_token rotates, so serialize check→refresh→persist
    # in-process with a double-checked re-read inside the lock.
    if refresh_if_needed and _kimi_should_refresh(credential):
        if not refresh_token:
            raise ValueError(
                f"Kimi credentials at {source_path} are expired and cannot be refreshed "
                "(no refresh_token). Run `oh auth kimi-login` to re-authenticate."
            )
        with _KIMI_REFRESH_LOCK:
            try:
                cur_payload = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cur_payload = payload
            cur_access = str(cur_payload.get("access_token", "") or "")
            cur_refresh = str(cur_payload.get("refresh_token", "") or refresh_token)
            cur_expiry_raw = cur_payload.get("expires_at_ms")
            cur_expiry = int(cur_expiry_raw) if isinstance(cur_expiry_raw, (int, float)) else None
            if cur_access:
                cur_cred = _build(cur_access, cur_refresh, cur_expiry)
                if not _kimi_should_refresh(cur_cred):
                    return cur_cred  # another task already refreshed
            refreshed = refresh_kimi_oauth_credential(cur_refresh)
            _write_kimi_auth_json(source_path, cur_payload, refreshed)
            credential = _build(
                str(refreshed["access_token"]),
                str(refreshed.get("refresh_token", cur_refresh) or cur_refresh),
                int(refreshed["expires_at_ms"]),
            )
    return credential


def _load_claude_credential(
    payload: dict[str, Any],
    source_path: Path,
    binding: ExternalAuthBinding,
    *,
    refresh_if_needed: bool,
    keychain_service: str | None = None,
    keychain_account: str | None = None,
) -> ExternalAuthCredential:
    claude_oauth = payload.get("claudeAiOauth")
    if not isinstance(claude_oauth, dict):
        raise ValueError("Claude auth source does not contain claudeAiOauth.")

    access_token = str(claude_oauth.get("accessToken", "") or "")
    refresh_token = str(claude_oauth.get("refreshToken", "") or "")
    expires_at_raw = claude_oauth.get("expiresAt")
    if not access_token:
        raise ValueError("Claude auth source does not contain an access token.")

    expires_at_ms = _coerce_int(expires_at_raw)
    credential = ExternalAuthCredential(
        provider=CLAUDE_PROVIDER,
        value=access_token,
        auth_kind="auth_token",
        source_path=source_path,
        managed_by=binding.managed_by,
        profile_label=keychain_account or binding.profile_label,
        refresh_token=refresh_token,
        expires_at_ms=expires_at_ms,
    )
    if refresh_if_needed and is_credential_expired(credential):
        if not refresh_token:
            raise ValueError(
                f"Claude credentials at {source_path} are expired and cannot be refreshed."
            )
        refreshed = refresh_claude_oauth_credential(refresh_token)
        if binding.source_kind == "claude_credentials_keychain":
            _write_claude_credentials_to_keychain(
                service=keychain_service or CLAUDE_KEYCHAIN_SERVICE,
                account=keychain_account or os.environ.get("USER", ""),
                payload=payload,
                access_token=str(refreshed["access_token"]),
                refresh_token=str(refreshed["refresh_token"]),
                expires_at_ms=int(refreshed["expires_at_ms"]),
            )
        else:
            write_claude_credentials(
                source_path,
                access_token=str(refreshed["access_token"]),
                refresh_token=str(refreshed["refresh_token"]),
                expires_at_ms=int(refreshed["expires_at_ms"]),
            )
        credential = ExternalAuthCredential(
            provider=CLAUDE_PROVIDER,
            value=str(refreshed["access_token"]),
            auth_kind="auth_token",
            source_path=source_path,
            managed_by=binding.managed_by,
            profile_label=keychain_account or binding.profile_label,
            refresh_token=str(refreshed["refresh_token"]),
            expires_at_ms=int(refreshed["expires_at_ms"]),
        )
    return credential


def _load_claude_payload(
    binding: ExternalAuthBinding,
) -> tuple[dict[str, Any], Path, str | None, str | None]:
    if binding.source_kind == "claude_credentials_keychain":
        return _read_claude_credentials_from_keychain(binding)

    source_path = Path(binding.source_path).expanduser()
    if not source_path.exists():
        raise ValueError(f"External auth source not found: {source_path}")
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in external auth source: {source_path}") from exc
    return payload, source_path, None, None


def _read_claude_credentials_from_keychain(
    binding: ExternalAuthBinding,
) -> tuple[dict[str, Any], Path, str, str | None]:
    service = binding.source_path.removeprefix(_KEYCHAIN_BINDING_PREFIX).strip() or CLAUDE_KEYCHAIN_SERVICE
    try:
        raw_payload = subprocess.check_output(
            ["security", "find-generic-password", "-w", "-s", service],
            text=True,
        )
        metadata = subprocess.check_output(
            ["security", "find-generic-password", "-s", service],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Claude Keychain credential not found for service: {service}") from exc

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in Claude Keychain secret for service: {service}") from exc

    keychain_path = _extract_keychain_path(metadata) or (Path.home() / "Library/Keychains/login.keychain-db")
    account = _extract_keychain_attr(metadata, "acct")
    return payload, keychain_path, service, account


def _extract_keychain_path(metadata: str) -> Path | None:
    match = re.search(r'^keychain:\s+"([^"]+)"$', metadata, re.MULTILINE)
    if not match:
        return None
    return Path(match.group(1))


def _extract_keychain_attr(metadata: str, attr_name: str) -> str | None:
    match = re.search(rf'"{re.escape(attr_name)}"<blob>="([^"]*)"', metadata)
    if not match:
        return None
    return match.group(1)


def describe_external_binding(binding: ExternalAuthBinding) -> ExternalAuthState:
    """Return a human-readable state for an external auth binding."""
    source_path = Path(binding.source_path).expanduser()
    if binding.source_kind != "claude_credentials_keychain" and not source_path.exists():
        return ExternalAuthState(
            configured=False,
            state="missing",
            source="missing",
            detail=f"external auth source not found: {source_path}",
        )
    try:
        credential = load_external_credential(binding, refresh_if_needed=False)
    except ValueError as exc:
        detail = str(exc)
        if "not found" in detail.lower():
            return ExternalAuthState(
                configured=False,
                state="missing",
                source="missing",
                detail=detail,
            )
        return ExternalAuthState(
            configured=False,
            state="invalid",
            source="external",
            detail=detail,
        )
    resolved_source = credential.source_path
    if binding.provider in {CLAUDE_PROVIDER, KIMI_PROVIDER} and is_credential_expired(credential):
        if credential.refresh_token:
            return ExternalAuthState(
                configured=True,
                state="refreshable",
                source="external",
                detail=f"expired token can be refreshed from {resolved_source}",
            )
        return ExternalAuthState(
            configured=False,
            state="expired",
            source="external",
            detail=f"expired token at {resolved_source}",
        )
    return ExternalAuthState(
        configured=True,
        state="configured",
        source="external",
        detail=str(resolved_source),
    )


def is_credential_expired(credential: ExternalAuthCredential, *, now_ms: int | None = None) -> bool:
    """Return True when the external credential is definitely expired."""
    if credential.expires_at_ms is None:
        return False
    if now_ms is None:
        import time

        now_ms = int(time.time() * 1000)
    return credential.expires_at_ms <= now_ms


def get_claude_code_version() -> str:
    """Return the locally installed Claude Code version or a fallback."""
    global _claude_code_version_cache
    if _claude_code_version_cache is not None:
        return _claude_code_version_cache
    for command in ("claude", "claude-code"):
        try:
            result = subprocess.run(
                [command, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            continue
        version = (result.stdout or "").strip().split(" ", 1)[0]
        if result.returncode == 0 and version and version[0].isdigit():
            _claude_code_version_cache = version
            return version
    _claude_code_version_cache = CLAUDE_CODE_VERSION_FALLBACK
    return _claude_code_version_cache


def get_claude_code_session_id() -> str:
    """Return a stable Claude Code-style session identifier for this process."""
    global _claude_code_session_id
    if _claude_code_session_id is None:
        _claude_code_session_id = str(uuid.uuid4())
    return _claude_code_session_id


def claude_oauth_betas() -> list[str]:
    """Return Claude OAuth betas as a list for SDK beta endpoints."""
    return list(CLAUDE_COMMON_BETAS + CLAUDE_OAUTH_ONLY_BETAS)


def claude_attribution_header() -> str:
    """Return the Claude Code billing attribution prefix used in system prompts."""
    version = get_claude_code_version()
    return (
        "x-anthropic-billing-header: "
        f"cc_version={version}; cc_entrypoint=cli;"
    )


def claude_oauth_headers() -> dict[str, str]:
    """Return Claude Code-style headers for subscription OAuth traffic."""
    all_betas = ",".join(claude_oauth_betas())
    return {
        "anthropic-beta": all_betas,
        "user-agent": f"claude-cli/{get_claude_code_version()} (external, cli)",
        "x-app": "cli",
        "X-Claude-Code-Session-Id": get_claude_code_session_id(),
    }


def refresh_claude_oauth_credential(
    refresh_token: str,
    *,
    scopes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Refresh a Claude OAuth token without mutating local files."""
    if not refresh_token:
        raise ValueError("refresh_token is required")

    requested_scopes = list(scopes or CLAUDE_AI_OAUTH_SCOPES)
    payload = json.dumps(
        {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": CLAUDE_OAUTH_CLIENT_ID,
            "scope": " ".join(requested_scopes),
        }
    ).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"claude-cli/{get_claude_code_version()} (external, cli)",
    }
    last_error: Exception | None = None
    for endpoint in CLAUDE_OAUTH_TOKEN_ENDPOINTS:
        request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                body = ""
            if "invalid_grant" in body:
                last_error = ValueError(
                    "Claude OAuth refresh token is invalid or expired. "
                    "Run `claude auth login` to refresh the official Claude CLI "
                    "credentials, then run `oh auth claude-login` again."
                )
                continue
            detail = f"{exc.code} {exc.reason}"
            if body:
                detail = f"{detail}: {body}"
            last_error = ValueError(f"Claude OAuth refresh failed at {endpoint}: {detail}")
            continue
        except Exception as exc:
            last_error = exc
            continue
        access_token = str(result.get("access_token", "") or "")
        if not access_token:
            raise ValueError("Claude OAuth refresh response missing access_token")
        next_refresh = str(result.get("refresh_token", refresh_token) or refresh_token)
        expires_in = int(result.get("expires_in", 3600) or 3600)
        return {
            "access_token": access_token,
            "refresh_token": next_refresh,
            "expires_at_ms": int(time.time() * 1000) + expires_in * 1000,
            "scopes": result.get("scope"),
        }
    if last_error is not None:
        raise ValueError(f"Claude OAuth refresh failed: {last_error}") from last_error
    raise ValueError("Claude OAuth refresh failed")


def write_claude_credentials(
    source_path: Path,
    *,
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
) -> None:
    """Write refreshed Claude credentials back to the upstream credentials file."""
    existing: dict[str, Any] = {}
    if source_path.exists():
        try:
            existing = json.loads(source_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    existing["claudeAiOauth"] = _merge_claude_oauth_payload(
        existing.get("claudeAiOauth"),
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_ms=expires_at_ms,
    )
    atomic_write_text(
        source_path,
        json.dumps(existing, indent=2) + "\n",
        mode=0o600,
    )


def _write_claude_credentials_to_keychain(
    *,
    service: str,
    account: str,
    payload: dict[str, Any],
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
) -> None:
    next_payload = dict(payload)
    next_payload["claudeAiOauth"] = _merge_claude_oauth_payload(
        payload.get("claudeAiOauth"),
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at_ms=expires_at_ms,
    )
    subprocess.run(
        [
            "security",
            "add-generic-password",
            "-U",
            "-s",
            service,
            "-a",
            account,
            "-w",
            json.dumps(next_payload, separators=(",", ":")),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _merge_claude_oauth_payload(
    previous: Any,
    *,
    access_token: str,
    refresh_token: str,
    expires_at_ms: int,
) -> dict[str, Any]:
    next_oauth: dict[str, Any] = {
        "accessToken": access_token,
        "refreshToken": refresh_token,
        "expiresAt": expires_at_ms,
    }
    if isinstance(previous, dict):
        for key in ("scopes", "rateLimitTier", "subscriptionType"):
            if key in previous:
                next_oauth[key] = previous[key]
    return next_oauth


def is_third_party_anthropic_endpoint(base_url: str | None) -> bool:
    """Return True for non-Anthropic endpoints using Anthropic-compatible APIs."""
    if not base_url:
        return False
    normalized = base_url.rstrip("/").lower()
    return "anthropic.com" not in normalized and "claude.com" not in normalized


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.isdigit():
            return int(trimmed)
    return None


def _decode_jwt_expiry(token: str) -> int | None:
    exp = _decode_json_web_token_claim(token, ["exp"])
    if exp is None:
        return None
    if isinstance(exp, int):
        return exp * 1000
    if isinstance(exp, float):
        return int(exp * 1000)
    if isinstance(exp, str) and exp.strip().isdigit():
        return int(exp.strip()) * 1000
    return None


def _decode_json_web_token_claim(token: str, path: list[str]) -> Any | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        encoded = parts[1]
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:
        return None

    current: Any = payload
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current

"""OAuth refresh-token support for HTTP MCP servers (opt-in per server).

When an :class:`~openharness.mcp.types.McpHttpServerConfig` carries an ``oauth``
block, :func:`ensure_bearer` returns a valid access token, transparently doing a
``refresh_token`` grant when the cached token is missing or near expiry. The
rotating secrets live in ``oauth.token_file`` (JSON) so a rotated refresh token
is persisted across runs without rewriting settings.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import urllib.request

from openharness.mcp.types import McpOAuthConfig

_EXPIRY_SKEW_S = 60

# Refresh tokens rotate: two concurrent refreshes would consume the same token
# and break the chain (the loser's rotated token is lost). Serialize refreshes.
_refresh_lock = threading.Lock()


def _read_store(path: str) -> dict:
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        return {}
    with open(expanded, encoding="utf-8") as fh:
        return json.load(fh)


def _write_store(path: str, data: dict) -> None:
    expanded = os.path.expanduser(path)
    tmp = f"{expanded}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, expanded)


def _refresh(oauth: McpOAuthConfig, refresh_token: str) -> dict:
    form = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": oauth.client_id,
    }
    if oauth.client_secret:
        form["client_secret"] = oauth.client_secret
    if oauth.resource:
        form["resource"] = oauth.resource
    # NOTE: deliberately omit `scope` on the refresh grant. RFC 6749 §6 makes it
    # optional and forbids broadening; strict auth servers (e.g. auth.worfalomey.cc)
    # return 400 when `scope` is present on a refresh, which silently kills the
    # rotating refresh chain. `oauth.scope` is still used at authorization time
    # (outside this client). Re-add here only if a server is found that requires it.
    req = urllib.request.Request(
        oauth.token_url,
        data=urllib.parse.urlencode(form).encode(),
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def ensure_bearer(oauth, *, now=None, force=False, stale_token=None) -> str:
    """Return a valid access token, refreshing (and persisting rotation) if needed.

    Safe to call on every request (see ``_OAuthBearerAuth`` in ``client.py``):
    when the cached token is still valid it is a cheap file read. Tokens near
    expiry and forced refreshes trigger a lock-serialized refresh.
    """
    now = time.time() if now is None else now
    if not force:
        store = _read_store(oauth.token_file)
        access = store.get("access_token")
        expires_at = store.get("expires_at") or 0
        if access and expires_at - now > _EXPIRY_SKEW_S:
            return access
    with _refresh_lock:
        # Re-read inside the lock: another caller may have refreshed while we waited.
        store = _read_store(oauth.token_file)
        access = store.get("access_token")
        expires_at = store.get("expires_at") or 0
        valid = access and expires_at - now > _EXPIRY_SKEW_S
        if (not force and valid) or (
            force and stale_token is not None and access != stale_token and valid
        ):
            return access
        refresh_token = store.get("refresh_token")
        if not refresh_token:
            raise ValueError(
                f"No refresh_token in {oauth.token_file}; re-authenticate the MCP server."
            )
        tok = _refresh(oauth, refresh_token)
        access = tok["access_token"]
        store["access_token"] = access
        store["expires_at"] = now + int(tok.get("expires_in", 3600))
        if tok.get("refresh_token"):  # rotation
            store["refresh_token"] = tok["refresh_token"]
        _write_store(oauth.token_file, store)
        return access

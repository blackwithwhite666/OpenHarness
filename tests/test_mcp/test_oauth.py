from __future__ import annotations

import json
import time

import pytest

from openharness.mcp import oauth as oauth_mod
from openharness.mcp.types import McpOAuthConfig


def _cfg(token_file) -> McpOAuthConfig:
    return McpOAuthConfig(
        token_url="https://auth.example/token",
        client_id="cid",
        client_secret="sec",
        token_file=str(token_file),
        resource="https://mcp.example/mcp",
        scope="user",
    )


def test_uses_cached_token_without_network(tmp_path, monkeypatch):
    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({"access_token": "A1", "expires_at": time.time() + 3600, "refresh_token": "R1"}))

    def _boom(*a, **k):  # network must not be called
        raise AssertionError("refresh should not run for a valid cached token")

    monkeypatch.setattr(oauth_mod, "_refresh", _boom)
    assert oauth_mod.ensure_bearer(_cfg(tf)) == "A1"


def test_refreshes_expired_and_persists_rotation(tmp_path, monkeypatch):
    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({"access_token": "OLD", "expires_at": 0, "refresh_token": "R1"}))
    seen = {}

    def fake_refresh(oauth, refresh_token):
        seen["rt"] = refresh_token
        return {"access_token": "NEW", "expires_in": 3600, "refresh_token": "R2"}

    monkeypatch.setattr(oauth_mod, "_refresh", fake_refresh)
    assert oauth_mod.ensure_bearer(_cfg(tf)) == "NEW"
    assert seen["rt"] == "R1"
    saved = json.loads(tf.read_text())
    assert saved["access_token"] == "NEW"
    assert saved["refresh_token"] == "R2"  # rotation persisted
    assert saved["expires_at"] > time.time()


def test_near_expiry_triggers_refresh(tmp_path, monkeypatch):
    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({"access_token": "OLD", "expires_at": time.time() + 10, "refresh_token": "R1"}))
    monkeypatch.setattr(oauth_mod, "_refresh", lambda o, rt: {"access_token": "NEW", "expires_in": 3600})
    assert oauth_mod.ensure_bearer(_cfg(tf)) == "NEW"


def test_missing_refresh_token_raises(tmp_path):
    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({}))
    with pytest.raises(ValueError):
        oauth_mod.ensure_bearer(_cfg(tf))

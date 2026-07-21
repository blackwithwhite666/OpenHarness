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


def test_ensure_bearer_force_refreshes_even_when_token_valid(tmp_path, monkeypatch):
    tf = tmp_path / "tok.json"
    tf.write_text(
        json.dumps(
            {"access_token": "OLD", "expires_at": time.time() + 3600, "refresh_token": "R1"}
        )
    )
    calls = []

    def fake_refresh(oauth, refresh_token):
        calls.append(refresh_token)
        return {"access_token": "NEW", "expires_in": 3600}

    monkeypatch.setattr(oauth_mod, "_refresh", fake_refresh)
    assert oauth_mod.ensure_bearer(_cfg(tf), force=True) == "NEW"
    assert calls == ["R1"]


def test_ensure_bearer_force_dedups_via_stale_token(tmp_path, monkeypatch):
    tf = tmp_path / "tok.json"
    tf.write_text(
        json.dumps(
            {"access_token": "NEW", "expires_at": time.time() + 3600, "refresh_token": "R2"}
        )
    )

    def _boom(*a, **k):
        raise AssertionError("refresh should not run after another caller rotated the token")

    monkeypatch.setattr(oauth_mod, "_refresh", _boom)
    assert oauth_mod.ensure_bearer(_cfg(tf), force=True, stale_token="OLD") == "NEW"


def test_refresh_omits_scope_but_keeps_resource(tmp_path, monkeypatch):
    """`scope` must NOT be sent on the refresh grant (RFC 6749 §6 optional; strict
    servers 400 on it, silently killing the rotating chain). `resource` is kept."""
    import urllib.parse

    captured: dict[str, str] = {}

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"access_token": "NEW", "expires_in": 3600}).encode()

    def fake_urlopen(req, timeout=None):
        captured["data"] = req.data.decode()
        return _FakeResp()

    monkeypatch.setattr(oauth_mod.urllib.request, "urlopen", fake_urlopen)
    out = oauth_mod._refresh(_cfg(tmp_path / "tok.json"), "R1")  # cfg has scope="user"
    assert out["access_token"] == "NEW"
    form = dict(urllib.parse.parse_qsl(captured["data"]))
    assert "scope" not in form  # the bug being fixed
    assert form["resource"] == "https://mcp.example/mcp"
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "R1"
    assert form["client_id"] == "cid"


def test_missing_refresh_token_raises(tmp_path):
    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({}))
    with pytest.raises(ValueError):
        oauth_mod.ensure_bearer(_cfg(tf))


def test_concurrent_refresh_runs_once(tmp_path, monkeypatch):
    """Rotating refresh tokens: concurrent callers must trigger only one refresh."""
    import threading

    tf = tmp_path / "tok.json"
    tf.write_text(json.dumps({"access_token": "OLD", "expires_at": 0, "refresh_token": "R1"}))
    calls: list[str] = []
    start = threading.Barrier(2)

    def fake_refresh(oauth, refresh_token):
        calls.append(refresh_token)
        time.sleep(0.05)
        return {"access_token": "NEW", "expires_in": 3600, "refresh_token": "R2"}

    monkeypatch.setattr(oauth_mod, "_refresh", fake_refresh)
    results: dict[int, str] = {}

    def worker(i: int) -> None:
        start.wait()
        results[i] = oauth_mod.ensure_bearer(_cfg(tf))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results[0] == results[1] == "NEW"
    assert len(calls) == 1  # lock + double-check collapses the duplicate refresh


def test_oauth_bearer_auth_retries_once_on_401(tmp_path, monkeypatch):
    from openharness.mcp.client import _OAuthBearerAuth

    cfg = _cfg(tmp_path / "tok.json")
    auth = _OAuthBearerAuth(cfg)
    calls = []

    def fake_ensure_bearer(oauth, **kwargs):
        calls.append(kwargs)
        return "NEW" if kwargs.get("force") else "OLD"

    monkeypatch.setattr(oauth_mod, "ensure_bearer", fake_ensure_bearer)

    class _Request:
        def __init__(self):
            self.headers = {}

    class _Response:
        def __init__(self, status_code):
            self.status_code = status_code

    request = _Request()
    flow = auth.sync_auth_flow(request)
    first = next(flow)
    assert first.headers[cfg.header] == "Bearer OLD"
    second = flow.send(_Response(401))
    assert second is request
    assert second.headers[cfg.header] == "Bearer NEW"
    assert calls == [{}, {"force": True, "stale_token": "OLD"}]
    with pytest.raises(StopIteration):
        flow.send(_Response(401))

    calls.clear()
    request = _Request()
    flow = auth.sync_auth_flow(request)
    next(flow)
    with pytest.raises(StopIteration):
        flow.send(_Response(200))
    assert calls == [{}]


@pytest.mark.asyncio
async def test_oauth_bearer_auth_injects_token_per_request(tmp_path):
    """_OAuthBearerAuth sets a fresh bearer on each request via ensure_bearer."""
    import httpx

    from openharness.mcp.client import _OAuthBearerAuth

    tf = tmp_path / "tok.json"
    tf.write_text(
        json.dumps(
            {"access_token": "TKN", "expires_at": time.time() + 3600, "refresh_token": "R1"}
        )
    )
    cfg = _cfg(tf)
    auth = _OAuthBearerAuth(cfg)
    request = httpx.Request("POST", "https://mcp.example/mcp")
    flow = auth.async_auth_flow(request)
    sent = await flow.__anext__()
    assert sent.headers[cfg.header] == "Bearer TKN"
    await flow.aclose()

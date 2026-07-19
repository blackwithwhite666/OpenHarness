"""Tests for consented family onboarding and per-tenant lifecycle tools."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from ohmo.gateway.memory_gate import resolve_memory_scope
from ohmo.gateway.models import GatewayConfig
from ohmo.gateway.turn_context import TurnContext
from ohmo.memory_audit import get_memory_audit_counters, reset_memory_audit_counters
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_service.honcho_client import HonchoClient
from ohmo.tools import onboard_family_member as onboarding_tool
from ohmo.tools.onboard_family_member import onboard_family_member
from ohmo.tools.tenant_lifecycle import delete_tenant, export_tenant


def test_consent_record_round_trip(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    catalog.ensure_tenant("marina", "private")

    assert catalog.has_consent("marina") is False
    catalog.record_consent("marina", "200", "Confirmed directly.")
    assert catalog.has_consent("marina") is True

    exported = catalog.export_tenant("marina")
    assert exported["consent"] == [
        {
            "tenant_id": "marina",
            "principal": "200",
            "consented": 1,
            "consented_at": exported["consent"][0]["consented_at"],
            "note": "Confirmed directly.",
        }
    ]


async def test_onboarding_refuses_before_mutation_without_consent(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")

    with pytest.raises(PermissionError, match="--i-have-consent"):
        await onboard_family_member(
            catalog,
            tenant_id="marina",
            person_peer="marina",
            principal_ids=("200",),
            workspace="ohmo-marina",
            base_url="https://honcho.invalid",
            admin_jwt="admin",
            i_have_consent=False,
        )

    assert catalog.export_tenant("marina")["tenant"] is None
    assert catalog.has_consent("marina") is False


def test_onboarding_command_prints_ready_config_and_records_consent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_onboard_tenant(**kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "base_url": "https://honcho.example",
            "admin_jwt": "admin-secret",
            "workspace": "family-marina",
            "person_peer": "marina-peer",
            "ttl": onboarding_tool._DEFAULT_RUNTIME_KEY_TTL,
        }
        return SimpleNamespace(
            workspace="family-marina",
            jwt="scoped-runtime-key",
            observed_peer="marina-peer",
        )

    monkeypatch.setattr(onboarding_tool, "onboard_tenant", fake_onboard_tenant)
    onboarding_tool.main(
        [
            "--tenant-id",
            "marina",
            "--person-peer",
            "marina-peer",
            "--principal-id",
            "200",
            "201",
            "--workspace",
            "family-marina",
            "--base-url",
            "https://honcho.example",
            "--admin-jwt",
            "admin-secret",
            "--catalog-workspace",
            str(tmp_path),
            "--i-have-consent",
        ]
    )

    assert json.loads(capsys.readouterr().out) == {
        "family_principals": {"200": "marina", "201": "marina"},
        "enabled_memory_tenants": ["marina"],
        "tenant_honcho": {
            "marina": {
                "workspace": "family-marina",
                "api_key": "scoped-runtime-key",
                "observed_peer": "marina-peer",
            }
        },
    }
    catalog = MemoryCatalog(tmp_path)
    assert catalog.has_consent("marina") is True
    assert catalog.export_tenant("marina")["tenant"]["kind"] == "private"
    assert [row["principal"] for row in catalog.export_tenant("marina")["consent"]] == [
        "200",
        "201",
    ]


def test_export_is_deterministic_and_contains_only_requested_tenant(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    catalog.ensure_tenant("marina", "private")
    catalog.ensure_tenant("zoya", "private")
    assert catalog.add("marina", "Marina two", "marina-content-two").ok
    assert catalog.add("marina", "Marina one", "marina-content-one").ok
    assert catalog.add("zoya", "Zoya only", "zoya-private-content").ok

    first = export_tenant(catalog, "marina")
    second = export_tenant(catalog, "marina")

    assert first == second
    assert [row["tenant_id"] for row in first["memories"]] == ["marina", "marina"]
    assert [row["slug"] for row in first["memories"]] == ["marina_one", "marina_two"]
    assert {row["content"] for row in first["memories"]} == {
        "marina-content-one",
        "marina-content-two",
    }
    assert "zoya-private-content" not in json.dumps(first)


def test_delete_is_scoped_owner_protected_and_idempotent(tmp_path: Path) -> None:
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")
    catalog.ensure_tenant("marina", "private")
    catalog.ensure_tenant("zoya", "private")
    catalog.record_consent("marina", "200")
    catalog.record_consent("zoya", "300")
    assert catalog.add("owner", "Owner note", "owner-content").ok
    assert catalog.add("marina", "Marina note", "marina-content").ok
    assert catalog.add("zoya", "Zoya note", "zoya-content").ok

    assert delete_tenant(catalog, "marina") is True
    assert delete_tenant(catalog, "marina") is False
    assert catalog.export_tenant("marina") == {
        "schema_version": 6,
        "tenant_id": "marina",
        "tenant": None,
        "consent": [],
        "memories": [],
    }
    assert catalog.get("owner", "owner_note").content == "owner-content"
    assert catalog.get("zoya", "zoya_note").content == "zoya-content"
    assert catalog.has_consent("zoya") is True
    with pytest.raises(ValueError, match="owner tenant cannot be deleted"):
        delete_tenant(catalog, "owner")


def test_denied_family_scope_emits_content_free_event_and_counter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reset_memory_audit_counters()
    fact_content = "Marina's private medical fact must never enter audit logs."
    cfg = GatewayConfig(
        owner_principals=("100",),
        family_principals={"200": "marina"},
        enabled_memory_tenants=("owner",),
    )
    turn = TurnContext(
        principal="200",
        is_owner=False,
        is_private=True,
        channel="telegram",
        chat_id="200",
        session_id="session-200",
    )

    with caplog.at_level(logging.INFO, logger="ohmo.memory_audit"):
        assert resolve_memory_scope(cfg, turn, principal_isolated=True) is None

    [record] = [record for record in caplog.records if hasattr(record, "memory_audit")]
    assert record.memory_audit == {
        "event": "memory_gate",
        "count": 1,
        "tenant_id": "marina",
        "requester_tenant": "marina",
        "outcome": "deny",
        "failing_conjuncts": ["tenant_memory_enabled"],
        "cross_scope_denial_count": 1,
    }
    assert get_memory_audit_counters()["cross_scope_denial:marina:deny"] == 1
    assert fact_content not in caplog.text
    assert "medical fact" not in json.dumps(record.memory_audit)


async def test_backend_audit_events_bind_tenant_and_workspace_without_content(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    reset_memory_audit_counters()
    fact_content = "never-log-this-memory-content"
    catalog = MemoryCatalog(db_path=tmp_path / "catalog.sqlite3")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    with caplog.at_level(logging.INFO, logger="ohmo.memory_audit"):
        catalog.add("owner", "Private title", fact_content)
        async with HonchoClient(
            "https://honcho.example",
            "scoped-key",
            "owner-workspace",
            transport=httpx.MockTransport(respond),
        ) as client:
            await client.create_conclusions([])

    events = [record.memory_audit for record in caplog.records if hasattr(record, "memory_audit")]
    assert {event["event"] for event in events} == {"catalog_op", "honcho_request"}
    assert any(
        event.get("tenant_id") == "owner" and event.get("operation") == "add"
        for event in events
    )
    assert any(
        event.get("workspace") == "owner-workspace"
        and event.get("operation") == "post"
        for event in events
    )
    assert fact_content not in caplog.text
    assert fact_content not in json.dumps(events)

"""Tests for MCP tool adapters — input model generation and argument serialization."""

import copy
import asyncio
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

from openharness.mcp.client import (
    McpServerNotConnectedError,
    McpToolCallResult,
    McpToolTimeoutError,
)
from openharness.mcp.types import McpResourceInfo, McpToolInfo
from openharness.tools.base import ToolExecutionContext
from openharness.tools.list_mcp_resources_tool import ListMcpResourcesTool
from openharness.tools.mcp_tool import (
    McpToolAdapter,
    WellnessLoginInjectingAdapter,
    _input_model_from_schema,
)
from openharness.tools.read_mcp_resource_tool import ReadMcpResourceTool
from openharness.untrusted import UNTRUSTED_BANNER


class _FakeMcpManager:
    def __init__(
        self,
        *,
        tool_output: str = "",
        resource_output: str = "",
        resources: list[McpResourceInfo] | None = None,
    ) -> None:
        self.tool_output = tool_output
        self.resource_output = resource_output
        self.resources = resources or []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        del server_name, tool_name, arguments
        return self.tool_output

    async def read_resource(self, server_name: str, uri: str) -> str:
        del server_name, uri
        return self.resource_output

    def list_resources(self) -> list[McpResourceInfo]:
        return self.resources


class _TypedFakeMcpManager:
    """Manager double exposing the typed call_tool_result path."""

    def __init__(
        self,
        *,
        outcome: McpToolCallResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.outcome = outcome
        self.error = error

    async def call_tool_result(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> McpToolCallResult:
        del server_name, tool_name, arguments
        if self.error is not None:
            raise self.error
        assert self.outcome is not None
        return self.outcome


def _demo_adapter(manager) -> McpToolAdapter:
    return McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="demo",
            name="hello",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )


@pytest.mark.asyncio
async def test_mcp_tool_adapter_typed_success_is_fenced_and_not_error():
    manager = _TypedFakeMcpManager(outcome=McpToolCallResult(output="server supplied output"))
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is False
    assert result.output == f"{UNTRUSTED_BANNER}\n\nserver supplied output"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_tool_declared_error_preserves_body():
    manager = _TypedFakeMcpManager(
        outcome=McpToolCallResult(
            output="interval must not exceed 31 days",
            is_error=True,
        )
    )
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert "interval must not exceed 31 days" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_adapter_tool_declared_error_with_empty_body_stays_error():
    manager = _TypedFakeMcpManager(outcome=McpToolCallResult(output="", is_error=True))
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert result.output == ""


@pytest.mark.asyncio
async def test_mcp_tool_adapter_timeout_remains_error_on_typed_path():
    manager = _TypedFakeMcpManager(
        error=McpToolTimeoutError("MCP server 'demo' tool 'hello' timed out after 1s")
    )
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert "timed out" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_adapter_disconnection_remains_error_on_typed_path():
    manager = _TypedFakeMcpManager(
        error=McpServerNotConnectedError("MCP server 'demo' is not connected: boom")
    )
    adapter = _demo_adapter(manager)

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is True
    assert "not connected" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_adapter_fences_nonempty_success_output():
    manager = _FakeMcpManager(tool_output="server supplied output")
    adapter = McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="demo",
            name="hello",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is False
    assert result.output == f"{UNTRUSTED_BANNER}\n\nserver supplied output"


@pytest.mark.asyncio
async def test_mcp_tool_adapter_does_not_fence_empty_output():
    manager = _FakeMcpManager(tool_output="")
    adapter = McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="demo",
            name="hello",
            description="test",
            input_schema={"type": "object", "properties": {}},
        ),
    )

    result = await adapter.execute(
        adapter.input_model(),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.output == ""
    assert UNTRUSTED_BANNER not in result.output


@pytest.mark.asyncio
async def test_read_mcp_resource_fences_success_output():
    tool = ReadMcpResourceTool(_FakeMcpManager(resource_output="resource body"))

    result = await tool.execute(
        tool.input_model(server="demo", uri="demo://readme"),
        ToolExecutionContext(cwd=Path(".")),
    )

    assert result.is_error is False
    assert result.output == f"{UNTRUSTED_BANNER}\n\nresource body"


@pytest.mark.asyncio
async def test_list_mcp_resources_fences_server_supplied_descriptions():
    manager = _FakeMcpManager(
        resources=[
            McpResourceInfo(
                server_name="demo",
                name="Readme",
                uri="demo://readme",
                description="server supplied description",
            )
        ]
    )
    tool = ListMcpResourcesTool(manager)

    result = await tool.execute(tool.input_model(), ToolExecutionContext(cwd=Path(".")))

    assert result.is_error is False
    assert result.output == (
        f"{UNTRUSTED_BANNER}\n\ndemo:demo://readme server supplied description"
    )


class _RecordingMcpManager:
    def __init__(self, tool_output: str = "wellness payload") -> None:
        self.tool_output = tool_output
        self.calls: list[tuple[str, str, dict, dict | None]] = []

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> str:
        self.calls.append((server_name, tool_name, arguments, None))
        return self.tool_output

    async def call_tool_result(
        self, server_name: str, tool_name: str, arguments: dict, *, meta=None
    ) -> McpToolCallResult:
        self.calls.append((server_name, tool_name, arguments, meta))
        return McpToolCallResult(output=self.tool_output)


def _set_synthetic_wellness_config(monkeypatch) -> None:
    for name, value in {
        "WELLNESS_DELEGATION_SIGNING_KEY": "synthetic-secret-" + "x" * 32,
        "WELLNESS_DELEGATION_KID": "test-key",
        "WELLNESS_DELEGATION_ISSUER": "oh-test",
        "WELLNESS_DELEGATION_AUDIENCE": "tg-test",
        "WELLNESS_DELEGATION_CLIENT_ID": "oh-client",
    }.items():
        monkeypatch.setenv(name, value)


def _admitted_context(telegram_id: str = "116870365") -> ToolExecutionContext:
    from openharness.mcp.wellness_delegation import TrustedWellnessActor

    return ToolExecutionContext(
        cwd=Path("."),
        metadata={"wellness_trusted_actor": TrustedWellnessActor(telegram_id)},
    )


def test_wellness_signer_config_matches_verifier_bounds(monkeypatch):
    from openharness.mcp.wellness_delegation import WellnessDelegationConfig

    _set_synthetic_wellness_config(monkeypatch)
    padded_secret = " " + "s" * 32 + " "
    monkeypatch.setenv("WELLNESS_DELEGATION_SIGNING_KEY", padded_secret)
    monkeypatch.setenv("WELLNESS_DELEGATION_KID", "k" * 64)
    config = WellnessDelegationConfig.from_env()
    assert config.kid == "k" * 64
    assert config.key == padded_secret.encode("utf-8")

    monkeypatch.setenv("WELLNESS_DELEGATION_KID", "k" * 65)
    with pytest.raises(ValueError):
        WellnessDelegationConfig.from_env()

    monkeypatch.setenv("WELLNESS_DELEGATION_KID", "test-key")
    monkeypatch.setenv("WELLNESS_DELEGATION_SIGNING_KEY", "s" * 513)
    with pytest.raises(ValueError):
        WellnessDelegationConfig.from_env()

    monkeypatch.setenv("WELLNESS_DELEGATION_SIGNING_KEY", "s" * 32 + "\x01")
    with pytest.raises(ValueError):
        WellnessDelegationConfig.from_env()

    monkeypatch.setenv("WELLNESS_DELEGATION_SIGNING_KEY", padded_secret)
    monkeypatch.setenv("WELLNESS_DELEGATION_ISSUER", "i" * 513)
    with pytest.raises(ValueError):
        WellnessDelegationConfig.from_env()


def _energy_fixture() -> str:
    return (Path(__file__).parents[1] / "fixtures" / "wellness_energy_days.json").read_text(
        encoding="utf-8"
    )


def test_energy_fixture_covers_positive_and_negative_balance_gates() -> None:
    payload = json.loads(_energy_fixture())
    resolved_correction, possible_replay, unresolved_key, legacy_synthetic = payload["energy_days"][
        :4
    ]
    complete, incomplete, basal_conflict = payload["energy_days"][4:]

    response_fields = {
        "device_id",
        "local_day",
        "timezone",
        "basal_sum",
        "basal_unit",
        "basal_points",
        "basal_minutes_with_samples",
        "day_minutes",
        "active_sum",
        "active_unit",
        "active_points",
        "next_day_basal_observed",
        "basal_conflicting_timestamps",
        "active_conflicting_timestamps",
        "snapshot_revision",
        "unresolved_key_count",
        "possible_replay_count",
        "legacy_synthetic_count",
    }
    assert all(set(day) == response_fields for day in payload["energy_days"])
    assert all(day["basal_unit"] == day["active_unit"] == "kJ" for day in payload["energy_days"])
    assert {day["snapshot_revision"] for day in payload["energy_days"]} == {5}

    assert payload["nutrition_status"] == "complete"
    assert complete["basal_minutes_with_samples"] == complete["day_minutes"]
    assert complete["active_points"] > 0
    assert complete["next_day_basal_observed"] is True
    assert complete["basal_conflicting_timestamps"] == 0
    assert complete["active_conflicting_timestamps"] == 0
    assert complete["basal_sum"] / 4.184 == pytest.approx(2000.0)
    assert complete["active_sum"] / 4.184 == pytest.approx(100.0)

    assert incomplete["basal_minutes_with_samples"] < incomplete["day_minutes"]
    assert incomplete["active_points"] == 0
    assert incomplete["next_day_basal_observed"] is True  # next fixture day has basal points
    assert incomplete["active_conflicting_timestamps"] == 1
    assert incomplete["unresolved_key_count"] > 0
    assert basal_conflict["basal_conflicting_timestamps"] == 1
    assert basal_conflict["active_conflicting_timestamps"] == 0
    assert basal_conflict["unresolved_key_count"] > 0

    assert resolved_correction["basal_conflicting_timestamps"] == 0
    assert resolved_correction["active_conflicting_timestamps"] == 0
    assert resolved_correction["unresolved_key_count"] == 0
    assert resolved_correction["legacy_synthetic_count"] == 0
    assert possible_replay["possible_replay_count"] > 0
    assert possible_replay["unresolved_key_count"] == 0
    assert possible_replay["legacy_synthetic_count"] == 0
    assert unresolved_key["unresolved_key_count"] > 0
    assert legacy_synthetic["legacy_synthetic_count"] > 0


def test_energy_balance_offline_cases_fail_closed_on_missing_or_uncertain_facts() -> None:
    """Exercise factual reportability cases without provider calls or meal data."""
    payload = json.loads(_energy_fixture())
    full_day = payload["energy_days"][4]
    frozen_now = datetime(2026, 9, 22, 21, 0, tzinfo=ZoneInfo("UTC"))

    def allowed(
        response: dict, nutrition_status: str = "complete", *, now_utc: datetime = frozen_now
    ) -> bool:
        day = response["energy_days"][0]
        return (
            nutrition_status == "complete"
            and full_day.keys() <= day.keys()
            and now_utc.astimezone(ZoneInfo(day["timezone"])).date().isoformat() > day["local_day"]
            and day["basal_minutes_with_samples"] == day["day_minutes"]
            and day["active_points"] > 0
            and day["next_day_basal_observed"] is True
            and day["basal_conflicting_timestamps"] == 0
            and day["active_conflicting_timestamps"] == 0
            and day["unresolved_key_count"] == 0
            and day["legacy_synthetic_count"] == 0
            and day["basal_unit"] in {"kJ", "kcal"}
            and day["active_unit"] in {"kJ", "kcal"}
        )

    def response_with(**changes: object) -> dict:
        day = dict(full_day)
        day.update(changes)
        return {"energy_days": [day]}

    assert allowed(response_with())  # full trusted day
    assert allowed(dict(energy_days=[payload["energy_days"][0]]))  # resolved correction
    assert allowed(dict(energy_days=[payload["energy_days"][1]]))  # A→B→A stays provisional
    revised = response_with(snapshot_revision=6, basal_sum=9000.0)
    assert revised["energy_days"][0]["basal_sum"] != full_day["basal_sum"]
    assert allowed(revised)  # a later resolved correction revises sums without veto
    assert not allowed(response_with(basal_minutes_with_samples=1439))  # incomplete basal
    assert not allowed(response_with(active_points=0))
    assert not allowed(response_with(next_day_basal_observed=False))
    assert not allowed(dict(energy_days=[payload["energy_days"][2]]))  # same-request unresolved
    assert not allowed(response_with(unresolved_key_count=1))  # uncertified legacy point
    assert not allowed(dict(energy_days=[payload["energy_days"][3]]))  # certified max_value
    assert not allowed(response_with(legacy_synthetic_count=1))
    assert not allowed(response_with(basal_unit="J"))
    assert not allowed(response_with(active_unit=None))
    assert not allowed(response_with(), "stale")
    before_midnight = datetime(2026, 9, 22, 20, 59, tzinfo=ZoneInfo("UTC"))
    assert not allowed(response_with(local_day="2026-09-22"), now_utc=before_midnight)
    assert allowed(response_with(local_day="2026-09-22"))  # same day becomes past at midnight
    assert not allowed(response_with(local_day="2026-09-23"))  # current local day
    for field in ("unresolved_key_count", "possible_replay_count", "legacy_synthetic_count"):
        old_api = dict(full_day)
        old_api.pop(field)
        assert not allowed({"energy_days": [old_api]})


def _wellness_delegate(manager: _RecordingMcpManager) -> McpToolAdapter:
    return McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="worfalomey",
            name="get_wellness_data",
            description="wellness",
            input_schema={
                "type": "object",
                "properties": {
                    "params": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "health_types": {"type": "array"},
                            "login": {"type": "string"},
                            "interval": {"type": "string"},
                            "start": {"type": "string"},
                            "end": {"type": "string"},
                            "include_health": {"type": "boolean"},
                        },
                    }
                },
            },
        ),
    )


def _wellness_ref_delegate(manager: _RecordingMcpManager) -> McpToolAdapter:
    return McpToolAdapter(
        manager,
        McpToolInfo(
            server_name="worfalomey",
            name="get_wellness_data",
            description="wellness",
            input_schema={
                "type": "object",
                "properties": {
                    "params": {"$ref": "#/$defs/WellnessParams"},
                },
                "required": ["params"],
                "$defs": {
                    "WellnessParams": {
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "health_types": {
                                "anyOf": [
                                    {"type": "array", "items": {"type": "string"}},
                                    {"type": "null"},
                                ],
                                "default": None,
                            },
                            "participant_id": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "default": None,
                            },
                            "login": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "interval": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "start": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                            "end": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "default": None,
                            },
                        },
                    }
                },
            },
        ),
    )


class TestWellnessLoginInjectingAdapter:
    def test_legacy_identity_fields_are_hidden_from_model_schema(self):
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(_RecordingMcpManager()))
        serialized = json.dumps(adapter.input_model.model_json_schema())
        for field in ("user_id", "health_types", "participant_id"):
            assert field not in serialized

    async def test_missing_turn_actor_fails_closed(self, monkeypatch):
        _set_synthetic_wellness_config(monkeypatch)
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        adapter.set_trusted_principal("116870365", trusted_login="must_not_be_used", owner_turn=True)
        result = await adapter.execute(adapter.input_model(params={"interval": "7d"}),
                                       ToolExecutionContext(cwd=Path(".")))
        assert result.is_error
        assert manager.calls == []

    async def test_admitted_actor_signs_final_args_without_contact_or_role_injection(self, monkeypatch):
        from openharness.mcp.wellness_delegation import META_KEY, TrustedWellnessActor
        _set_synthetic_wellness_config(monkeypatch)
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        context = ToolExecutionContext(cwd=Path("."),
            metadata={"wellness_trusted_actor": TrustedWellnessActor("116870365")})
        result = await adapter.execute(adapter.input_model(params={"login": "reader", "interval": "7d"}),
                                       context)
        assert result.is_error is False
        assert manager.calls[-1][2]["params"] == {"login": "reader", "interval": "7d"}
        token = manager.calls[-1][3][META_KEY]
        assert isinstance(token, str)
        assert token not in result.output
        header, claims, signature = token.split(".")
        import base64
        import hashlib
        import hmac
        decoded_header = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
        decoded = json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4)))
        assert decoded_header == {"alg": "HS256", "kid": "test-key", "typ": "JWT"}
        assert set(decoded) == {
            "schema_version", "iss", "aud", "azp", "sub", "iat", "exp",
            "tool", "body_sha256",
        }
        assert decoded["schema_version"] == 1
        assert decoded["iss"] == "oh-test" and decoded["aud"] == "tg-test"
        assert decoded["azp"] == "oh-client" and decoded["tool"] == "get_wellness_data"
        assert decoded["exp"] - decoded["iat"] == 60
        assert decoded["sub"] == "telegram:116870365"
        assert decoded["body_sha256"] == hashlib.sha256(
            json.dumps(manager.calls[-1][2], sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode()).hexdigest()
        signed = f"{header}.{claims}".encode("ascii")
        actual = base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
        assert hmac.compare_digest(actual, hmac.new(
            ("synthetic-secret-" + "x" * 32).encode(), signed, hashlib.sha256
        ).digest())

    async def test_two_admitted_readers_share_adapter_without_identity_crossing(self, monkeypatch):
        from openharness.mcp.wellness_delegation import META_KEY

        _set_synthetic_wellness_config(monkeypatch)

        class InterleavingManager(_RecordingMcpManager):
            async def call_tool_result(self, server_name, tool_name, arguments, *, meta=None):
                await asyncio.sleep(0)
                self.calls.append((server_name, tool_name, arguments, meta))
                return McpToolCallResult(output="{}")

        manager = InterleavingManager(tool_output="{}")
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        await asyncio.gather(
            adapter.execute(adapter.input_model(params={"login": "reader-a"}),
                            _admitted_context("101")),
            adapter.execute(adapter.input_model(params={"login": "reader-b"}),
                            _admitted_context("202")),
        )
        claims_by_login = {}
        import base64
        for _, _, body, meta in manager.calls:
            token = meta[META_KEY]
            encoded_claims = token.split(".")[1]
            claims = json.loads(base64.urlsafe_b64decode(
                encoded_claims + "=" * (-len(encoded_claims) % 4)
            ))
            claims_by_login[body["params"]["login"]] = claims["sub"]
        assert claims_by_login == {"reader-a": "telegram:101", "reader-b": "telegram:202"}

    async def test_rolling_interval_bounds_and_json_response_are_preserved(self, monkeypatch):
        _set_synthetic_wellness_config(monkeypatch)
        payload = Path(__file__).parents[1].joinpath(
            "fixtures", "wellness_energy_intervals.json"
        ).read_text(encoding="utf-8")
        manager = _RecordingMcpManager(tool_output=payload)
        adapter = WellnessLoginInjectingAdapter(_wellness_ref_delegate(manager))
        start = "2026-10-01T05:00:00+00:00"
        end = "2026-10-02T05:00:00+00:00"
        result = await adapter.execute(
            adapter.input_model(params={"start": start, "end": end}),
            _admitted_context(),
        )
        assert result.is_error is False
        returned = json.loads(result.output.split("\n\n", 1)[1])
        assert returned == json.loads(payload)
        assert returned["interval"] == {"start": start, "end": end}
        assert returned["linked_device_ids"] == ["watch-1"]
        assert returned["energy_snapshot_revision"] == 8
        assert returned["nutrition_unassigned_records"] == []
        interval = returned["energy_intervals"][0]
        day = returned["energy_days"][0]
        assert interval["start"] == start and interval["end"] == end
        assert interval["timezone"] == "Europe/Moscow"
        assert interval["device_id"] == day["device_id"]
        assert interval["snapshot_revision"] == day["snapshot_revision"] == 8
        assert interval["basal_sum"] != day["basal_sum"]
        assert interval["basal_minutes_with_samples"] < 24 * 60
        assert interval["possible_replay_count"] == 1
        assert interval["basal_sum"] / 4.184 == pytest.approx(1900.0956022944565)
        assert interval["active_sum"] / 4.184 == pytest.approx(90.0)
        assert manager.calls[-1][2]["params"] == {"start": start, "end": end}

    async def test_calendar_bounds_and_omitted_login_are_not_rewritten(self, monkeypatch):
        _set_synthetic_wellness_config(monkeypatch)
        manager = _RecordingMcpManager(tool_output=_energy_fixture())
        adapter = WellnessLoginInjectingAdapter(_wellness_ref_delegate(manager))
        start = "2026-10-01T00:00:00+03:00"
        end = "2026-10-02T00:00:00+03:00"
        result = await adapter.execute(
            adapter.input_model(params={"start": start, "end": end}),
            _admitted_context(),
        )
        assert result.is_error is False
        assert manager.calls[-1][2]["params"] == {"start": start, "end": end}
        assert json.loads(result.output.split("\n\n", 1)[1]) == json.loads(_energy_fixture())

    async def test_foreign_health_reason_and_empty_health_arrays_survive_json_formatter(self, monkeypatch):
        _set_synthetic_wellness_config(monkeypatch)
        payload = {
            "health_authorized": False,
            "health_authorization_reason": (
                "Nutrition-only permission. Health and energy are not available for this reader."
            ),
            "energy_days": [],
            "nutrition_status": "complete",
        }
        manager = _RecordingMcpManager(tool_output=json.dumps(payload))
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        result = await adapter.execute(
            adapter.input_model(params={"login": "reader"}), _admitted_context()
        )
        returned = json.loads(result.output.split("\n\n", 1)[1])
        assert returned == payload

    async def test_explicit_null_login_remains_distinct_from_omission(self, monkeypatch):
        from openharness.mcp.wellness_delegation import META_KEY

        _set_synthetic_wellness_config(monkeypatch)
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        await adapter.execute(
            adapter.input_model(params={"login": None}), _admitted_context()
        )
        await adapter.execute(adapter.input_model(params={}), _admitted_context())
        explicit, omitted = manager.calls
        assert explicit[2]["params"] == {"login": None}
        assert omitted[2]["params"] == {}
        assert explicit[3][META_KEY] != omitted[3][META_KEY]

    async def test_hidden_legacy_fields_are_stripped_after_model_dump(self, monkeypatch):
        _set_synthetic_wellness_config(monkeypatch)
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_ref_delegate(manager))

        class LegacyArguments(BaseModel):
            params: dict[str, object]

        result = await adapter.execute(
            LegacyArguments(params={
                "interval": "7d", "participant_id": 116870365,
                "user_id": "model-value", "health_types": ["weight"],
            }),
            _admitted_context(),
        )
        assert result.is_error is False
        assert manager.calls[-1][2] == {"params": {"interval": "7d"}}

    def test_ref_resolution_does_not_mutate_the_source_schema(self):
        delegate = _wellness_ref_delegate(_RecordingMcpManager())
        original = copy.deepcopy(delegate._tool_info.input_schema)
        adapter = WellnessLoginInjectingAdapter(delegate)
        schema = json.dumps(adapter.input_model.model_json_schema())
        assert "login" in schema
        for field in ("user_id", "participant_id", "health_types"):
            assert field not in schema
        assert delegate._tool_info.input_schema == original

    @pytest.mark.parametrize(
        "ref,definitions",
        [
            ("#/components/schemas/WellnessParams", {"WellnessParams": {"type": "object"}}),
            ("#/$defs/MissingParams", {}),
        ],
    )
    def test_rejects_unsupported_or_broken_params_ref(self, ref, definitions):
        delegate = _wellness_ref_delegate(_RecordingMcpManager())
        delegate._tool_info.input_schema["properties"]["params"] = {"$ref": ref}
        delegate._tool_info.input_schema["$defs"] = definitions
        with pytest.raises(ValueError, match="JSON Schema reference"):
            WellnessLoginInjectingAdapter(delegate)

    async def test_invalid_body_or_actor_fails_before_transport(self, monkeypatch):
        from openharness.mcp.wellness_delegation import TrustedWellnessActor

        _set_synthetic_wellness_config(monkeypatch)
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        bad_actor_context = ToolExecutionContext(
            cwd=Path("."), metadata={"wellness_trusted_actor": TrustedWellnessActor("001")}
        )
        result = await adapter.execute(
            adapter.input_model(params={"interval": "7d"}), bad_actor_context
        )
        assert result.is_error is True
        assert manager.calls == []

    async def test_missing_private_config_denies_before_transport(self, monkeypatch):
        from openharness.mcp.wellness_delegation import TrustedWellnessActor

        for name in (
            "WELLNESS_DELEGATION_SIGNING_KEY", "WELLNESS_DELEGATION_KID",
            "WELLNESS_DELEGATION_ISSUER", "WELLNESS_DELEGATION_AUDIENCE",
            "WELLNESS_DELEGATION_CLIENT_ID",
        ):
            monkeypatch.delenv(name, raising=False)
        manager = _RecordingMcpManager()
        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(manager))
        context = ToolExecutionContext(
            cwd=Path("."),
            metadata={"wellness_trusted_actor": TrustedWellnessActor("116870365")},
        )
        result = await adapter.execute(
            adapter.input_model(params={"interval": "7d"}), context
        )
        assert result.is_error is True
        assert manager.calls == []

    def test_canonical_body_rejects_python_coercions_and_nonfinite_values(self):
        from openharness.mcp.wellness_delegation import canonical_body

        for body in ({1: "value"}, {"value": (1, 2)}, {"value": float("nan")}):
            with pytest.raises(ValueError):
                canonical_body(body)

    async def test_transport_error_is_reported_without_token(self, monkeypatch):
        _set_synthetic_wellness_config(monkeypatch)

        class FailingManager(_RecordingMcpManager):
            async def call_tool_result(self, *args, **kwargs):
                raise McpServerNotConnectedError("synthetic transport failure")

        adapter = WellnessLoginInjectingAdapter(_wellness_delegate(FailingManager()))
        result = await adapter.execute(
            adapter.input_model(params={"interval": "7d"}), _admitted_context()
        )
        assert result.is_error is True
        assert "synthetic transport failure" in result.output
        assert "synthetic-secret" not in result.output

class TestInputModelFromSchema:
    """Verify _input_model_from_schema maps JSON Schema types correctly."""

    def test_required_string_rejects_none(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        with pytest.raises(ValidationError):
            Model(query=None)

    def test_required_string_accepts_value(self):
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        m = Model(query="zigzag")
        assert m.query == "zigzag"

    def test_optional_string_defaults_to_none(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "wing": {"type": "string"},
            },
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        m = Model(query="test")
        assert m.wing is None

    def test_exclude_none_omits_optional_keeps_required(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "wing": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        Model = _input_model_from_schema("search", schema)
        m = Model(query="test")
        dumped = m.model_dump(mode="json", exclude_none=True)
        assert dumped == {"query": "test"}

    def test_all_json_types_mapped(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "score": {"type": "number"},
                "active": {"type": "boolean"},
                "tags": {"type": "array"},
                "meta": {"type": "object"},
            },
            "required": ["name", "count", "score", "active", "tags", "meta"],
        }
        Model = _input_model_from_schema("full", schema)
        m = Model(name="x", count=1, score=0.5, active=True, tags=["a"], meta={"k": "v"})
        dumped = m.model_dump(mode="json")
        assert dumped == {
            "name": "x",
            "count": 1,
            "score": 0.5,
            "active": True,
            "tags": ["a"],
            "meta": {"k": "v"},
        }

    def test_empty_schema_creates_valid_model(self):
        Model = _input_model_from_schema("empty", {"type": "object"})
        m = Model()
        assert m.model_dump(mode="json") == {}

    def test_model_rejects_null_for_required_integer(self):
        schema = {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
            "required": ["limit"],
        }
        Model = _input_model_from_schema("limited", schema)
        with pytest.raises(ValidationError):
            Model(limit=None)

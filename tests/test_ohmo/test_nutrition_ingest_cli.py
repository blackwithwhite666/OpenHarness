"""Focused CLI tests for the nutrition-ingest ``status`` command.

These exercise the Typer command surface only; coordinator behaviour is
covered by ``test_nutrition_ingest_coordinator.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

from ohmo.cli import app


def _stub_status(*, enabled: bool = True, **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "enabled": enabled,
        "counts_by_state": {},
        "stage_latest_at": {},
        "last_delivery_receipt_at": None,
        "alert_conditions": {},
        "last_error": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def _stub_coordinator(monkeypatch):
    from ohmo import cli

    status_response = _stub_status()

    class StubCoordinator:
        def __init__(self, _config):
            pass

        def status(self):
            return status_response

    monkeypatch.setattr(cli, "load_gateway_config", lambda _workspace: type("Cfg", (), {"nutrition_ingest": None})())
    monkeypatch.setattr(cli, "NutritionIngestCoordinator", StubCoordinator)
    return status_response


def test_nutrition_status_cli_prints_aggregate_fields(_stub_coordinator) -> None:
    _stub_coordinator.clear()
    _stub_coordinator.update(
        _stub_status(
            enabled=True,
            counts_by_state={"pending_confirmation": 2, "completed": 3},
            stage_latest_at={"published": "2026-08-05T13:06:27+00:00"},
            last_delivery_receipt_at="2026-08-05T13:06:29+00:00",
            alert_conditions={"prompt_without_receipt": 1},
            last_error={
                "stage": "prompt",
                "error_class": "transport",
                "state": "retryable_error",
                "at": "2026-08-05T13:07:00+00:00",
            },
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["nutrition-ingest", "status"])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    assert payload["enabled"] is True
    assert payload["counts_by_state"] == {"completed": 3, "pending_confirmation": 2}
    assert payload["alert_conditions"] == {"prompt_without_receipt": 1}
    assert payload["last_delivery_receipt_at"] == "2026-08-05T13:06:29+00:00"
    assert payload["last_error"]["error_class"] == "transport"


def test_nutrition_status_cli_disabled_outputs_empty_aggregate(_stub_coordinator) -> None:
    _stub_coordinator.clear()
    _stub_coordinator.update(_stub_status(enabled=False))

    runner = CliRunner()
    result = runner.invoke(app, ["nutrition-ingest", "status"])
    assert result.exit_code == 0

    payload = json.loads(result.output)
    assert payload == {
        "enabled": False,
        "counts_by_state": {},
        "stage_latest_at": {},
        "last_delivery_receipt_at": None,
        "alert_conditions": {},
        "last_error": None,
    }


def test_nutrition_status_cli_never_leaks_identifiers(_stub_coordinator) -> None:
    _stub_coordinator.clear()
    _stub_coordinator.update(
        _stub_status(
            counts_by_state={"delivery_unknown": 1},
            last_error={
                "stage": "prompt",
                "error_class": "acknowledgement",
                "state": "delivery_unknown",
                "at": "2026-08-05T13:10:00+00:00",
            },
        )
    )

    runner = CliRunner()
    result = runner.invoke(app, ["nutrition-ingest", "status"])
    assert result.exit_code == 0

    for forbidden in (
        "dropbox-camera-v1-",
        "secret-candidate",
        "@private-user",
        "/private/camera/",
        "candidate_id",
    ):
        assert forbidden not in result.output

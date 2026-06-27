from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openharness.evals import (
    DECISION_TRACE_ENV_VAR,
    DECISION_TRACE_MAX_PAYLOAD_BYTES,
    STRUCTURAL_TOOL_COMPLETED,
    TRACE_DECISION,
    TRACE_MISSING_REQUIRED,
    TRACE_OBSERVATION,
    DecisionTraceRecorder,
    DecisionTraceValidationError,
    EvalEpisode,
    EvalStore,
    decision_trace_enabled,
    validate_decision_trace_payload,
)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "trace_event_id": "trace-1",
    }
    payload.update(overrides)
    return payload


def _store_with_episode(tmp_path: Path, episode_id: str = "ep-1") -> EvalStore:
    store = EvalStore(tmp_path / "evals")
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="test",
            app="openharness",
            session_id="session-1",
            user_text="hello",
        )
    )
    return store


def test_decision_trace_enabled_defaults_on_and_recognizes_disabled_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(DECISION_TRACE_ENV_VAR, raising=False)

    assert decision_trace_enabled() is True
    assert decision_trace_enabled({}) is True
    for value in ("0", "false", "no", "off", " FALSE "):
        assert decision_trace_enabled({DECISION_TRACE_ENV_VAR: value}) is False
    for value in ("1", "true", "yes", "on", ""):
        assert decision_trace_enabled({DECISION_TRACE_ENV_VAR: value}) is True


def test_recorder_appends_valid_decision_with_default_metadata(tmp_path: Path) -> None:
    store = _store_with_episode(tmp_path)
    recorder = DecisionTraceRecorder(store=store, episode_id="ep-1", enabled=True)

    event = recorder.record(
        TRACE_DECISION,
        _payload(decision="use the search tool"),
        tool_name="planner",
    )

    assert event is not None
    [recorded] = list(store.iter_events("ep-1"))
    assert recorded.kind == TRACE_DECISION
    assert recorded.tool_name == "planner"
    assert recorded.tool_call_id is None
    assert recorded.is_error is False
    assert recorded.payload == {
        "schema_version": 1,
        "trace_event_id": "trace-1",
        "decision": "use the search tool",
        "sensitivity": "private",
        "retention": "durable",
    }


def test_related_tool_call_id_populates_event_tool_call_id(tmp_path: Path) -> None:
    store = _store_with_episode(tmp_path)
    recorder = DecisionTraceRecorder(store=store, episode_id="ep-1", enabled=True)

    event = recorder.record(
        TRACE_OBSERVATION,
        _payload(related_tool_call_id="tool-call-1", observation="tool finished"),
    )

    assert event is not None
    assert event.tool_call_id == "tool-call-1"
    [recorded] = list(store.iter_events("ep-1"))
    assert recorded.tool_call_id == "tool-call-1"


def test_disabled_recorder_writes_no_events(tmp_path: Path) -> None:
    store = _store_with_episode(tmp_path)
    recorder = DecisionTraceRecorder(store=store, episode_id="ep-1", enabled=False)

    event = recorder.record(TRACE_DECISION, _payload(decision="skip recording"))

    assert event is None
    assert store.count_events("ep-1") == 0
    assert list(store.iter_events("ep-1")) == []


@pytest.mark.parametrize(
    ("kind", "payload", "match"),
    [
        ("not_a_trace_event", _payload(), "unknown decision trace event kind"),
        (TRACE_DECISION, {}, "schema_version"),
        (TRACE_DECISION, _payload(schema_version=2), "schema_version"),
        (TRACE_DECISION, _payload(schema_version=True), "schema_version"),
        (TRACE_DECISION, {"schema_version": 1}, "trace_event_id"),
        (TRACE_DECISION, _payload(trace_event_id=""), "trace_event_id"),
        (TRACE_DECISION, _payload(trace_event_id="   "), "trace_event_id"),
        (TRACE_DECISION, _payload(trace_event_id=123), "trace_event_id"),
        (TRACE_DECISION, _payload(parent_event_id=""), "parent_event_id"),
        (TRACE_DECISION, _payload(parent_event_id=None), "parent_event_id"),
        (TRACE_DECISION, _payload(related_tool_call_id=123), "related_tool_call_id"),
        (TRACE_DECISION, _payload(sensitivity="internal"), "sensitivity"),
        (TRACE_DECISION, _payload(retention="forever"), "retention"),
        (TRACE_DECISION, _payload(score=float("nan")), "JSON-serializable"),
        (
            TRACE_DECISION,
            _payload(details="x" * (DECISION_TRACE_MAX_PAYLOAD_BYTES + 1)),
            "exceeds",
        ),
    ],
)
def test_validate_decision_trace_payload_rejects_invalid_payloads(
    kind: str,
    payload: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(DecisionTraceValidationError, match=match):
        validate_decision_trace_payload(kind, payload)


@pytest.mark.parametrize(
    "payload",
    [
        _payload(api_key="a" * 24),
        _payload(note="sk-" + ("a" * 24)),
        _payload(
            note="-----BEGIN PRIVATE KEY-----\n"
            "not-a-real-key\n"
            "-----END PRIVATE KEY-----"
        ),
    ],
)
def test_validate_decision_trace_payload_rejects_obvious_secret_payloads(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(DecisionTraceValidationError, match="obvious secret"):
        validate_decision_trace_payload(TRACE_DECISION, payload)


def test_trace_missing_required_diagnostic_event_is_accepted_as_error(
    tmp_path: Path,
) -> None:
    store = _store_with_episode(tmp_path)
    recorder = DecisionTraceRecorder(store=store, episode_id="ep-1", enabled=True)

    event = recorder.record(
        TRACE_MISSING_REQUIRED,
        _payload(trace_event_id="diag-1", missing=["trace_finalization"]),
        is_error=True,
    )

    assert event is not None
    [recorded] = list(store.iter_events("ep-1"))
    assert recorded.kind == TRACE_MISSING_REQUIRED
    assert recorded.is_error is True
    assert recorded.payload["sensitivity"] == "private"
    assert recorded.payload["retention"] == "durable"


def test_record_structural_rejects_oversized_compact_payload(tmp_path: Path) -> None:
    store = _store_with_episode(tmp_path)
    recorder = DecisionTraceRecorder(store=store, episode_id="ep-1", enabled=True)

    with pytest.raises(DecisionTraceValidationError, match="structural payload exceeds"):
        recorder.record_structural(
            STRUCTURAL_TOOL_COMPLETED,
            {"output": "x" * (DECISION_TRACE_MAX_PAYLOAD_BYTES + 1)},
        )

    assert list(store.iter_events("ep-1")) == []

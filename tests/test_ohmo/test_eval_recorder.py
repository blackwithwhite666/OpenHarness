from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock
from openharness.engine.stream_events import (
    AssistantTurnComplete,
    ErrorEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from openharness.evals import (
    DecisionTraceValidationError,
    DECISION_TRACE_ENV_VAR,
    DECISION_TRACE_MAX_PAYLOAD_BYTES,
    STRUCTURAL_ASSISTANT_FINAL,
    STRUCTURAL_MODEL_CALL,
    STRUCTURAL_TOOL_COMPLETED,
    STRUCTURAL_TOOL_PERMISSION,
    STRUCTURAL_TOOL_STARTED,
    STRUCTURAL_TURN_STARTED,
    TRACE_DECISION,
    TRACE_FINALIZATION,
    DecisionTraceRecorder,
    EvalEpisode,
    EvalEvent,
    EvalStore,
)

from ohmo.evals import GatewayEvalRecorder, get_eval_store


def _new_recorder(
    tmp_path: Path,
    *,
    episode_id: str = "ep-recorder",
    user_goal: str = "",
) -> tuple[GatewayEvalRecorder, EvalStore]:
    store = get_eval_store(tmp_path)
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="hello",
        )
    )
    return GatewayEvalRecorder(store=store, episode_id=episode_id, user_goal=user_goal), store


def _finalization_payload(annotations: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "trace_event_id": "trace-final-1",
        "annotations": annotations,
    }


def test_gateway_eval_recorder_record_event_delegates_to_legacy_structural_recorder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    calls: list[dict[str, Any]] = []
    original_record_legacy_structural = DecisionTraceRecorder.record_legacy_structural

    def record_legacy_structural_spy(
        self: DecisionTraceRecorder,
        kind: str,
        payload: Mapping[str, Any],
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        is_error: bool = False,
    ) -> EvalEvent | None:
        calls.append(
            {
                "episode_id": self.episode_id,
                "enabled": self.enabled,
                "kind": kind,
                "payload": dict(payload),
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "is_error": is_error,
            }
        )
        return original_record_legacy_structural(
            self,
            kind,
            payload,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            is_error=is_error,
        )

    monkeypatch.setattr(
        DecisionTraceRecorder,
        "record_legacy_structural",
        record_legacy_structural_spy,
    )

    recorder.record_event(
        "gateway_error",
        payload={"text": "gateway failed", "metadata": {"path": Path("trace.json")}},
        tool_name="gateway",
        tool_call_id="call-err",
        is_error=True,
    )

    assert calls == [
        {
            "episode_id": "ep-recorder",
            "enabled": True,
            "kind": "gateway_error",
            "payload": {
                "text": "gateway failed",
                "metadata": {"path": "trace.json"},
            },
            "tool_name": "gateway",
            "tool_call_id": "call-err",
            "is_error": True,
        }
    ]
    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded == EvalEvent(
        episode_id="ep-recorder",
        kind="gateway_error",
        timestamp=recorded.timestamp,
        payload={"text": "gateway failed", "metadata": {"path": "trace.json"}},
        tool_name="gateway",
        tool_call_id="call-err",
        is_error=True,
    )


def test_gateway_eval_recorder_keeps_legacy_capture_when_decision_trace_env_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(DECISION_TRACE_ENV_VAR, "0")
    recorder, store = _new_recorder(tmp_path)

    recorder.record_gateway_final(text="done")

    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == "gateway_final"
    assert recorded.payload == {"text": "done", "metadata": {}}


def test_gateway_eval_recorder_runtime_adapter_records_trace_and_allowed_structural_events(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    runtime_recorder = recorder.decision_trace_recorder

    runtime_recorder.record(
        TRACE_DECISION,
        {
            "schema_version": 1,
            "trace_event_id": "trace-gateway-1",
            "decision": "route gateway trace events into the gateway episode",
        },
    )
    runtime_recorder.record_structural(
        STRUCTURAL_TURN_STARTED,
        {"model": "gpt-prod", "user_text_summary": "hello"},
    )
    runtime_recorder.record_structural(
        STRUCTURAL_TOOL_PERMISSION,
        {"allowed": True, "requires_confirmation": False, "read_only": True},
        tool_name="trace",
        tool_call_id="trace-call-1",
    )

    events = list(store.iter_events("ep-recorder"))
    assert [event.kind for event in events] == [
        TRACE_DECISION,
        STRUCTURAL_TURN_STARTED,
        STRUCTURAL_TOOL_PERMISSION,
    ]
    assert {event.episode_id for event in events} == {"ep-recorder"}
    assert events[0].payload["decision"] == (
        "route gateway trace events into the gateway episode"
    )
    assert events[2].tool_name == "trace"
    assert events[2].tool_call_id == "trace-call-1"


def test_gateway_eval_recorder_runtime_adapter_records_completed_duration_but_skips_start_duplicates(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    runtime_recorder = recorder.decision_trace_recorder

    assert runtime_recorder.record_structural(
        STRUCTURAL_MODEL_CALL,
        {"model": "gpt-prod", "input_tokens": 1, "output_tokens": 1},
    ) is None
    assert runtime_recorder.record_structural(
        STRUCTURAL_TOOL_STARTED,
        {"input_keys": ["url"]},
        tool_name="web_fetch",
        tool_call_id="toolu-1",
    ) is None
    completed = runtime_recorder.record_structural(
        STRUCTURAL_TOOL_COMPLETED,
        {"is_error": False, "duration_ms": 12.5},
        tool_name="web_fetch",
        tool_call_id="toolu-1",
    )
    assert completed is not None
    runtime_recorder.record_structural(
        STRUCTURAL_ASSISTANT_FINAL,
        {"assistant_text_summary": "done", "model": "gpt-prod"},
    )

    events = list(store.iter_events("ep-recorder"))
    assert [event.kind for event in events] == [
        STRUCTURAL_TOOL_COMPLETED,
        STRUCTURAL_ASSISTANT_FINAL,
    ]
    recorded_completed = events[0]
    assert recorded_completed.tool_name == "web_fetch"
    assert recorded_completed.tool_call_id == "toolu-1"
    assert isinstance(recorded_completed.payload["duration_ms"], (int, float))
    assert recorded_completed.payload["duration_ms"] == 12.5
    assert events[1].payload["assistant_text_summary"] == "done"


def test_gateway_eval_recorder_record_model_call_writes_tokens(tmp_path: Path) -> None:
    episode_id = "ep-recorder"
    recorder, store = _new_recorder(tmp_path, episode_id=episode_id)
    event = AssistantTurnComplete(
        message=ConversationMessage(
            role="assistant",
            content=[TextBlock(text="done")],
        ),
        usage=UsageSnapshot(input_tokens=12, output_tokens=5),
    )

    recorder.record_model_call(event, model="gpt-prod")

    [recorded] = list(store.iter_events(episode_id))
    assert recorded.kind == "model_call"
    assert recorded.payload == {
        "model": "gpt-prod",
        "input_tokens": 12,
        "output_tokens": 5,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
    }


def test_gateway_eval_recorder_record_tool_started_preserves_input_and_capability(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    tool_input = {"command": "maps-cli reviews 9089 --json", "cwd": "/work"}

    recorder.record_tool_started(
        ToolExecutionStarted(
            tool_name="bash",
            tool_input=tool_input,
            tool_call_id="call-1",
        )
    )

    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == "tool_started"
    assert recorded.tool_name == "bash"
    assert recorded.tool_call_id == "call-1"
    assert recorded.is_error is False
    assert recorded.payload == {
        "input_summary": "{'command': 'maps-cli reviews 9089 --json', 'cwd': '/work'}",
        "input": tool_input,
        "binaries": ["maps-cli"],
        "capability": "bash:maps-cli reviews",
    }


def test_gateway_eval_recorder_record_tool_completed_preserves_output_and_error(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)

    recorder.record_tool_completed(
        ToolExecutionCompleted(
            tool_name="web_fetch",
            output="line one\n  line two",
            is_error=True,
            tool_call_id="call-2",
        )
    )

    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == "tool_completed"
    assert recorded.tool_name == "web_fetch"
    assert recorded.tool_call_id == "call-2"
    assert recorded.is_error is True
    assert recorded.payload == {
        "output_summary": "line one line two",
        "output": "line one\n  line two",
    }


def test_gateway_eval_recorder_record_tool_completed_preserves_large_output(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    large_output = "x" * (DECISION_TRACE_MAX_PAYLOAD_BYTES + 1024)

    recorder.record_tool_completed(
        ToolExecutionCompleted(
            tool_name="web_fetch",
            output=large_output,
            is_error=False,
            tool_call_id="call-large",
        )
    )

    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == "tool_completed"
    assert recorded.tool_name == "web_fetch"
    assert recorded.tool_call_id == "call-large"
    assert recorded.is_error is False
    assert recorded.payload["output"] == large_output
    assert len(recorded.payload["output"]) > DECISION_TRACE_MAX_PAYLOAD_BYTES


def test_gateway_eval_recorder_records_error_and_gateway_terminal_events(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)

    recorder.record_engine_error(ErrorEvent(message="engine failed", recoverable=False))
    recorder.record_gateway_final(text="done", metadata={"trace_id": "trace-1"})
    recorder.record_gateway_error(text="gateway failed", metadata={"phase": "send"})
    recorder.record_exception(RuntimeError("boom"))
    recorder.finish(status="failed")

    events = list(store.iter_events("ep-recorder"))
    assert [(event.kind, event.payload, event.is_error) for event in events] == [
        ("engine_error", {"message": "engine failed", "recoverable": False}, True),
        ("gateway_final", {"text": "done", "metadata": {"trace_id": "trace-1"}}, False),
        ("gateway_error", {"text": "gateway failed", "metadata": {"phase": "send"}}, True),
        ("exception", {"type": "RuntimeError", "message": "boom"}, True),
        ("episode_finished", {"status": "failed"}, True),
    ]


def test_gateway_eval_recorder_finish_is_idempotent(tmp_path: Path) -> None:
    recorder, store = _new_recorder(tmp_path)

    recorder.finish(status="completed")
    recorder.finish(status="failed")

    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == "episode_finished"
    assert recorded.payload == {"status": "completed"}
    assert recorded.is_error is False


def test_gateway_eval_recorder_payloads_are_json_safe(tmp_path: Path) -> None:
    recorder, store = _new_recorder(tmp_path)
    timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    recorder.record_gateway_final(
        text="done",
        metadata={
            "path": Path("workspace/file.txt"),
            "timestamp": timestamp,
            "bytes": b"hello",
            "nan": float("nan"),
            "inf": float("inf"),
            "negative_inf": float("-inf"),
        },
    )

    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == "gateway_final"
    assert recorded.payload == {
        "text": "done",
        "metadata": {
            "path": "workspace/file.txt",
            "timestamp": "2026-01-02T03:04:05+00:00",
            "bytes": "hello",
            "nan": "nan",
            "inf": "inf",
            "negative_inf": "-inf",
        },
    }


def test_gateway_eval_recorder_finalization_status_records_invalid_then_later_valid_recovery(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    runtime_recorder = recorder.decision_trace_recorder

    assert recorder.decision_trace_status == "missing"
    assert recorder.nutrition_annotation_status == "not_applicable"
    assert recorder.decision_trace_envelope is None

    with pytest.raises(DecisionTraceValidationError, match="energy_kcal_min"):
        runtime_recorder.record(
            TRACE_FINALIZATION,
            _finalization_payload({"nutrition": {"energy_kcal_min": -1}}),
        )

    assert recorder.decision_trace_status == "invalid"
    assert recorder.nutrition_annotation_status == "invalid"
    assert recorder.decision_trace_envelope is None
    assert store.count_events("ep-recorder") == 0

    runtime_recorder.record(
        TRACE_FINALIZATION,
        _finalization_payload(
            {
                "nutrition": {
                    "energy_kcal_min": 10,
                    "items": [{"name": "egg", "quantity_text": "1", "energy_kcal_min": 80}],
                }
            }
        ),
    )

    assert recorder.decision_trace_status == "recorded"
    assert recorder.nutrition_annotation_status == "recorded"
    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == TRACE_FINALIZATION
    assert recorded.payload["annotations"]["nutrition"]["energy_kcal_min"] == 10.0


def test_gateway_eval_recorder_nutrition_status_is_missing_when_applicable_without_nutrition(
    tmp_path: Path,
) -> None:
    recorder, store = _new_recorder(tmp_path)
    runtime_recorder = recorder.decision_trace_recorder

    runtime_recorder.trace_requirement_signals("сколько калорий в ужине?")

    runtime_recorder.record(
        TRACE_FINALIZATION,
        {"schema_version": 1, "trace_event_id": "trace-final-2"},
    )

    assert recorder.decision_trace_status == "recorded"
    assert recorder.nutrition_annotation_status == "missing"
    [recorded] = list(store.iter_events("ep-recorder"))
    assert recorded.kind == TRACE_FINALIZATION
    assert "annotations" not in recorded.payload


@pytest.mark.parametrize(
    "text, expected_signal",
    [
        ("посчитай калорийность", True),
        ("сколько калорий", True),
        ("Сколько ккал в этом супе", True),
        ("Сколько белков и жиров в блюде", True),
        ("Мне важно знать БЖУ этого блюда", True),
        ("Мне нужен белок и жиры, пожалуйста", True),
        ("Я сейчас посмотрю фильм", False),
        ("Сколько белая рубашка стоит?", False),
    ],
)
def test_gateway_eval_recorder_trace_requirement_signals_is_marker_driven(
    tmp_path: Path,
    text: str,
    expected_signal: bool,
) -> None:
    recorder, _ = _new_recorder(tmp_path, episode_id="ep-markers")
    signals = recorder.decision_trace_recorder.trace_requirement_signals(text)

    if expected_signal:
        assert signals == ("ohmo_nutrition_request",)
    else:
        assert signals == ()


def test_gateway_eval_recorder_trace_requirement_signals_prefers_user_goal_when_final_text_is_neutral(
    tmp_path: Path,
) -> None:
    recorder, _ = _new_recorder(tmp_path, user_goal="Сколько калорий в обеде сегодня?")
    signals = recorder.decision_trace_recorder.trace_requirement_signals("можно краткий апдейт?")

    assert signals == ("ohmo_nutrition_request",)
    assert recorder.nutrition_annotation_status == "missing"


def test_gateway_eval_recorder_nutrition_applicability_is_monotonic_within_turn(
    tmp_path: Path,
) -> None:
    recorder, _ = _new_recorder(tmp_path, episode_id="ep-monotonic")
    runtime_recorder = recorder.decision_trace_recorder

    assert recorder.nutrition_annotation_status == "not_applicable"

    runtime_recorder.trace_requirement_signals("сколько калорий в ужине?")
    assert recorder.nutrition_annotation_status == "missing"

    runtime_recorder.trace_requirement_signals("я сейчас посмотрю фильм")
    assert recorder.nutrition_annotation_status == "missing"


def test_gateway_eval_recorder_nutrition_applicability_can_be_marked_without_finalization(
    tmp_path: Path,
) -> None:
    recorder, _ = _new_recorder(tmp_path, episode_id="ep-applicability")
    runtime_recorder = recorder.decision_trace_recorder

    assert recorder.nutrition_annotation_status == "not_applicable"
    assert runtime_recorder.trace_requirement_signals("подскажи, какой обед был, пожалуйста") == ()

    runtime_recorder.trace_requirement_signals("посчитай калорийность обеда")
    assert recorder.nutrition_annotation_status == "missing"


def test_gateway_eval_recorder_decision_trace_envelope_is_json_safe_and_immutable(
    tmp_path: Path,
) -> None:
    recorder, _ = _new_recorder(tmp_path)
    runtime_recorder = recorder.decision_trace_recorder

    runtime_recorder.record(
        TRACE_FINALIZATION,
        _finalization_payload(
            {
                "nutrition": {
                    "energy_kcal_min": 100,
                    "energy_kcal_max": 120,
                    "items": [
                        {
                            "name": "egg",
                            "quantity_text": "1",
                            "energy_kcal_min": 100,
                            "energy_kcal_max": 120,
                        }
                    ],
                }
            }
        ),
    )

    envelope = recorder.decision_trace_envelope
    assert envelope is not None
    assert envelope["kind"] == TRACE_FINALIZATION
    assert envelope["episode_id"] == "ep-recorder"
    assert envelope["schema_version"] == 1
    assert envelope["trace_event_id"] == "trace-final-1"
    assert envelope["timestamp"] is not None
    assert "annotations" in envelope
    assert "payload" not in envelope
    json.dumps(envelope)

    with pytest.raises(TypeError):
        envelope["kind"] = "mutated"

    envelope["annotations"]["nutrition"]["assumptions"].append("example")
    assert (
        recorder.decision_trace_envelope["annotations"]["nutrition"]["assumptions"]
        == []
    )

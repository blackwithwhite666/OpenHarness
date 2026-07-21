from __future__ import annotations

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
    DECISION_TRACE_ENV_VAR,
    DECISION_TRACE_MAX_PAYLOAD_BYTES,
    STRUCTURAL_ASSISTANT_FINAL,
    STRUCTURAL_MODEL_CALL,
    STRUCTURAL_TOOL_COMPLETED,
    STRUCTURAL_TOOL_PERMISSION,
    STRUCTURAL_TOOL_STARTED,
    STRUCTURAL_TURN_STARTED,
    TRACE_DECISION,
    DecisionTraceRecorder,
    EvalEpisode,
    EvalEvent,
    EvalStore,
)

from ohmo.evals import GatewayEvalRecorder, get_eval_store


def _new_recorder(
    tmp_path: Path, episode_id: str = "ep-recorder"
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
    return GatewayEvalRecorder(store=store, episode_id=episode_id), store


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
    recorder, store = _new_recorder(tmp_path, episode_id)
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

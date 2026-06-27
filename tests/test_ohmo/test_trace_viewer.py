from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openharness.evals import (
    TRACE_DECISION,
    TRACE_MISSING_REQUIRED,
    TRACE_UNCERTAINTY,
    EvalEpisode,
    EvalEvent,
)
from ohmo.evals import get_eval_store
from ohmo.evals.viewer import (
    create_app,
    episode_to_trace_viewer_data,
    eval_case_to_trace_viewer_data,
    list_eval_runs,
    list_eval_traces,
    list_prod_traces,
)
from starlette.testclient import TestClient


def test_episode_to_trace_viewer_data_maps_successful_tool_turn(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, base=base)

    data = episode_to_trace_viewer_data(store, "ep-1")

    assert data["traceRecord"]["id"] == "ep-1"
    assert data["traceRecord"]["spansCount"] == 2
    assert data["traceRecord"]["durationMs"] == 500
    assert data["traceRecord"]["startTimeMs"] == int(base.timestamp() * 1000)

    spans = data["spans"]
    assert len(spans) == 1
    root = spans[0]
    assert root["type"] == "agent_invocation"
    assert root["input"] == "посмотри отзывы"
    assert root["output"] == "вот отзывы"
    assert root["status"] == "success"

    children = root["children"]
    assert len(children) == 1
    child = children[0]
    assert child["title"].startswith("bash:maps-cli")
    assert child["type"] == "tool_execution"
    assert child["status"] == "success"
    assert child["durationMs"] > 0
    assert child["durationMs"] == 200
    assert child["output"] == "..."


def test_episode_to_trace_viewer_data_renders_prod_model_calls(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    episode_id = "ep-model-calls"
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            created_at=base,
            user_text="проверь отзывы",
            metadata={"model": "gpt-prod", "cwd": "/tmp/project"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="inbound_message",
            timestamp=base,
            payload={"user_text": "проверь отзывы"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="model_call",
            timestamp=base + timedelta(milliseconds=80),
            payload={"model": "gpt-prod", "input_tokens": 3, "output_tokens": 2},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            timestamp=base + timedelta(milliseconds=100),
            payload={"input": {"command": "maps-cli reviews Xander"}},
            tool_name="bash",
            tool_call_id="c1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=200),
            payload={"output": "review one"},
            tool_name="bash",
            tool_call_id="c1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="model_call",
            timestamp=base + timedelta(milliseconds=350),
            payload={"model": "gpt-prod", "input_tokens": 11, "output_tokens": 7},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            timestamp=base + timedelta(milliseconds=400),
            payload={"input": {"command": "maps-cli orgs Xander"}},
            tool_name="bash",
            tool_call_id="c2",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=500),
            payload={"output": "org details"},
            tool_name="bash",
            tool_call_id="c2",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            timestamp=base + timedelta(milliseconds=600),
            payload={"text": "готово"},
        )
    )

    data = episode_to_trace_viewer_data(store, episode_id)

    assert data["traceRecord"]["spansCount"] == 5
    assert data["traceRecord"]["totalTokens"] == 23
    children = data["spans"][0]["children"]
    assert [child["type"] for child in children] == [
        "llm_call",
        "tool_execution",
        "llm_call",
        "tool_execution",
    ]
    base_ms = int(base.timestamp() * 1000)
    assert [child["startTimeMs"] for child in children] == [
        base_ms,
        base_ms + 100,
        base_ms + 200,
        base_ms + 400,
    ]

    first_llm = children[0]
    assert first_llm["title"] == "gpt-prod"
    assert first_llm["tokensCount"] == 5
    assert first_llm["durationMs"] == 80
    assert first_llm["input"] is None
    assert first_llm["output"] is None

    second_llm = children[2]
    assert second_llm["title"] == "gpt-prod"
    assert second_llm["tokensCount"] == 18
    assert second_llm["startTimeMs"] == base_ms + 200
    assert second_llm["endTimeMs"] == base_ms + 350
    attributes = {
        item["key"]: item["value"]["stringValue"]
        for item in second_llm["attributes"]
    }
    assert attributes["input_tokens"] == "11"
    assert attributes["output_tokens"] == "7"


def test_episode_to_trace_viewer_data_renders_decision_trace_metadata_only(
    tmp_path: Path,
) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    episode_id = "ep-decision-trace"
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            created_at=base,
            user_text="check task",
            metadata={"model": "gpt-prod", "cwd": "/tmp/project"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="inbound_message",
            timestamp=base,
            payload={"user_text": "check task"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="model_call",
            timestamp=base + timedelta(milliseconds=50),
            payload={"model": "gpt-prod", "input_tokens": 5, "output_tokens": 8},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind=TRACE_DECISION,
            timestamp=base + timedelta(milliseconds=70),
            payload={
                "schema_version": 1,
                "trace_event_id": "trace-decision-1",
                "related_tool_call_id": "c1",
                "sensitivity": "private",
                "retention": "durable",
                "reason": "evidence_gap",
                "signals": ["unsupported_claim"],
                "decision": "PRIVATE_TRACE_DECISION_RAW",
                "unsupported_claims": ["PRIVATE_UNSUPPORTED_CLAIM_RAW"],
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            timestamp=base + timedelta(milliseconds=100),
            payload={"input": {"command": "safe-tool"}},
            tool_name="bash",
            tool_call_id="c1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind=TRACE_UNCERTAINTY,
            timestamp=base + timedelta(milliseconds=120),
            payload={
                "schema_version": 1,
                "trace_event_id": "trace-uncertainty-1",
                "parent_event_id": "trace-decision-1",
                "sensitivity": "secret",
                "retention": "session",
                "uncertainty_status": "open",
                "uncertainty": "PRIVATE_TRACE_UNCERTAINTY_RAW",
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=200),
            payload={"output": "safe output"},
            tool_name="bash",
            tool_call_id="c1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="assistant_final",
            timestamp=base + timedelta(milliseconds=250),
            payload={
                "trace_required": True,
                "trace_required_reason": "sensitive_action",
                "trace_required_signals": ["tool_use"],
                "model_trace_recorded": False,
                "assistant_text_summary": "PRIVATE_ASSISTANT_FINAL_SUMMARY",
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind=TRACE_MISSING_REQUIRED,
            timestamp=base + timedelta(milliseconds=260),
            payload={
                "schema_version": 1,
                "trace_event_id": "trace-missing-1",
                "sensitivity": "private",
                "retention": "durable",
                "reason": "required_trace_absent",
                "missing": ["trace_finalization"],
            },
            is_error=True,
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            timestamp=base + timedelta(milliseconds=300),
            payload={"text": "done"},
        )
    )

    data = episode_to_trace_viewer_data(store, episode_id)

    root = data["spans"][0]
    assert root["status"] == "error"
    assert data["traceRecord"]["spansCount"] == 6
    children = root["children"]
    assert [child["type"] for child in children] == [
        "llm_call",
        "decision_trace",
        "tool_execution",
        "decision_trace",
        "decision_trace",
    ]
    assert [child["title"] for child in children if child["type"] == "decision_trace"] == [
        TRACE_DECISION,
        TRACE_UNCERTAINTY,
        TRACE_MISSING_REQUIRED,
    ]

    root_attrs = {item["key"]: item["value"]["stringValue"] for item in root["attributes"]}
    assert root_attrs["decision_trace_event_count"] == "3"
    assert root_attrs["decision_trace_model_event_count"] == "2"
    assert root_attrs["decision_trace_diagnostic_event_count"] == "1"
    assert root_attrs["decision_trace_missing_required_count"] == "1"
    assert root_attrs["decision_trace_required_count"] == "1"
    assert root_attrs["decision_trace_recorded_count"] == "0"
    assert root_attrs["decision_trace_coverage_status"] == "missing"
    assert root_attrs["unsupported_claim_count"] == "1"
    assert root_attrs["uncertainty_trace_count"] == "1"
    assert root_attrs["uncertainty_status"] == "present"
    assert root_attrs["decision_trace_sensitivity_labels"] == '["private", "secret"]'
    assert root_attrs["decision_trace_max_sensitivity"] == "secret"

    decision_span = children[1]
    assert decision_span["input"] is None
    assert decision_span["output"] is None
    decision_attrs = {
        item["key"]: item["value"]["stringValue"]
        for item in decision_span["attributes"]
    }
    assert decision_attrs["kind"] == TRACE_DECISION
    assert decision_attrs["trace_event_id"] == "trace-decision-1"
    assert decision_attrs["related_tool_call_id"] == "c1"
    assert decision_attrs["sensitivity"] == "private"
    assert decision_attrs["retention"] == "durable"
    assert decision_attrs["reason"] == "evidence_gap"
    assert decision_attrs["signals"] == '["unsupported_claim"]'
    assert decision_attrs["unsupported_claim_count"] == "1"
    assert children[-1]["status"] == "error"

    serialized = json.dumps(data, ensure_ascii=False)
    for private_fragment in (
        "PRIVATE_TRACE_DECISION_RAW",
        "PRIVATE_TRACE_UNCERTAINTY_RAW",
        "PRIVATE_UNSUPPORTED_CLAIM_RAW",
        "PRIVATE_ASSISTANT_FINAL_SUMMARY",
    ):
        assert private_fragment not in serialized


def test_episode_to_trace_viewer_data_sanitizes_legacy_trace_tool_input(
    tmp_path: Path,
) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    episode_id = "ep-legacy-trace-tool"
    private_fragment = "PRIVATE_RAW_TRACE_TOOL_PAYLOAD"
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            created_at=base,
            user_text="trace a sensitive step",
            metadata={"model": "gpt-prod", "cwd": "/tmp/project"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="inbound_message",
            timestamp=base,
            payload={"user_text": "trace a sensitive step"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            timestamp=base + timedelta(milliseconds=100),
            payload={
                "input_summary": private_fragment,
                "input": {
                    "kind": TRACE_DECISION,
                    "payload": {
                        "schema_version": 1,
                        "trace_event_id": "trace-legacy-1",
                        "related_tool_call_id": "trace-call-1",
                        "sensitivity": "secret",
                        "retention": "session",
                        "decision": private_fragment,
                        "unsupported_claims": [private_fragment],
                    },
                },
            },
            tool_name="trace",
            tool_call_id="trace-call-1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=160),
            payload={
                "is_error": False,
                "duration_ms": 42.5,
                "output_summary": "Recorded decision trace event: trace_decision",
                "output_length": 44,
            },
            tool_name="trace",
            tool_call_id="trace-call-1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=200),
            payload={"output": "Recorded decision trace event: trace_decision"},
            tool_name="trace",
            tool_call_id="trace-call-1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            timestamp=base + timedelta(milliseconds=250),
            payload={"text": "done"},
        )
    )

    data = episode_to_trace_viewer_data(store, episode_id)

    root = data["spans"][0]
    [trace_span] = root["children"]
    assert trace_span["title"] == "trace"
    assert trace_span["type"] == "tool_execution"
    assert trace_span["durationMs"] == 60
    assert trace_span["output"] is None
    assert "duration_ms" in trace_span["raw"]
    safe_input = json.loads(trace_span["input"])
    assert safe_input == {
        "kind": TRACE_DECISION,
        "trace_event_id": "trace-legacy-1",
        "related_tool_call_id": "trace-call-1",
        "sensitivity": "secret",
        "retention": "session",
        "schema_version": 1,
        "unsupported_claim_count": 1,
    }

    serialized = json.dumps(data, ensure_ascii=False)
    assert private_fragment not in serialized
    assert "input_summary" not in root["raw"]


def test_episode_to_trace_viewer_data_propagates_tool_error_status(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, episode_id="ep-error", base=base, is_error=True)

    data = episode_to_trace_viewer_data(store, "ep-error")

    root = data["spans"][0]
    child = root["children"][0]
    assert child["status"] == "error"
    assert root["status"] == "error"


def test_list_prod_traces_returns_prod_items_and_filters_by_query(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, episode_id="ep-1", base=base, user_text="посмотри отзывы")
    _append_trace_episode(
        store,
        episode_id="ep-2",
        base=base + timedelta(seconds=1),
        user_text="проверь погоду",
    )

    all_items = list_prod_traces(store)
    assert all_items["total"] == 2
    assert all_items["traces"][0]["id"] == "ep-2"
    assert all_items["traces"][0]["kind"] == "prod"

    filtered = list_prod_traces(store, q="ОТЗЫВЫ")
    assert filtered["total"] == 1
    assert filtered["traces"][0]["id"] == "ep-1"
    assert filtered["traces"][0]["kind"] == "prod"


def test_eval_report_lane_lists_and_renders_metadata_traces(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    _write_eval_report(store)

    runs = list_eval_runs(store)
    created_at = int(datetime(2026, 1, 1, 12, 44, 5, tzinfo=timezone.utc).timestamp() * 1000)
    assert runs["runs"] == [
        {
            "run": "eval_report_x.json",
            "reportId": "report-x",
            "packId": "pack-x",
            "createdAt": created_at,
            "scorer": "trajectory_judge_v1",
            "executor": "replay-tools",
            "samples": 3,
            "caseCount": 1,
            "passedCount": 0,
            "failedCount": 1,
        }
    ]

    traces = list_eval_traces(store, "eval_report_x.json")
    assert traces["total"] == 1
    assert traces["traces"][0]["id"] == "case-1"
    assert traces["traces"][0]["kind"] == "eval"
    assert traces["traces"][0]["goldEpisodeId"] == "ep-gold"
    assert traces["traces"][0]["sampleCount"] == 3
    assert traces["traces"][0]["spansCount"] == 3

    data = eval_case_to_trace_viewer_data(store, "eval_report_x.json", "case-1")
    assert data["goldEpisodeId"] == "ep-gold"
    assert data["sample"] == 0
    assert data["sampleCount"] == 3
    assert data["badges"] == [
        {"label": "score 0.9"},
        {"label": "failed"},
        {"label": "1/3"},
    ]
    root = data["spans"][0]
    assert root["status"] == "error"
    assert len(root["children"]) == 2
    assert root["children"][1]["status"] == "error"
    attributes = {item["key"]: item["value"]["stringValue"] for item in root["attributes"]}
    assert attributes["decision_trace_event_count"] == "2"
    assert attributes["decision_trace_coverage_status"] == "complete"
    assert attributes["decision_trace_sensitivity_labels"] == '["private"]'

    client = TestClient(create_app(tmp_path))
    assert client.get("/api/runs").json()["runs"][0]["run"] == "eval_report_x.json"

    route_traces = client.get("/api/eval-traces", params={"run": "eval_report_x.json"})
    assert route_traces.status_code == 200
    assert route_traces.json()["traces"][0]["kind"] == "eval"

    missing_run = client.get("/api/eval-traces")
    assert missing_run.status_code == 400

    route_trace = client.get(
        "/api/eval-traces/case-1",
        params={"run": "eval_report_x.json"},
    )
    assert route_trace.status_code == 200
    assert route_trace.json()["goldEpisodeId"] == "ep-gold"


def test_eval_case_to_trace_viewer_data_prefers_rich_trace(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    _write_eval_report(store)
    _write_rich_eval_trace(store)

    data = eval_case_to_trace_viewer_data(store, "eval_report_x.json", "case-1")

    assert data["goldEpisodeId"] == "ep-gold"
    assert data["traceRecord"]["spansCount"] == 4
    assert data["traceRecord"]["durationMs"] == 600
    assert data["traceRecord"]["totalTokens"] == 40
    assert data["traceRecord"]["agentDescription"] == "replay-tools · trajectory_judge_v1"
    assert data["badges"] == [
        {"label": "score 0.9"},
        {"label": "failed"},
        {"label": "1/3"},
        {"label": "judge: fail"},
    ]

    root = data["spans"][0]
    assert root["status"] == "error"
    assert root["input"] == "покажи отзывы Xander"
    assert root["output"] == "нашёл два отзыва"
    assert root["startTimeMs"] == 900
    assert root["endTimeMs"] == 1_500
    assert root["durationMs"] == 600
    attributes = {item["key"]: item["value"]["stringValue"] for item in root["attributes"]}
    assert attributes["judge_verdict"] == "fail"
    assert attributes["judge_reason"] == "final answer missed one required detail"
    assert attributes["decision_trace_event_count"] == "3"
    assert attributes["decision_trace_coverage_status"] == "partial"

    assert len(root["children"]) == 3
    assert [child["type"] for child in root["children"]] == [
        "llm_call",
        "tool_execution",
        "llm_call",
    ]
    assert [child["startTimeMs"] for child in root["children"]] == [900, 1_100, 1_300]

    first_llm = root["children"][0]
    assert first_llm["title"] == "gpt-5.5"
    assert first_llm["status"] == "success"
    assert first_llm["tokensCount"] == 13
    assert first_llm["input"] is None
    assert first_llm["output"] is None
    llm_attributes = {
        item["key"]: item["value"]["stringValue"]
        for item in first_llm["attributes"]
    }
    assert llm_attributes["input_tokens"] == "10"
    assert llm_attributes["output_tokens"] == "3"

    child = root["children"][1]
    assert child["title"] == "bash:maps-cli reviews"
    assert child["type"] == "tool_execution"
    assert child["status"] == "success"
    assert child["input"] == json.dumps(
        {"command": "maps-cli reviews Xander"},
        ensure_ascii=False,
        sort_keys=True,
    )
    assert child["output"] == "review one\nreview two"
    assert child["startTimeMs"] == 1_100
    assert child["endTimeMs"] == 1_250
    assert child["durationMs"] > 0

    final_llm = root["children"][2]
    assert final_llm["title"] == "gpt-5.5"
    assert final_llm["tokensCount"] == 27


def test_eval_trace_multi_sample_count_and_requested_sample_load(tmp_path: Path) -> None:
    store = get_eval_store(tmp_path)
    _write_eval_report(store)
    _write_rich_eval_trace(store, final_text="sample zero answer")
    _write_rich_eval_trace(
        store,
        sample_index=1,
        final_text="sample one answer",
        judge_verdict="pass",
        judge_reason="sample one passed",
        score=1.0,
        passed=True,
    )

    traces = list_eval_traces(store, "eval_report_x.json")

    assert traces["traces"][0]["sampleCount"] == 2

    data = eval_case_to_trace_viewer_data(
        store,
        "eval_report_x.json",
        "case-1",
        sample=1,
    )

    assert data["sample"] == 1
    assert data["sampleCount"] == 2
    assert data["spans"][0]["output"] == "sample one answer"
    assert data["spans"][0]["status"] == "success"
    assert data["badges"][-1] == {"label": "judge: pass"}

    client = TestClient(create_app(tmp_path))
    route_trace = client.get(
        "/api/eval-traces/case-1",
        params={"run": "eval_report_x.json", "sample": "1"},
    )
    assert route_trace.status_code == 200
    route_payload = route_trace.json()
    assert route_payload["sample"] == 1
    assert route_payload["spans"][0]["output"] == "sample one answer"


def test_episode_session_conversation_renders_session_chat(tmp_path: Path) -> None:
    from ohmo.evals.viewer import episode_session_conversation

    store = get_eval_store(tmp_path)
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _append_trace_episode(store, episode_id="ep-1", base=base, user_text="первый вопрос")
    _append_trace_episode(
        store,
        episode_id="ep-2",
        base=base + timedelta(minutes=5),
        user_text="второй вопрос",
    )

    data = episode_session_conversation(store, "ep-2")

    assert data["kind"] == "session"
    assert data["sessionId"] == "session-1"
    assert data["anchorEpisodeId"] == "ep-2"
    roles = [(m["role"], m["episodeId"]) for m in data["messages"]]
    assert roles == [
        ("user", "ep-1"),
        ("assistant", "ep-1"),
        ("user", "ep-2"),
        ("assistant", "ep-2"),
    ]
    assistant = data["messages"][1]
    assert assistant["text"] == "вот отзывы"
    assert assistant["toolCalls"][0]["name"].startswith("bash:maps-cli")
    assert assistant["toolCalls"][0]["status"] == "success"

    client = TestClient(create_app(tmp_path))
    route = client.get("/api/session/ep-2")
    assert route.status_code == 200
    assert route.json()["anchorEpisodeId"] == "ep-2"
    assert client.get("/api/session/missing").status_code == 404


def test_eval_case_conversation_uses_rich_trace(tmp_path: Path) -> None:
    from ohmo.evals.viewer import eval_case_conversation

    store = get_eval_store(tmp_path)
    _write_eval_report(store)
    _write_rich_eval_trace(store, final_text="нашёл два отзыва")

    data = eval_case_conversation(store, "eval_report_x.json", "case-1")

    assert data["kind"] == "observed"
    assert data["goldEpisodeId"] == "ep-gold"
    assert data["judgeVerdict"] == "fail"
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]
    assert data["messages"][0]["text"] == "покажи отзывы Xander"
    assert data["messages"][1]["text"] == "нашёл два отзыва"
    assert data["messages"][1]["toolCalls"][0]["name"].startswith("bash:maps-cli")

    client = TestClient(create_app(tmp_path))
    route = client.get(
        "/api/eval-conversation/case-1",
        params={"run": "eval_report_x.json"},
    )
    assert route.status_code == 200
    assert route.json()["goldEpisodeId"] == "ep-gold"
    assert client.get("/api/eval-conversation/case-1").status_code == 400


def _append_trace_episode(
    store,
    *,
    episode_id: str = "ep-1",
    base: datetime,
    user_text: str = "посмотри отзывы",
    is_error: bool = False,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id="session-1",
            created_at=base,
            user_text=user_text,
            metadata={"model": "gpt-test", "cwd": "/tmp/project"},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="inbound_message",
            timestamp=base,
            payload={"user_text": user_text},
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_started",
            timestamp=base + timedelta(milliseconds=100),
            payload={"input": {"command": 'maps-cli reviews "Xander"'}},
            tool_name="bash",
            tool_call_id="c1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="tool_completed",
            timestamp=base + timedelta(milliseconds=300),
            payload={"output": "..."},
            tool_name="bash",
            tool_call_id="c1",
            is_error=is_error,
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            timestamp=base + timedelta(milliseconds=500),
            payload={"text": "вот отзывы"},
        )
    )


def _write_eval_report(store) -> None:
    report = {
        "report_id": "report-x",
        "pack_id": "pack-x",
        "created_at": "2026-01-01T12:44:05Z",
        "report_kind": "execution_report",
        "schema_version": 1,
        "case_count": 1,
        "passed_count": 0,
        "failed_count": 1,
        "blocked_count": 0,
        "error_count": 0,
        "metadata": {
            "executor_name": "replay-tools",
            "scorer_name": "trajectory_judge_v1",
            "fixture_match": "order",
            "samples": 3,
            "judge_model": "judge-model",
            "history_model": "history-model",
            "mode": "execution_replay",
            "privacy": "metadata_only",
        },
        "cases": [
            {
                "case_id": "case-1",
                "gold_case_id": "gold-1",
                "status": "failed",
                "score": 0.9,
                "max_score": 1.0,
                "context": {
                    "episode_id": "ep-gold",
                    "capability_path": ["bash:maps-cli reviews", "send_message"],
                    "tool_path": ["bash", "send_message"],
                    "event_kind_path": ["tool_started", "tool_completed"],
                    "status": "open",
                },
                "observed_trace": {
                    "tool_path": ["bash", "send_message"],
                    "event_kind_path": ["tool_started", "tool_completed"],
                    "final_text_length": 42,
                    "final_text_hash": "hash-final",
                    "error_count": 1,
                    "metadata": _decision_trace_summary_fixture(event_count=2),
                    "tool_calls": [
                        {
                            "tool_name": "bash",
                            "is_error": False,
                            "started": True,
                            "completed": True,
                            "start_event_index": 0,
                            "complete_event_index": 1,
                            "input_summary_length": 12,
                            "output_summary_length": 34,
                            "call_key_hash": "call-1",
                        },
                        {
                            "tool_name": "send_message",
                            "is_error": True,
                            "started": True,
                            "completed": True,
                            "start_event_index": 2,
                            "complete_event_index": 3,
                            "input_summary_length": 56,
                            "output_summary_length": 78,
                            "call_key_hash": "call-2",
                        },
                    ],
                },
                "metadata": {
                    "scorer_name": "trajectory_judge_v1",
                    "executor_name": "replay-tools",
                    "pass_count": 1,
                    "sample_count": 3,
                    "pass_rate": 1 / 3,
                    "tool_count": 2,
                    "rubric_count": 1,
                },
                "checks": ["trajectory_judge_v1"],
                "warnings": [],
            }
        ],
    }
    reports_dir = store.root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "eval_report_x.json").write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_rich_eval_trace(
    store,
    *,
    sample_index: int = 0,
    prompt: str = "покажи отзывы Xander",
    final_text: str = "нашёл два отзыва",
    judge_verdict: str = "fail",
    judge_reason: str = "final answer missed one required detail",
    score: float = 0.9,
    passed: bool = False,
) -> None:
    rich_trace = {
        "case_id": "case-1",
        "sample_index": sample_index,
        "prompt": prompt,
        "final_text": final_text,
        "model_calls": [
            {
                "model": "gpt-5.5",
                "input_tokens": 10,
                "output_tokens": 3,
                "started_ms": 900,
                "ended_ms": 1_000,
            },
            {
                "model": "gpt-5.5",
                "input_tokens": 20,
                "output_tokens": 7,
                "started_ms": 1_300,
                "ended_ms": 1_500,
            },
        ],
        "tool_calls": [
            {
                "tool_name": "bash",
                "input": {"command": "maps-cli reviews Xander"},
                "output": "review one\nreview two",
                "is_error": False,
                "started_ms": 1_100,
                "ended_ms": 1_250,
            }
        ],
        "judge": {
            "verdict": judge_verdict,
            "reason": judge_reason,
        },
        "score": score,
        "passed": passed,
        "metadata": _decision_trace_summary_fixture(
            event_count=3,
            recorded_count=1,
            coverage_status="partial",
            sensitivity_labels=["private", "secret"],
            max_sensitivity="secret",
        ),
    }
    trace_dir = store.root / "traces" / "report-x"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"case-1-{sample_index}.json").write_text(
        json.dumps(rich_trace, ensure_ascii=False),
        encoding="utf-8",
    )


def _decision_trace_summary_fixture(
    *,
    event_count: int,
    recorded_count: int = 1,
    coverage_status: str = "complete",
    sensitivity_labels: list[str] | None = None,
    max_sensitivity: str = "private",
) -> dict[str, object]:
    return {
        "decision_trace_event_count": event_count,
        "decision_trace_model_event_count": event_count,
        "decision_trace_diagnostic_event_count": 0,
        "decision_trace_missing_required_count": 0,
        "decision_trace_required_count": 1,
        "decision_trace_recorded_count": recorded_count,
        "decision_trace_coverage_status": coverage_status,
        "unsupported_claim_count": 0,
        "uncertainty_trace_count": 0,
        "uncertainty_status": "absent",
        "decision_trace_sensitivity_labels": sensitivity_labels or ["private"],
        "decision_trace_max_sensitivity": max_sensitivity,
    }

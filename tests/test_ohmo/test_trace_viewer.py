from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from openharness.evals import EvalEpisode, EvalEvent
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
    assert traces["traces"][0]["spansCount"] == 3

    data = eval_case_to_trace_viewer_data(store, "eval_report_x.json", "case-1")
    assert data["goldEpisodeId"] == "ep-gold"
    assert data["badges"] == [
        {"label": "score 0.9"},
        {"label": "failed"},
        {"label": "1/3"},
    ]
    root = data["spans"][0]
    assert root["status"] == "error"
    assert len(root["children"]) == 2
    assert root["children"][1]["status"] == "error"

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


def _write_rich_eval_trace(store) -> None:
    rich_trace = {
        "case_id": "case-1",
        "sample_index": 0,
        "prompt": "покажи отзывы Xander",
        "final_text": "нашёл два отзыва",
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
            "verdict": "fail",
            "reason": "final answer missed one required detail",
        },
        "score": 0.9,
        "passed": False,
    }
    trace_dir = store.root / "traces" / "report-x"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "case-1-0.json").write_text(
        json.dumps(rich_trace, ensure_ascii=False),
        encoding="utf-8",
    )

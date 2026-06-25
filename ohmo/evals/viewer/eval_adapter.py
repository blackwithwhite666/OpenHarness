"""Adapters from ohmo eval reports to trace-viewer DTOs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ohmo.evals.viewer.adapter import category_for

EXECUTION_REPORT_KIND = "execution_report"


def list_eval_runs(store: Any) -> dict[str, Any]:
    """Return execution eval reports in newest-first order."""
    runs: list[dict[str, Any]] = []
    for path in (Path(store.root) / "reports").glob("eval_report_*.json"):
        report = _read_report_or_none(path)
        if report is None:
            continue
        metadata = _mapping(report.get("metadata"))
        runs.append(
            {
                "run": path.name,
                "reportId": _string_value(report.get("report_id")),
                "packId": _string_value(report.get("pack_id")),
                "createdAt": _epoch_ms(report.get("created_at")),
                "scorer": _string_value(metadata.get("scorer_name")),
                "executor": _string_value(metadata.get("executor_name")),
                "samples": _int_value(metadata.get("samples")),
                "caseCount": _int_value(report.get("case_count")),
                "passedCount": _int_value(report.get("passed_count")),
                "failedCount": _int_value(report.get("failed_count")),
            }
        )
    runs.sort(key=lambda item: item["createdAt"], reverse=True)
    return {"runs": runs}


def list_eval_traces(store: Any, run: str) -> dict[str, Any]:
    """Return case summaries for one execution report."""
    report = _read_execution_report(store, run)
    traces = [_case_summary(case) for case in _cases(report)]
    return {"traces": traces, "total": len(traces)}


def eval_case_to_trace_viewer_data(store: Any, run: str, case_id: str) -> dict[str, Any]:
    """Return one eval report case as the JSON-friendly trace viewer DTO."""
    report = _read_execution_report(store, run)
    case = _find_case(report, case_id)
    if case is None:
        raise KeyError(case_id)

    report_metadata = _mapping(report.get("metadata"))
    case_metadata = _mapping(case.get("metadata"))
    context = _mapping(case.get("context"))
    observed_trace = _mapping(case.get("observed_trace"))
    tool_calls = _sequence_of_mappings(observed_trace.get("tool_calls"))
    capability_path = _string_sequence(context.get("capability_path"))
    status = _status_for_case(_string_value(case.get("status")))
    scorer = _first_string(case_metadata.get("scorer_name"), report_metadata.get("scorer_name"))
    executor = _first_string(case_metadata.get("executor_name"), report_metadata.get("executor_name"))
    fixture_match = _string_value(report_metadata.get("fixture_match"))
    pass_count = _int_value(case_metadata.get("pass_count"))
    sample_count = _int_value(case_metadata.get("sample_count"))
    score = case.get("score")
    gold_episode_id = _string_value(context.get("episode_id"))

    child_spans = [
        _tool_call_span(
            case_id=case_id,
            index=index,
            tool_call=tool_call,
            capability=capability_path[index - 1] if index <= len(capability_path) else None,
        )
        for index, tool_call in enumerate(tool_calls, start=1)
    ]

    root_span = {
        "id": case_id,
        "title": case_id,
        "startTimeMs": 0,
        "endTimeMs": len(child_spans),
        "durationMs": 0,
        "type": "agent_invocation",
        "status": status,
        "input": None,
        "output": None,
        "raw": _json_dumps(case),
        "attributes": _root_attributes(
            scorer=scorer,
            executor=executor,
            fixture_match=fixture_match,
            pass_count=pass_count,
            sample_count=sample_count,
            score=score,
            gold_episode_id=gold_episode_id,
        ),
        "children": child_spans,
    }

    return {
        "traceRecord": {
            "id": case_id,
            "name": case_id,
            "spansCount": 1 + len(child_spans),
            "durationMs": 0,
            "agentDescription": _agent_description(executor, scorer),
            "startTimeMs": 0,
        },
        "spans": [root_span],
        "goldEpisodeId": gold_episode_id,
        "badges": [
            {"label": f"score {_label_value(score)}"},
            {"label": _string_value(case.get("status")) or ""},
            {"label": f"{_label_value(pass_count)}/{_label_value(sample_count)}"},
        ],
    }


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    case_metadata = _mapping(case.get("metadata"))
    context = _mapping(case.get("context"))
    observed_trace = _mapping(case.get("observed_trace"))
    tool_calls = _sequence_of_mappings(observed_trace.get("tool_calls"))
    case_id = _string_value(case.get("case_id")) or ""
    return {
        "id": case_id,
        "name": case_id,
        "kind": "eval",
        "status": _status_for_case(_string_value(case.get("status"))),
        "score": case.get("score"),
        "passCount": _int_value(case_metadata.get("pass_count")),
        "sampleCount": _int_value(case_metadata.get("sample_count")),
        "goldEpisodeId": _string_value(context.get("episode_id")),
        "spansCount": 1 + len(tool_calls),
    }


def _tool_call_span(
    *,
    case_id: str,
    index: int,
    tool_call: dict[str, Any],
    capability: str | None,
) -> dict[str, Any]:
    tool_name = _string_value(tool_call.get("tool_name"))
    title = capability or tool_name or f"tool {index}"
    return {
        "id": _string_value(tool_call.get("call_key_hash")) or f"{case_id}:tool:{index}",
        "title": title,
        "startTimeMs": index,
        "endTimeMs": index,
        "durationMs": 0,
        "type": category_for(tool_name),
        "status": "error" if bool(tool_call.get("is_error")) else "success",
        "input": None,
        "output": None,
        "raw": _json_dumps(tool_call),
        "attributes": _tool_attributes(tool_call, ordinal=index),
        "children": [],
    }


def _root_attributes(
    *,
    scorer: str | None,
    executor: str | None,
    fixture_match: str | None,
    pass_count: int,
    sample_count: int,
    score: Any,
    gold_episode_id: str | None,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "scorer", scorer)
    _append_attribute(attributes, "executor", executor)
    _append_attribute(attributes, "fixture_match", fixture_match)
    _append_attribute(attributes, "pass_count", pass_count)
    _append_attribute(attributes, "sample_count", sample_count)
    _append_attribute(attributes, "score", score)
    _append_attribute(attributes, "gold_episode_id", gold_episode_id)
    return attributes


def _tool_attributes(
    tool_call: dict[str, Any],
    *,
    ordinal: int,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "tool_name", tool_call.get("tool_name"))
    _append_attribute(attributes, "input_summary_length", tool_call.get("input_summary_length"))
    _append_attribute(attributes, "output_summary_length", tool_call.get("output_summary_length"))
    _append_attribute(attributes, "ordinal", ordinal)
    _append_attribute(attributes, "start_event_index", tool_call.get("start_event_index"))
    _append_attribute(attributes, "complete_event_index", tool_call.get("complete_event_index"))
    _append_attribute(attributes, "call_key_hash", tool_call.get("call_key_hash"))
    return attributes


def _append_attribute(
    attributes: list[dict[str, dict[str, str] | str]],
    key: str,
    value: Any,
) -> None:
    string_value = _string_value(value)
    if string_value is None:
        return
    attributes.append({"key": key, "value": {"stringValue": string_value}})


def _read_execution_report(store: Any, run: str) -> dict[str, Any]:
    path = _report_path(store, run)
    report = _read_report_or_none(path)
    if report is None:
        raise KeyError(run)
    return report


def _read_report_or_none(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("report_kind") != EXECUTION_REPORT_KIND:
        return None
    return value


def _report_path(store: Any, run: str) -> Path:
    reports_dir = (Path(store.root) / "reports").resolve()
    path = (reports_dir / run).resolve()
    if path.name != run or not _is_relative_to(path, reports_dir):
        raise KeyError(run)
    return path


def _find_case(report: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    for case in _cases(report):
        if case.get("case_id") == case_id:
            return case
    return None


def _cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    return _sequence_of_mappings(report.get("cases"))


def _status_for_case(status: str | None) -> str:
    if status == "passed":
        return "success"
    if status in {"failed", "error"}:
        return "error"
    return "warning"


def _agent_description(executor: str | None, scorer: str | None) -> str:
    if executor and scorer:
        return f"{executor} · {scorer}"
    return executor or scorer or ""


def _epoch_ms(value: Any) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str) or not value:
        return 0
    timestamp = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return 0
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sequence_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_sequence(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _first_string(*values: Any) -> str | None:
    for value in values:
        string_value = _string_value(value)
        if string_value:
            return string_value
    return None


def _string_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _label_value(value: Any) -> str:
    return _string_value(value) or "0"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

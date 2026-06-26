"""Adapters from ohmo eval reports to trace-viewer DTOs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openharness.evals.tool_labels import effective_tool_label
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
    report_metadata = _mapping(report.get("metadata"))
    report_id = _string_value(report.get("report_id"))
    traces = [
        _case_summary(store, report_id, report_metadata, case) for case in _cases(report)
    ]
    return {"traces": traces, "total": len(traces)}


def eval_case_to_trace_viewer_data(
    store: Any,
    run: str,
    case_id: str,
    sample: int = 0,
) -> dict[str, Any]:
    """Return one eval report case as the JSON-friendly trace viewer DTO."""
    report = _read_execution_report(store, run)
    case = _find_case(report, case_id)
    if case is None:
        raise KeyError(case_id)

    report_metadata = _mapping(report.get("metadata"))
    case_metadata = _mapping(case.get("metadata"))
    report_id = _string_value(report.get("report_id"))
    context = _mapping(case.get("context"))
    observed_trace = _mapping(case.get("observed_trace"))
    tool_calls = _sequence_of_mappings(observed_trace.get("tool_calls"))
    capability_path = _string_sequence(context.get("capability_path"))
    status = _status_for_case(_string_value(case.get("status")))
    scorer = _first_string(case_metadata.get("scorer_name"), report_metadata.get("scorer_name"))
    executor = _first_string(case_metadata.get("executor_name"), report_metadata.get("executor_name"))
    fixture_match = _string_value(report_metadata.get("fixture_match"))
    pass_count = _int_value(case_metadata.get("pass_count"))
    sample_count = _first_int(
        case_metadata.get("sample_count"),
        report_metadata.get("sample_count"),
        report_metadata.get("samples"),
    )
    effective_sample_count = _effective_sample_count(
        store,
        report_id=report_id,
        case_id=case_id,
        fallback=sample_count,
    )
    score = case.get("score")
    gold_episode_id = _string_value(context.get("episode_id"))
    badges = _case_badges(case, score=score, pass_count=pass_count, sample_count=sample_count)

    rich_result = _read_rich_trace_or_none(
        store,
        report_id=report_id,
        case_id=case_id,
        sample=sample,
    )
    if rich_result is not None:
        rich_trace, loaded_sample = rich_result
        return _rich_trace_viewer_data(
            case_id=case_id,
            case=case,
            rich_trace=rich_trace,
            sample=loaded_sample,
            sample_count=effective_sample_count,
            status=status,
            scorer=scorer,
            executor=executor,
            fixture_match=fixture_match,
            pass_count=pass_count,
            metadata_sample_count=sample_count,
            score=score,
            gold_episode_id=gold_episode_id,
            badges=badges,
        )

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
        "badges": badges,
        "sample": max(0, sample),
        "sampleCount": effective_sample_count,
    }


def eval_case_conversation(
    store: Any,
    run: str,
    case_id: str,
    sample: int = 0,
) -> dict[str, Any]:
    """Render one eval case's observed turn as a chat transcript.

    The eval is a single user turn: the prompt the agent received and the reply
    it produced, with the tool calls it made along the way. Also surfaces the
    gold episode id so the UI can cross-link to the real-chat session, and the
    judge verdict/reason when present. Raises ``KeyError`` for unknown cases.
    """
    report = _read_execution_report(store, run)
    case = _find_case(report, case_id)
    if case is None:
        raise KeyError(case_id)

    report_id = _string_value(report.get("report_id"))
    context = _mapping(case.get("context"))
    gold_episode_id = _string_value(context.get("episode_id"))
    rich_result = _read_rich_trace_or_none(
        store, report_id=report_id, case_id=case_id, sample=sample
    )

    prompt: str | None = None
    final_text: str | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None
    loaded_sample = max(0, sample)

    if rich_result is not None:
        rich_trace, loaded_sample = rich_result
        prompt = _text_or_json(rich_trace.get("prompt"))
        final_text = _text_or_json(rich_trace.get("final_text"))
        tool_summaries = [
            {
                "name": effective_tool_label(
                    _string_value(tool_call.get("tool_name")) or "",
                    tool_call.get("input"),
                ),
                "status": "error" if bool(tool_call.get("is_error")) else "success",
            }
            for tool_call in _sequence_of_mappings(rich_trace.get("tool_calls"))
        ]
        judge = _mapping(rich_trace.get("judge"))
        judge_verdict = _string_value(judge.get("verdict"))
        judge_reason = _string_value(judge.get("reason"))
    else:
        observed = _mapping(case.get("observed_trace"))
        tool_summaries = [
            {
                "name": _string_value(tool_call.get("tool_name")) or f"tool {index}",
                "status": "error" if bool(tool_call.get("is_error")) else "success",
            }
            for index, tool_call in enumerate(
                _sequence_of_mappings(observed.get("tool_calls")), start=1
            )
        ]

    messages: list[dict[str, Any]] = []
    if prompt:
        messages.append(
            {
                "role": "user",
                "text": prompt,
                "ts": 0,
                "episodeId": None,
                "toolCalls": [],
                "status": "success",
            }
        )
    messages.append(
        {
            "role": "assistant",
            "text": final_text or "",
            "ts": 0,
            "episodeId": None,
            "toolCalls": tool_summaries,
            "status": "error"
            if any(summary["status"] == "error" for summary in tool_summaries)
            else "success",
        }
    )

    return {
        "title": case_id,
        "kind": "observed",
        "sessionId": "",
        "goldEpisodeId": gold_episode_id,
        "sample": loaded_sample,
        "judgeVerdict": judge_verdict,
        "judgeReason": judge_reason,
        "messages": messages,
    }


def _rich_trace_viewer_data(
    *,
    case_id: str,
    case: dict[str, Any],
    rich_trace: dict[str, Any],
    sample: int,
    sample_count: int,
    status: str,
    scorer: str | None,
    executor: str | None,
    fixture_match: str | None,
    pass_count: int,
    metadata_sample_count: int,
    score: Any,
    gold_episode_id: str | None,
    badges: list[dict[str, str]],
) -> dict[str, Any]:
    tool_calls = _sequence_of_mappings(rich_trace.get("tool_calls"))
    model_calls = _sequence_of_mappings(rich_trace.get("model_calls"))
    tool_spans = [
        _rich_tool_call_span(case_id=case_id, index=index, tool_call=tool_call)
        for index, tool_call in enumerate(tool_calls, start=1)
    ]
    model_spans = [
        _rich_model_call_span(case_id=case_id, index=index, model_call=model_call)
        for index, model_call in enumerate(model_calls, start=1)
    ]
    child_spans = sorted(
        [*model_spans, *tool_spans],
        key=lambda span: _int_value(span.get("startTimeMs")),
    )
    start_ms, end_ms = _rich_trace_bounds_ms(child_spans)
    duration_ms = max(0, end_ms - start_ms)
    total_tokens = sum(_model_call_tokens(model_call) for model_call in model_calls)
    judge = _mapping(rich_trace.get("judge"))
    judge_verdict = _string_value(judge.get("verdict"))
    judge_reason = _string_value(judge.get("reason"))
    if judge_verdict:
        badges = [*badges, {"label": f"judge: {judge_verdict}"}]
    root_status = _rich_trace_status(rich_trace, default=status)

    root_span = {
        "id": case_id,
        "title": case_id,
        "startTimeMs": start_ms,
        "endTimeMs": end_ms,
        "durationMs": duration_ms,
        "type": "agent_invocation",
        "status": root_status,
        "input": _text_or_json(rich_trace.get("prompt")),
        "output": _text_or_json(rich_trace.get("final_text")),
        "raw": _json_dumps({"case": case, "trace": rich_trace}),
        "attributes": _root_attributes(
            scorer=scorer,
            executor=executor,
            fixture_match=fixture_match,
            pass_count=pass_count,
            sample_count=metadata_sample_count,
            score=score,
            gold_episode_id=gold_episode_id,
            judge_verdict=judge_verdict,
            judge_reason=judge_reason,
        ),
        "children": child_spans,
    }

    return {
        "traceRecord": {
            "id": case_id,
            "name": case_id,
            "spansCount": 1 + len(child_spans),
            "durationMs": duration_ms,
            "agentDescription": _agent_description(executor, scorer),
            "startTimeMs": start_ms,
            "totalTokens": total_tokens,
        },
        "spans": [root_span],
        "goldEpisodeId": gold_episode_id,
        "badges": badges,
        "sample": sample,
        "sampleCount": sample_count,
    }


def _case_summary(
    store: Any,
    report_id: str | None,
    report_metadata: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    case_metadata = _mapping(case.get("metadata"))
    context = _mapping(case.get("context"))
    observed_trace = _mapping(case.get("observed_trace"))
    tool_calls = _sequence_of_mappings(observed_trace.get("tool_calls"))
    case_id = _string_value(case.get("case_id")) or ""
    metadata_sample_count = _first_int(
        case_metadata.get("sample_count"),
        report_metadata.get("sample_count"),
        report_metadata.get("samples"),
    )
    return {
        "id": case_id,
        "name": case_id,
        "kind": "eval",
        "status": _status_for_case(_string_value(case.get("status"))),
        "score": case.get("score"),
        "passCount": _int_value(case_metadata.get("pass_count")),
        "sampleCount": _effective_sample_count(
            store,
            report_id=report_id,
            case_id=case_id,
            fallback=metadata_sample_count,
        ),
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


def _rich_tool_call_span(
    *,
    case_id: str,
    index: int,
    tool_call: dict[str, Any],
) -> dict[str, Any]:
    tool_name = _string_value(tool_call.get("tool_name")) or ""
    input_value = tool_call.get("input")
    start_ms = _int_value(tool_call.get("started_ms"))
    end_ms = _int_value(tool_call.get("ended_ms"))
    return {
        "id": f"{case_id}:tool:{index}",
        "title": effective_tool_label(tool_name, input_value),
        "startTimeMs": start_ms,
        "endTimeMs": end_ms,
        "durationMs": max(0, end_ms - start_ms),
        "type": category_for(tool_name),
        "status": "error" if bool(tool_call.get("is_error")) else "success",
        "input": _json_or_none(input_value),
        "output": _text_or_json(tool_call.get("output")),
        "raw": _json_dumps(tool_call),
        "attributes": _tool_attributes(tool_call, ordinal=index),
        "children": [],
    }


def _rich_model_call_span(
    *,
    case_id: str,
    index: int,
    model_call: dict[str, Any],
) -> dict[str, Any]:
    model = _string_value(model_call.get("model")) or f"model call {index}"
    input_tokens = _int_value(model_call.get("input_tokens"))
    output_tokens = _int_value(model_call.get("output_tokens"))
    start_ms = _int_value(model_call.get("started_ms"))
    end_ms = _int_value(model_call.get("ended_ms"))
    return {
        "id": f"{case_id}:llm:{index}",
        "title": model,
        "startTimeMs": start_ms,
        "endTimeMs": end_ms,
        "durationMs": max(0, end_ms - start_ms),
        "type": "llm_call",
        "status": "success",
        "tokensCount": input_tokens + output_tokens,
        "input": None,
        "output": None,
        "raw": _json_dumps(model_call),
        "attributes": _model_call_attributes(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
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
    judge_verdict: Any = None,
    judge_reason: Any = None,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "scorer", scorer)
    _append_attribute(attributes, "executor", executor)
    _append_attribute(attributes, "fixture_match", fixture_match)
    _append_attribute(attributes, "pass_count", pass_count)
    _append_attribute(attributes, "sample_count", sample_count)
    _append_attribute(attributes, "score", score)
    _append_attribute(attributes, "gold_episode_id", gold_episode_id)
    _append_attribute(attributes, "judge_verdict", judge_verdict)
    _append_attribute(attributes, "judge_reason", judge_reason)
    return attributes


def _model_call_attributes(
    *,
    input_tokens: int,
    output_tokens: int,
) -> list[dict[str, dict[str, str] | str]]:
    attributes: list[dict[str, dict[str, str] | str]] = []
    _append_attribute(attributes, "input_tokens", input_tokens)
    _append_attribute(attributes, "output_tokens", output_tokens)
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
    value = _read_json_mapping_or_none(path)
    if value is None:
        return None
    if value.get("report_kind") != EXECUTION_REPORT_KIND:
        return None
    return value


def _read_rich_trace_or_none(
    store: Any,
    *,
    report_id: str | None,
    case_id: str,
    sample: int,
) -> tuple[dict[str, Any], int] | None:
    if not report_id:
        return None
    requested_sample = max(0, sample)
    sample_indexes = [requested_sample]
    if requested_sample != 0:
        sample_indexes.append(0)
    for sample_index in sample_indexes:
        path = _rich_trace_path(store, report_id=report_id, case_id=case_id, sample=sample_index)
        if path is None or not path.exists():
            continue
        rich_trace = _read_json_mapping_or_none(path)
        if rich_trace is not None:
            return rich_trace, sample_index
    return None


def _effective_sample_count(
    store: Any,
    *,
    report_id: str | None,
    case_id: str,
    fallback: int,
) -> int:
    rich_samples = _rich_trace_samples(store, report_id=report_id, case_id=case_id)
    if rich_samples:
        return len(rich_samples)
    return fallback


def _rich_trace_samples(
    store: Any,
    *,
    report_id: str | None,
    case_id: str,
) -> list[int]:
    if not report_id:
        return []
    report_dir = _rich_trace_report_dir(store, report_id)
    if report_dir is None or not report_dir.is_dir():
        return []

    prefix = f"{case_id}-"
    suffix = ".json"
    samples: set[int] = set()
    try:
        paths = list(report_dir.rglob(f"*{suffix}"))
    except OSError:
        return []
    for path in paths:
        relative_name = path.relative_to(report_dir).as_posix()
        if (
            not path.is_file()
            or not relative_name.startswith(prefix)
            or not relative_name.endswith(suffix)
        ):
            continue
        sample_text = relative_name[len(prefix) : -len(suffix)]
        try:
            sample = int(sample_text)
        except ValueError:
            continue
        if sample >= 0:
            samples.add(sample)
    return sorted(samples)


def _read_json_mapping_or_none(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def _rich_trace_path(
    store: Any,
    *,
    report_id: str,
    case_id: str,
    sample: int,
) -> Path | None:
    report_dir = _rich_trace_report_dir(store, report_id)
    if report_dir is None:
        return None
    path = (report_dir / f"{case_id}-{sample}.json").resolve()
    if not _is_relative_to(path, report_dir):
        return None
    return path


def _rich_trace_report_dir(store: Any, report_id: str) -> Path | None:
    traces_dir = (Path(store.root) / "traces").resolve()
    path = (traces_dir / report_id).resolve()
    if not _is_relative_to(path, traces_dir):
        return None
    return path


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


def _rich_trace_status(rich_trace: dict[str, Any], *, default: str) -> str:
    passed = rich_trace.get("passed")
    if isinstance(passed, bool):
        return "success" if passed else "error"

    judge = _mapping(rich_trace.get("judge"))
    verdict = (_string_value(judge.get("verdict")) or "").casefold()
    if verdict in {"pass", "passed", "success", "ok"}:
        return "success"
    if verdict in {"fail", "failed", "error"}:
        return "error"
    return default


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


def _rich_trace_bounds_ms(spans: list[dict[str, Any]]) -> tuple[int, int]:
    if not spans:
        return 0, 0
    return (
        min(_int_value(span.get("startTimeMs")) for span in spans),
        max(_int_value(span.get("endTimeMs")) for span in spans),
    )


def _model_call_tokens(model_call: dict[str, Any]) -> int:
    return _int_value(model_call.get("input_tokens")) + _int_value(
        model_call.get("output_tokens")
    )


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


def _first_int(*values: Any) -> int:
    for value in values:
        int_value = _int_value(value)
        if int_value:
            return int_value
    return 0


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


def _case_badges(
    case: dict[str, Any],
    *,
    score: Any,
    pass_count: int,
    sample_count: int,
) -> list[dict[str, str]]:
    return [
        {"label": f"score {_label_value(score)}"},
        {"label": _string_value(case.get("status")) or ""},
        {"label": f"{_label_value(pass_count)}/{_label_value(sample_count)}"},
    ]


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return _json_dumps(value)


def _text_or_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _json_dumps(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openharness.config.settings import Settings
from openharness.evals import (
    DECISION_TRACE_MAX_PAYLOAD_BYTES,
    TRACE_DECISION,
    TRACE_MISSING_REQUIRED,
    DecisionTraceRecorder,
    EvalEpisode,
    EvalStore,
)
from openharness.prompts.context import build_runtime_system_prompt
from openharness.tools import TraceTool, create_default_tool_registry
from openharness.tools.base import ToolExecutionContext
from openharness.tools.trace_tool import (
    DECISION_TRACE_RECORDER_METADATA_KEY,
    TraceToolInput,
)


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "trace_event_id": "trace-1",
    }
    payload.update(overrides)
    return payload


def _store_and_recorder(
    tmp_path: Path,
    *,
    enabled: bool = True,
    episode_id: str = "ep-1",
) -> tuple[EvalStore, DecisionTraceRecorder]:
    store = EvalStore(tmp_path / "evals")
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="test",
            app="openharness",
            session_id="session-1",
            user_text="trace tool",
        )
    )
    return store, DecisionTraceRecorder(store=store, episode_id=episode_id, enabled=enabled)


def _context(tmp_path: Path, recorder: DecisionTraceRecorder | None) -> ToolExecutionContext:
    metadata: dict[str, object] = {}
    if recorder is not None:
        metadata[DECISION_TRACE_RECORDER_METADATA_KEY] = recorder
    return ToolExecutionContext(cwd=tmp_path, metadata=metadata)


@pytest.mark.asyncio
async def test_trace_tool_records_valid_event_to_eval_store(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    result = await TraceTool().execute(
        TraceToolInput(
            kind=TRACE_DECISION,
            payload=_payload(decision="use grep before editing", evidence_ids=["search-1"]),
        ),
        _context(tmp_path, recorder),
    )

    assert result.is_error is False
    assert "Recorded decision trace event: trace_decision" in result.output
    [event] = list(store.iter_events("ep-1"))
    assert event.kind == TRACE_DECISION
    assert event.payload["decision"] == "use grep before editing"
    assert event.payload["evidence_ids"] == ["search-1"]
    assert event.payload["sensitivity"] == "private"
    assert event.payload["retention"] == "durable"


@pytest.mark.asyncio
async def test_trace_tool_missing_recorder_is_noop(tmp_path: Path) -> None:
    result = await TraceTool().execute(
        TraceToolInput(kind=TRACE_DECISION, payload=_payload(decision="nothing to record")),
        _context(tmp_path, None),
    )

    assert result.is_error is False
    assert "unavailable" in result.output


@pytest.mark.asyncio
async def test_trace_tool_disabled_recorder_is_noop(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path, enabled=False)
    result = await TraceTool().execute(
        TraceToolInput(kind=TRACE_DECISION, payload=_payload(decision="skip recording")),
        _context(tmp_path, recorder),
    )

    assert result.is_error is False
    assert "disabled" in result.output
    assert store.count_events("ep-1") == 0


@pytest.mark.asyncio
async def test_trace_tool_rejects_unsupported_kind(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path)
    arguments = TraceToolInput.model_construct(
        kind=TRACE_MISSING_REQUIRED,
        payload=_payload(missing=["trace_finalization"]),
    )

    result = await TraceTool().execute(arguments, _context(tmp_path, recorder))

    assert result.is_error is True
    assert "Unsupported model-authored decision trace kind" in result.output
    assert store.count_events("ep-1") == 0


@pytest.mark.asyncio
async def test_trace_tool_refuses_obvious_secret_payload(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path)

    result = await TraceTool().execute(
        TraceToolInput(
            kind=TRACE_DECISION,
            payload=_payload(api_key="a" * 24),
        ),
        _context(tmp_path, recorder),
    )

    assert result.is_error is True
    assert "obvious secret" in result.output
    assert store.count_events("ep-1") == 0


@pytest.mark.asyncio
async def test_trace_tool_refuses_oversized_payload(tmp_path: Path) -> None:
    store, recorder = _store_and_recorder(tmp_path)

    result = await TraceTool().execute(
        TraceToolInput(
            kind=TRACE_DECISION,
            payload=_payload(details="x" * (DECISION_TRACE_MAX_PAYLOAD_BYTES + 1)),
        ),
        _context(tmp_path, recorder),
    )

    assert result.is_error is True
    assert "exceeds" in result.output
    assert store.count_events("ep-1") == 0


def test_default_tool_registry_includes_trace() -> None:
    registry = create_default_tool_registry()

    assert isinstance(registry.get("trace"), TraceTool)


def test_runtime_prompt_includes_trace_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_COORDINATOR_MODE", raising=False)

    prompt = build_runtime_system_prompt(
        Settings(system_prompt="BASE"),
        cwd=tmp_path,
        include_project_memory=False,
    )

    assert "# Decision Trace" in prompt
    assert "`trace` tool" in prompt
    assert "chain-of-thought" in prompt
    assert "artifact paths" in prompt

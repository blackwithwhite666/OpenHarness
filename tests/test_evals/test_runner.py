from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import (
    ConversationMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    EvalExecutionContext,
    EvalExecutionScorerResult,
    EvalExecutorResult,
    EvalObservedCall,
    EvalResource,
    EvalResourceSnapshot,
    EvalRunPack,
    EvalRunPackCase,
    EvalStore,
    EvalToolFixture,
    QueryEngineEvalAgentRunner,
    ReplayToolsExecutor,
    TrajectoryJudgeScorer,
    build_case_candidates,
    build_case_drafts,
    build_replay_tool_registry,
    promote_case_drafts,
    resolve_execution_scorer,
    run_execution_report,
    run_replay_report,
    write_case_draft_pack,
    write_run_pack,
)
from openharness.evals.execution import HistoryContext, _session_conversation_history
from openharness.evals.executor import _run_query_engine_replay
from openharness.tools.base import ToolRegistry


def test_replay_report_reconstructs_metadata_context_without_private_text(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private replay request",
        final_text="private replay answer",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_replay_report(store, pack=pack)

    assert result.relative_path == "reports/eval_report.json"
    assert result.report.report_kind == "metadata_replay_report"
    assert result.report.case_count == 1
    assert result.report.passed_count == 1
    assert result.report.failed_count == 0
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.context is not None
    assert case.context.episode_id == "ep-1"
    assert case.context.event_kind_path == [
        "tool_started",
        "tool_completed",
        "gateway_final",
    ]
    assert case.context.tool_path == ["web_fetch"]
    assert case.context.input_facet_kinds == ["user_request"]
    assert "assistant_final" in case.context.expected_facet_kinds

    serialized = result.path.read_text(encoding="utf-8")
    assert "private replay request" not in serialized
    assert "private replay answer" not in serialized


def test_replay_report_fails_unresolvable_refs(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="cases/gold_cases.jsonl",
        cases=[
            EvalRunPackCase(
                gold_case_id="gold-1",
                case_id="case-1",
                episode_id="missing-episode",
                case_kind="tool_workflow",
                input_facet_ids=["missing-input"],
                expected_facet_ids=["missing-output"],
                tool_names=["missing_tool"],
                rubric=["must work"],
            )
        ],
    )

    result = run_replay_report(store, pack=pack)

    assert result.report.passed_count == 0
    assert result.report.failed_count == 1
    case = result.report.cases[0]
    assert case.status == "failed"
    assert case.context is None
    assert case.warnings == [
        "episode_exists",
        "has_events",
        "input_facets_resolve",
        "expected_facets_resolve",
        "tool_refs_observed",
    ]


def test_replay_report_validates_limit_empty_pack_and_paths(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    pack = EvalRunPack(pack_id="pack-1", source_records_path="cases/gold_cases.jsonl")

    with pytest.raises(ValueError, match="limit must be positive"):
        run_replay_report(store, pack=pack, limit=0)
    with pytest.raises(ValueError, match="eval pack must contain cases"):
        run_replay_report(store, pack=pack)
    with pytest.raises(ValueError, match="store.root/reports"):
        run_replay_report(
            store,
            pack=EvalRunPack(
                pack_id="pack-1",
                source_records_path="cases/gold_cases.jsonl",
                cases=[
                    EvalRunPackCase(
                        gold_case_id="gold",
                        case_id="case",
                        episode_id="ep",
                        case_kind="conversation_replay",
                    )
                ],
            ),
            report_filename="../eval_report.json",
        )


def test_replay_report_limit_changes_case_count_and_report_id(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(store, episode_id="ep-1", user_text="private one", final_text="private final")
    _add_episode(store, episode_id="ep-2", user_text="private two", final_text="private final")
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store)
    pack = write_run_pack(store).pack

    full = run_replay_report(store, pack=pack).report
    limited = run_replay_report(store, pack=pack, limit=1).report

    assert full.case_count == 2
    assert limited.case_count == 1
    assert full.report_id != limited.report_id


def test_execution_report_replay_tools_writes_observed_trace_without_private_text(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private execution request",
        final_text="private execution answer",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(store, pack=pack)

    assert result.relative_path == "reports/eval_report.json"
    assert result.report.report_kind == "execution_report"
    assert result.report.metadata["executor_name"] == "replay-tools"
    assert result.report.passed_count == 1
    assert result.report.failed_count == 0
    assert result.report.blocked_count == 0
    assert result.report.error_count == 0
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.observed_trace is not None
    assert case.observed_trace.executor_name == "replay-tools"
    assert case.observed_trace.tool_path == ["web_fetch"]
    assert case.observed_trace.tool_calls[0].started is True
    assert case.observed_trace.tool_calls[0].completed is True
    assert case.observed_trace.tool_calls[0].input_summary_length == len(
        "private tool input"
    )
    assert case.checks["tool_sequence_matches"] is True
    assert case.checks["final_output_matches"] is True
    assert case.metadata["scorer_name"] == "exact-final-text"
    assert "sample_count" not in case.metadata
    assert result.report.metadata["scorer_name"] == "exact-final-text"
    assert result.report.metadata["samples"] == 1

    serialized = result.path.read_text(encoding="utf-8")
    assert "private execution request" not in serialized
    assert "private execution answer" not in serialized
    assert "private raw tool input" not in serialized
    assert "private raw tool output" not in serialized
    assert "private tool input" not in serialized
    assert "private tool output" not in serialized


def test_execution_report_schema_contract_is_stable(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private schema request",
        final_text="private schema answer",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(store, pack=pack)

    assert result.report.report_kind == "execution_report"
    assert result.report.schema_version == 1
    assert result.report.metadata["privacy"] == "metadata_only"
    assert result.report.metadata["mode"] == "execution_replay"
    assert result.report.metadata["executor_name"] == "replay-tools"
    assert result.report.metadata["scorer_name"] == "exact-final-text"
    assert result.report.metadata["score_schema_version"] == 1
    case = result.report.cases[0]
    assert set(case.checks) == {
        "episode_exists",
        "has_events",
        "input_facets_resolve",
        "expected_facets_resolve",
        "tool_fixtures_resolve",
        "tool_trace_complete",
        "resource_snapshot_valid_or_absent",
        "replay_inputs_recoverable",
        "has_rubric",
        "execution_completed",
        "tool_sequence_matches",
        "final_output_matches",
        "privacy_report_metadata_only",
    }
    assert case.status in {"passed", "failed", "blocked", "error"}
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["tool_calls_source"] == "replay_fixtures"
    assert case.observed_trace.metadata["scorer_name"] == "exact-final-text"
    assert "final_text_hash" in case.observed_trace.model_dump()
    assert "final_text_length" in case.observed_trace.model_dump()


def test_execution_report_accepts_custom_metadata_only_scorer(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private scorer request",
        final_text="private scorer answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(
        store,
        pack=pack,
        executor=_MismatchExecutor(),
        scorer=_AlwaysPassScorer(),
    )

    assert result.report.passed_count == 1
    assert result.report.metadata["scorer_name"] == "always-pass"
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.checks["final_output_matches"] is True
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["scorer_name"] == "always-pass"
    assert case.observed_trace.metadata["scorer_metadata_key_count"] == 1
    serialized = result.path.read_text(encoding="utf-8")
    assert "private scorer request" not in serialized
    assert "private scorer answer" not in serialized
    assert "private scorer note" not in serialized


def test_execution_report_query_engine_runner_uses_reconstructed_prompt_and_replay_tools(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private model prompt",
        final_text="model final from replayed tool",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    api_client = _RecordingReplayApiClient()

    result = run_execution_report(
        store,
        pack=pack,
        executor=ReplayToolsExecutor(
            agent_runner=QueryEngineEvalAgentRunner(
                api_client=api_client,
                model="eval-model",
                system_prompt="eval system",
                cwd=tmp_path,
            )
        ),
    )

    assert result.report.passed_count == 1
    assert len(api_client.requests) == 2
    assert api_client.requests[0].messages[0].text == "private model prompt"
    assert api_client.requests[0].tools[0]["name"] == "web_fetch"
    tool_result_blocks = [
        block
        for message in api_client.requests[1].messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert tool_result_blocks[0].content == '{"text": "private raw tool output"}'
    case = result.report.cases[0]
    assert case.observed_trace is not None
    assert case.observed_trace.tool_path == ["web_fetch"]
    assert case.checks["final_output_matches"] is True

    serialized = result.path.read_text(encoding="utf-8")
    assert "private model prompt" not in serialized
    assert "private raw tool output" not in serialized
    assert "model final from replayed tool" not in serialized


def test_execution_report_query_engine_writes_rich_trace_and_keeps_report_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_judge_reason = "private raw judge reason"

    def _make_pack(root: Path) -> tuple[EvalStore, EvalRunPack]:
        store = EvalStore(root)
        _add_episode(
            store,
            episode_id="ep-1",
            user_text="private trace prompt",
            final_text="model final from replayed tool",
            tool_name="web_fetch",
        )
        drafts = build_case_drafts(store, build_case_candidates(store))
        write_case_draft_pack(store, drafts)
        promote_case_drafts(store, case_ids=[drafts[0].case_id])
        return store, write_run_pack(store).pack

    def _run(store: EvalStore, pack: EvalRunPack):
        return run_execution_report(
            store,
            pack=pack,
            executor=ReplayToolsExecutor(
                agent_runner=QueryEngineEvalAgentRunner(
                    api_client=_RecordingReplayApiClient(),
                    model="eval-model",
                    system_prompt="eval system",
                    cwd=tmp_path,
                )
            ),
            scorer=TrajectoryJudgeScorer(
                api_client=_FinalOnlyModelApiClient(
                    final_text=f"PASS {raw_judge_reason}"
                ),
                model="judge-model",
            ),
        )

    monkeypatch.delenv("OHMO_EVALS_TRACE_CAPTURE", raising=False)
    store, pack = _make_pack(tmp_path / "evals-enabled")
    result = _run(store, pack)

    trace_path = (
        store.root
        / "traces"
        / result.report.report_id
        / f"{pack.cases[0].case_id}-0.json"
    )
    assert trace_path.exists()
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["case_id"] == pack.cases[0].case_id
    assert trace["sample_index"] == 0
    assert trace["prompt"] == "private trace prompt"
    assert trace["final_text"] == "model final from replayed tool"
    assert trace["judge"] == {"verdict": "pass", "reason": raw_judge_reason}
    assert trace["score"] == 1.0
    assert trace["passed"] is True
    assert trace["tool_calls"] == [
        {
            "tool_name": "web_fetch",
            "input": {"query": "private raw tool input"},
            "output": '{"text": "private raw tool output"}',
            "is_error": False,
            "started_ms": trace["tool_calls"][0]["started_ms"],
            "ended_ms": trace["tool_calls"][0]["ended_ms"],
        }
    ]
    assert isinstance(trace["tool_calls"][0]["started_ms"], int)
    assert isinstance(trace["tool_calls"][0]["ended_ms"], int)
    assert trace["tool_calls"][0]["ended_ms"] >= trace["tool_calls"][0]["started_ms"]

    serialized_report = result.path.read_text(encoding="utf-8")
    assert raw_judge_reason not in serialized_report
    assert "raw_judge_reason" not in serialized_report

    monkeypatch.setenv("OHMO_EVALS_TRACE_CAPTURE", "0")
    disabled_store, disabled_pack = _make_pack(tmp_path / "evals-disabled")
    disabled = _run(disabled_store, disabled_pack)
    disabled_trace_path = (
        disabled_store.root
        / "traces"
        / disabled.report.report_id
        / f"{disabled_pack.cases[0].case_id}-0.json"
    )
    assert not disabled_trace_path.exists()


def test_session_conversation_history_falls_back_to_raw_and_filters_same_session_turns(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _add_session_episode(
        store,
        episode_id="ep-opt-1",
        session_id="session-1",
        app="ohmo",
        created_at=base,
        user_text="first user",
        final_text="first assistant",
    )
    _add_session_episode(
        store,
        episode_id="ep-other-session",
        session_id="session-2",
        app="ohmo",
        created_at=base + timedelta(seconds=1),
        user_text="excluded user",
        final_text="excluded assistant",
    )
    _add_session_episode(
        store,
        episode_id="ep-other-app",
        session_id="session-1",
        app="other",
        created_at=base + timedelta(seconds=2),
        user_text="excluded app user",
        final_text="excluded app assistant",
    )
    _add_session_episode(
        store,
        episode_id="ep-opt-2",
        session_id="session-1",
        app="ohmo",
        created_at=base + timedelta(seconds=3),
        user_text="second user",
        final_text="second assistant",
    )
    _add_session_episode(
        store,
        episode_id="ep-opt-target",
        session_id="session-1",
        app="ohmo",
        created_at=base + timedelta(seconds=4),
        user_text="current user",
        final_text="current assistant",
    )

    first = store.get_episode("ep-opt-1")
    current = store.get_episode("ep-opt-target")
    client = _HistorySegmentApiClient('{"start_index": 0}')
    history_context = HistoryContext(api_client=client, model="history-model")

    assert first is not None
    assert current is not None
    # First turn has no prior same-session turns -> empty regardless of segmenter.
    assert _session_conversation_history(store, first) == ()
    # Without an LLM segmenter, history falls back to the raw recent turns in the
    # window (filtered to the same session+app), instead of being silently empty.
    assert _session_conversation_history(store, current) == (
        ("user", "first user"),
        ("assistant", "first assistant"),
        ("user", "second user"),
        ("assistant", "second assistant"),
    )
    # With a segmenter, the start index is honored (here start_index=0 -> same).
    assert _session_conversation_history(
        store,
        current,
        history_context=history_context,
    ) == (
        ("user", "first user"),
        ("assistant", "first assistant"),
        ("user", "second user"),
        ("assistant", "second assistant"),
    )
    assert len(client.requests) == 1


def test_session_conversation_history_caps_to_recent_messages_and_chars(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(6):
        _add_session_episode(
            store,
            episode_id=f"ep-caps-{index}",
            session_id="session-1",
            app="ohmo",
            created_at=base + timedelta(seconds=index),
            user_text=f"user-{index}",
            final_text=f"assistant-{index}",
        )
    _add_session_episode(
        store,
        episode_id="ep-caps-target",
        session_id="session-1",
        app="ohmo",
        created_at=base + timedelta(seconds=6),
        user_text="target user",
        final_text="target assistant",
    )
    target = store.get_episode("ep-caps-target")
    history_context = HistoryContext(
        api_client=_HistorySegmentApiClient('{"start_index": 0}'),
        model="history-model",
    )

    assert target is not None
    assert _session_conversation_history(
        store,
        target,
        history_context=history_context,
        max_messages=4,
    ) == (
        ("user", "user-4"),
        ("assistant", "assistant-4"),
        ("user", "user-5"),
        ("assistant", "assistant-5"),
    )
    assert _session_conversation_history(
        store,
        target,
        history_context=history_context,
        max_messages=20,
        max_chars=len("user-5") + len("assistant-5"),
    ) == (
        ("user", "user-5"),
        ("assistant", "assistant-5"),
    )


def test_session_conversation_history_applies_24h_window(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    target_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
    _add_session_episode(
        store,
        episode_id="ep-window-old",
        session_id="session-window",
        app="ohmo",
        created_at=target_time - timedelta(hours=25),
        user_text="old user",
        final_text="old assistant",
    )
    _add_session_episode(
        store,
        episode_id="ep-window-near",
        session_id="session-window",
        app="ohmo",
        created_at=target_time - timedelta(hours=2),
        user_text="near user",
        final_text="near assistant",
    )
    _add_session_episode(
        store,
        episode_id="ep-window-target",
        session_id="session-window",
        app="ohmo",
        created_at=target_time,
        user_text="target user",
        final_text="target assistant",
    )
    target = store.get_episode("ep-window-target")
    client = _HistorySegmentApiClient('{"start_index": 0}')

    assert target is not None
    assert _session_conversation_history(
        store,
        target,
        history_context=HistoryContext(api_client=client, model="history-model"),
    ) == (
        ("user", "near user"),
        ("assistant", "near assistant"),
    )
    prompt = client.requests[0].messages[0].text
    assert "near user" in prompt
    assert "old user" not in prompt


def test_session_conversation_history_uses_segment_start_index(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(3):
        _add_session_episode(
            store,
            episode_id=f"ep-segment-{index}",
            session_id="session-segment",
            app="ohmo",
            created_at=base + timedelta(seconds=index),
            user_text=f"user-{index}",
            final_text=f"assistant-{index}",
        )
    _add_session_episode(
        store,
        episode_id="ep-segment-target",
        session_id="session-segment",
        app="ohmo",
        created_at=base + timedelta(seconds=3),
        user_text="target user",
        final_text="target assistant",
    )
    target = store.get_episode("ep-segment-target")

    assert target is not None
    assert _session_conversation_history(
        store,
        target,
        history_context=HistoryContext(
            api_client=_HistorySegmentApiClient('{"start_index": 1}'),
            model="history-model",
        ),
    ) == (
        ("user", "user-1"),
        ("assistant", "assistant-1"),
        ("user", "user-2"),
        ("assistant", "assistant-2"),
    )


@pytest.mark.parametrize(
    ("response", "target_suffix"),
    [
        ('{"start_index": null}', "null"),
        ("not json", "junk"),
        (RuntimeError("segment failed"), "raising"),
    ],
)
def test_session_conversation_history_fails_closed_on_segment_failure(
    tmp_path: Path,
    response: str | Exception,
    target_suffix: str,
):
    store = EvalStore(tmp_path / "evals")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _add_session_episode(
        store,
        episode_id=f"ep-fail-prior-{target_suffix}",
        session_id=f"session-fail-{target_suffix}",
        app="ohmo",
        created_at=base,
        user_text="prior user",
        final_text="prior assistant",
    )
    target_id = f"ep-fail-target-{target_suffix}"
    _add_session_episode(
        store,
        episode_id=target_id,
        session_id=f"session-fail-{target_suffix}",
        app="ohmo",
        created_at=base + timedelta(seconds=1),
        user_text="target user",
        final_text="target assistant",
    )
    target = store.get_episode(target_id)

    assert target is not None
    assert _session_conversation_history(
        store,
        target,
        history_context=HistoryContext(
            api_client=_HistorySegmentApiClient(response),
            model="history-model",
        ),
    ) == ()


def test_session_conversation_history_caps_candidate_window_to_last_40_turns(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(45):
        _add_session_episode(
            store,
            episode_id=f"ep-window-cap-{index:02d}",
            session_id="session-window-cap",
            app="ohmo",
            created_at=base + timedelta(minutes=index),
            user_text=f"user-{index:02d}",
            final_text=f"assistant-{index:02d}",
        )
    _add_session_episode(
        store,
        episode_id="ep-window-cap-target",
        session_id="session-window-cap",
        app="ohmo",
        created_at=base + timedelta(minutes=45),
        user_text="target user",
        final_text="target assistant",
    )
    target = store.get_episode("ep-window-cap-target")
    client = _HistorySegmentApiClient('{"start_index": 0}')

    assert target is not None
    _session_conversation_history(
        store,
        target,
        history_context=HistoryContext(api_client=client, model="history-model"),
        max_messages=100,
    )
    prompt = client.requests[0].messages[0].text
    assert prompt.count("user: user-") == 40
    assert "user-04" not in prompt
    assert "user-05" in prompt
    assert "user-44" in prompt


@pytest.mark.asyncio
async def test_run_query_engine_replay_seeds_conversation_history(tmp_path: Path):
    api_client = _FinalOnlyModelApiClient(final_text="seeded final")
    context = _query_replay_context(
        tmp_path,
        conversation_history=(
            ("user", "prior user"),
            ("assistant", "prior assistant"),
        ),
    )

    result = await _run_query_engine_replay(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        max_turns=1,
        max_tokens=128,
        prompt="current turn",
        tool_registry=ToolRegistry(),
        context=context,
    )

    assert api_client.message_snapshots[0][:3] == [
        ("user", "prior user"),
        ("assistant", "prior assistant"),
        ("user", "current turn"),
    ]
    assert result.final_text == "seeded final"
    assert result.metadata["seeded_history_message_count"] == 2


@pytest.mark.asyncio
async def test_run_query_engine_replay_empty_history_submits_only_prompt(
    tmp_path: Path,
):
    api_client = _FinalOnlyModelApiClient(final_text="plain final")
    context = _query_replay_context(tmp_path, conversation_history=())

    result = await _run_query_engine_replay(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        max_turns=1,
        max_tokens=128,
        prompt="current turn",
        tool_registry=ToolRegistry(),
        context=context,
    )

    assert api_client.message_snapshots[0][0] == ("user", "current turn")
    assert result.final_text == "plain final"
    assert result.metadata["seeded_history_message_count"] == 0


@pytest.mark.asyncio
async def test_run_query_engine_replay_keeps_partial_answer_on_max_turns(
    tmp_path: Path,
):
    """A truncated run keeps its last *non-empty* assistant text, not "".

    Truncation usually hits mid-tool-loop, so the very last turn is a tool call
    with empty text; the executor falls back to the last real partial answer.
    Blanking it made "turn budget too small" indistinguishable from "model said
    nothing" — both became an empty auto-fail. The run stays flagged.
    """
    api_client = _PartialThenLoopApiClient(
        partial_text="partial progress answer", command="maps-cli search x"
    )
    context = _query_replay_context(tmp_path, conversation_history=())

    result = await _run_query_engine_replay(
        api_client=api_client,
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        max_turns=2,
        max_tokens=128,
        prompt="current turn",
        tool_registry=build_replay_tool_registry(
            (
                EvalToolFixture(
                    tool_name="bash",
                    call_key_hash="fixture-bash",
                    output_text="replayed bash output",
                ),
            )
        ),
        context=context,
    )

    assert "max_turns_exceeded" in result.event_kind_path
    assert result.metadata.get("max_turns_exceeded") is True
    # final turn was a tool call (empty text) -> fall back to the real partial
    assert result.final_text == "partial progress answer"


def test_query_engine_capability_oracle_gates_command_regression(tmp_path: Path):
    """The query-engine runner + capability oracle is a real model-regression gate.

    On a shell-routed agent every capability runs through ``bash``, so the
    name-based ``tool_trace_oracle_v1`` is blind to a capability regression
    (right tool, wrong command). ``capability_trace_oracle_v1`` judges the
    effective capability and catches it. This proves the gate is not just
    golden-sanity.
    """
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private weather request",
        final_text="private weather answer",
        tool_name="bash",
        tool_input={"command": "weather-cli forecast 'СПб'"},
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    assert pack.cases[0].capability_path == ["bash:weather-cli forecast"]
    assert pack.cases[0].tool_names == ["bash"]

    def _run(
        *,
        command: str,
        scorer: str,
        report_filename: str,
        match_mode: str = "order",
        api_client: _ScriptedBashModelApiClient | None = None,
    ):
        return run_execution_report(
            store,
            pack=pack,
            report_filename=report_filename,
            executor=ReplayToolsExecutor(
                agent_runner=QueryEngineEvalAgentRunner(
                    api_client=api_client
                    or _ScriptedBashModelApiClient(
                        command=command,
                        final_text="private weather answer",
                    ),
                    model="eval-model",
                    system_prompt="eval system",
                    cwd=tmp_path,
                ),
                match_mode=match_mode,
            ),
            scorer=resolve_execution_scorer(scorer),
        )

    # 1. Correct capability — the model calls the expected command → passes.
    correct = _run(
        command="weather-cli forecast 'СПб'",
        scorer="capability_trace_oracle_v1",
        report_filename="eval_report_correct.json",
    )
    assert correct.report.passed_count == 1

    # 2. Capability regression — right tool (bash), wrong command → fails the
    #    capability oracle, even though the tool sequence still matches.
    regression = _run(
        command="python3 -c 'print(2+2)'",
        scorer="capability_trace_oracle_v1",
        report_filename="eval_report_regress.json",
    )
    assert regression.report.passed_count == 0
    assert regression.report.failed_count == 1
    case = regression.report.cases[0]
    assert case.checks["final_output_matches"] is False
    assert case.checks["tool_sequence_matches"] is True
    assert case.observed_trace is not None
    assert "missing_capabilities" in case.observed_trace.metadata
    assert "observed_capabilities" in case.observed_trace.metadata
    assert case.observed_trace.metadata["missing_capabilities"]
    assert case.observed_trace.metadata["observed_capabilities"] == ["bash:python3"]

    # 3. Blind-spot contrast — the SAME regression passes the name-based oracle,
    #    because the observed tool name is still "bash". This is the punchline.
    blind = _run(
        command="python3 -c 'print(2+2)'",
        scorer="tool_trace_oracle_v1",
        report_filename="eval_report_blind.json",
    )
    assert blind.report.passed_count == 1

    # 4. Argument-matched replay turns the same divergence into an honest tool
    #    error instead of returning the next captured output by order.
    arg_api_client = _ScriptedBashModelApiClient(
        command="python3 -c 'print(2+2)'",
        final_text="private weather answer",
    )
    arg_matched = _run(
        command="python3 -c 'print(2+2)'",
        scorer="tool_trace_oracle_v1",
        report_filename="eval_report_arg_match.json",
        match_mode="arguments",
        api_client=arg_api_client,
    )
    assert arg_matched.report.metadata["fixture_match"] == "arguments"
    assert arg_matched.report.passed_count == 0
    assert arg_matched.report.failed_count == 1
    arg_case = arg_matched.report.cases[0]
    assert arg_case.checks["tool_sequence_matches"] is True
    assert arg_case.checks["final_output_matches"] is False
    assert arg_case.observed_trace is not None
    assert "tool_completed_error" in arg_case.observed_trace.event_kind_path
    tool_result_blocks = [
        block
        for message in arg_api_client.requests[1].messages
        for block in message.content
        if isinstance(block, ToolResultBlock)
    ]
    assert tool_result_blocks[0].is_error is True
    assert "No replay fixture for bash with these arguments." in tool_result_blocks[0].content
    assert "private raw tool output" not in tool_result_blocks[0].content

    # Reports stay metadata-only: no raw commands or user/final text leak.
    for write in (correct, regression, blind, arg_matched):
        serialized = write.path.read_text(encoding="utf-8")
        assert "weather-cli forecast 'СПб'" not in serialized
        assert "python3 -c 'print(2+2)'" not in serialized
        assert "'СПб'" not in serialized
        assert "print(2+2)" not in serialized
        assert "private weather request" not in serialized
        assert "private weather answer" not in serialized


def test_query_engine_max_turns_exceeded_is_scored_failure(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private max turns request",
        final_text="private max turns answer",
        tool_name="bash",
        tool_input={"command": "maps-cli search x"},
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    runner = QueryEngineEvalAgentRunner(
        api_client=_LoopingBashModelApiClient(command="maps-cli search x"),
        model="eval-model",
        system_prompt="eval system",
        cwd=tmp_path,
        max_turns=2,
    )
    direct_result = runner.run(
        prompt="private max turns request",
        tool_registry=build_replay_tool_registry(
            (
                EvalToolFixture(
                    tool_name="bash",
                    call_key_hash="fixture-bash",
                    output_text="replayed bash output",
                ),
            )
        ),
        context=SimpleNamespace(events=()),
    )

    assert direct_result.metadata["max_turns_exceeded"] is True
    assert direct_result.final_text == ""
    assert direct_result.tool_path
    assert "max_turns_exceeded" in direct_result.event_kind_path

    result = run_execution_report(
        store,
        pack=pack,
        executor=ReplayToolsExecutor(
            agent_runner=QueryEngineEvalAgentRunner(
                api_client=_LoopingBashModelApiClient(command="maps-cli search x"),
                model="eval-model",
                system_prompt="eval system",
                cwd=tmp_path,
                max_turns=2,
            ),
        ),
    )

    assert result.report.failed_count == 1
    assert result.report.error_count == 0
    case = result.report.cases[0]
    assert case.status == "failed"
    assert case.observed_trace is not None
    assert "max_turns_exceeded" in case.observed_trace.event_kind_path


def test_execution_report_executor_gets_transient_text_but_report_omits_it(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private transient request",
        final_text="private transient answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    executor = _CapturingExecutor()

    result = run_execution_report(store, pack=pack, executor=executor)

    assert executor.seen_prompt == "private transient request"
    assert executor.seen_expected == "private transient answer"
    assert result.report.passed_count == 1
    serialized = result.path.read_text(encoding="utf-8")
    assert "private transient request" not in serialized
    assert "private transient answer" not in serialized
    assert "private executor metadata" not in serialized


def test_execution_report_sanitizes_untrusted_executor_trace_labels(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private label request",
        final_text="private label answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(store, pack=pack, executor=_LeakyLabelExecutor())

    case = result.report.cases[0]
    assert case.status == "failed"
    assert case.observed_trace is not None
    assert case.observed_trace.event_kind_path[0].startswith("event:")
    assert case.observed_trace.tool_path[0].startswith("tool:")
    serialized = result.path.read_text(encoding="utf-8")
    assert "private label request" not in serialized
    assert "private label answer" not in serialized


def test_execution_report_marks_behavior_mismatch_failed(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private mismatch request",
        final_text="private mismatch answer",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(store, pack=pack, executor=_MismatchExecutor())

    assert result.report.passed_count == 0
    assert result.report.failed_count == 1
    assert result.report.blocked_count == 0
    assert result.report.error_count == 0
    case = result.report.cases[0]
    assert case.status == "failed"
    assert case.checks["tool_sequence_matches"] is False
    assert case.checks["final_output_matches"] is False
    assert case.observed_trace is not None
    assert case.observed_trace.tool_path == []


def test_execution_report_samples_majority_passes(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private sampled pass request",
        final_text="private sampled pass answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    executor = _SequenceExecutor([True, False, True])

    result = run_execution_report(store, pack=pack, executor=executor, samples=3)

    assert executor.call_count == 3
    assert result.report.passed_count == 1
    assert result.report.failed_count == 0
    assert result.report.metadata["samples"] == 3
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.metadata["sample_count"] == 3
    assert case.metadata["pass_count"] == 2
    assert case.metadata["pass_rate"] == pytest.approx(2 / 3)
    assert case.checks["final_output_matches"] is True


def test_execution_report_samples_majority_fails(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private sampled fail request",
        final_text="private sampled fail answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    executor = _SequenceExecutor([False, True, False])

    result = run_execution_report(store, pack=pack, executor=executor, samples=3)

    assert executor.call_count == 3
    assert result.report.passed_count == 0
    assert result.report.failed_count == 1
    case = result.report.cases[0]
    assert case.status == "failed"
    assert case.metadata["sample_count"] == 3
    assert case.metadata["pass_count"] == 1
    assert case.metadata["pass_rate"] == pytest.approx(1 / 3)
    assert case.checks["final_output_matches"] is False


def test_execution_report_coverage_scorer_treats_tool_sequence_as_advisory(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private coverage request",
        final_text="private coverage answer",
        tool_name="bash",
        tool_input={"command": "weather-cli forecast 'СПб'"},
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(
        store,
        pack=pack,
        executor=_CoverageMismatchExecutor(),
        scorer=resolve_execution_scorer("capability_coverage_oracle_v1"),
    )

    assert result.report.passed_count == 1
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.checks["tool_sequence_matches"] is False
    assert "tool_sequence_matches" in case.warnings
    assert case.checks["final_output_matches"] is True
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["observed_capabilities"] == [
        "bash:weather-cli forecast"
    ]
    assert case.observed_trace.metadata["expected_capabilities"] == [
        "bash:weather-cli forecast"
    ]


def test_execution_report_capability_trace_scorer_treats_tool_sequence_as_advisory(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private capability trace request",
        final_text="private capability trace answer",
        tool_name="bash",
        tool_input={"command": "weather-cli forecast 'СПб'"},
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(
        store,
        pack=pack,
        executor=_CoverageMismatchExecutor(),
        scorer=resolve_execution_scorer("capability_trace_oracle_v1"),
    )

    assert result.report.passed_count == 1
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.checks["tool_sequence_matches"] is False
    assert "tool_sequence_matches" in case.warnings
    assert case.checks["final_output_matches"] is True
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["observed_capabilities"] == [
        "bash:weather-cli forecast"
    ]
    assert case.observed_trace.metadata["expected_capabilities"] == [
        "bash:weather-cli forecast"
    ]


def test_execution_report_blocks_unresolvable_refs(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="cases/gold_cases.jsonl",
        cases=[
            EvalRunPackCase(
                gold_case_id="gold-1",
                case_id="case-1",
                episode_id="missing-episode",
                case_kind="tool_workflow",
                input_facet_ids=["missing-input"],
                expected_facet_ids=["missing-output"],
                tool_names=["missing_tool"],
                rubric=["must work"],
            )
        ],
    )

    result = run_execution_report(store, pack=pack)

    assert result.report.passed_count == 0
    assert result.report.failed_count == 0
    assert result.report.blocked_count == 1
    case = result.report.cases[0]
    assert case.status == "blocked"
    assert case.observed_trace is None
    assert case.warnings == [
        "episode_exists",
        "has_events",
        "input_facets_resolve",
        "expected_facets_resolve",
        "tool_fixtures_resolve",
        "tool_trace_complete",
    ]


def test_execution_report_records_executor_errors_without_private_message(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private error request",
        final_text="private error answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(store, pack=pack, executor=_ErrorExecutor())

    assert result.report.passed_count == 0
    assert result.report.failed_count == 0
    assert result.report.error_count == 1
    case = result.report.cases[0]
    assert case.status == "error"
    assert case.observed_trace is not None
    assert case.observed_trace.error_type == "RuntimeError"
    assert case.observed_trace.error_hash
    serialized = result.path.read_text(encoding="utf-8")
    assert "private executor failure" not in serialized
    assert "private error request" not in serialized
    assert "private error answer" not in serialized


def test_execution_report_validates_limit_empty_pack_and_paths(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    pack = EvalRunPack(pack_id="pack-1", source_records_path="cases/gold_cases.jsonl")

    with pytest.raises(ValueError, match="limit must be positive"):
        run_execution_report(store, pack=pack, limit=0)
    with pytest.raises(ValueError, match="samples must be positive"):
        run_execution_report(store, pack=pack, samples=0)
    with pytest.raises(ValueError, match="eval pack must contain cases"):
        run_execution_report(store, pack=pack)
    with pytest.raises(ValueError, match="store.root/reports"):
        run_execution_report(
            store,
            pack=EvalRunPack(
                pack_id="pack-1",
                source_records_path="cases/gold_cases.jsonl",
                cases=[
                    EvalRunPackCase(
                        gold_case_id="gold",
                        case_id="case",
                        episode_id="ep",
                        case_kind="conversation_replay",
                    )
                ],
            ),
            report_filename="../eval_report.json",
        )


def test_execution_report_blocks_corrupt_resource_snapshot(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private snapshot request",
        final_text="private snapshot answer",
    )
    snapshot_path = store.root / "states" / "ep-1" / "resource_snapshot.json"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        EvalResourceSnapshot(
            episode_id="different-episode",
            resources=[
                EvalResource(
                    resource_id="bad",
                    kind="local_file",
                    name="bad",
                    path="/private/outside",
                    exists=True,
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="resource_snapshot",
            payload={"path": "states/ep-1/resource_snapshot.json"},
        )
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(store, pack=pack)

    assert result.report.passed_count == 0
    assert result.report.failed_count == 0
    assert result.report.blocked_count == 1
    case = result.report.cases[0]
    assert case.status == "blocked"
    assert case.checks["resource_snapshot_valid_or_absent"] is False
    assert case.metadata["resource_snapshot_status"] == "corrupt"


def test_execution_report_rejects_symlinked_report_dir(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private symlink request",
        final_text="private symlink answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    target = tmp_path / "outside-reports"
    target.mkdir()
    reports_dir = store.root / "reports"
    reports_dir.rmdir()
    try:
        os.symlink(target, reports_dir)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="directory must not be a symlink"):
        run_execution_report(store, pack=pack)


def test_execution_report_per_case_scorer_overrides_default(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private req",
        final_text="private ans",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    pack = pack.model_copy(
        update={
            "cases": [
                case.model_copy(update={"scorer": "tool_trace_oracle_v1"})
                for case in pack.cases
            ]
        }
    )

    result = run_execution_report(store, pack=pack)

    assert result.report.passed_count == 1
    case = result.report.cases[0]
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["scorer_name"] == "tool_trace_oracle_v1"
    assert case.checks["final_output_matches"] is True


def test_execution_report_run_scorer_overrides_per_case_scorer(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private req",
        final_text="private ans",
        tool_name="bash",
        tool_input={"command": "weather-cli forecast 'СПб'"},
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    pack = pack.model_copy(
        update={
            "cases": [
                case.model_copy(update={"scorer": "capability_coverage_oracle_v1"})
                for case in pack.cases
            ]
        }
    )
    scorer = _RecordingPassScorer()

    result = run_execution_report(
        store,
        pack=pack,
        executor=_SequenceExecutor([False]),
        scorer=scorer,
    )

    assert scorer.called is True
    assert result.report.passed_count == 1
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.checks["final_output_matches"] is True
    assert case.metadata["scorer_name"] == "recording-pass"
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["scorer_name"] == "recording-pass"


def test_execution_report_per_case_coverage_scorer_used_without_run_override(
    tmp_path: Path,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private req",
        final_text="private ans",
        tool_name="bash",
        tool_input={"command": "weather-cli forecast 'СПб'"},
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    pack = pack.model_copy(
        update={
            "cases": [
                case.model_copy(update={"scorer": "capability_coverage_oracle_v1"})
                for case in pack.cases
            ]
        }
    )

    result = run_execution_report(
        store,
        pack=pack,
        executor=_CoverageMismatchExecutor(),
    )

    assert result.report.passed_count == 1
    assert result.report.metadata["scorer_name"] == "exact-final-text"
    case = result.report.cases[0]
    assert case.status == "passed"
    assert case.checks["tool_sequence_matches"] is False
    assert case.checks["final_output_matches"] is True
    assert case.observed_trace is not None
    assert case.observed_trace.metadata["scorer_name"] == "capability_coverage_oracle_v1"
    assert case.observed_trace.metadata["observed_capabilities"] == [
        "bash:weather-cli forecast"
    ]


def test_execution_report_rejects_unknown_per_case_scorer(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private req",
        final_text="private ans",
        tool_name="web_fetch",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack
    pack = pack.model_copy(
        update={
            "cases": [case.model_copy(update={"scorer": "nope"}) for case in pack.cases]
        }
    )

    with pytest.raises(ValueError, match="unknown eval scorer"):
        run_execution_report(store, pack=pack)


def test_execution_report_surfaces_safe_executor_counters(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-counters",
        user_text="private req",
        final_text="private ans",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    pack = write_run_pack(store).pack

    result = run_execution_report(
        store,
        pack=pack,
        executor=_CounterMetadataExecutor(),
    )

    case = result.report.cases[0]
    assert case.metadata["seeded_history_message_count"] == 2
    assert case.metadata["materialized_file_count"] == 1
    assert "private_note" not in case.metadata
    assert "private executor metadata" not in result.path.read_text(encoding="utf-8")


class _CounterMetadataExecutor:
    name = "counter_metadata"

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        return EvalExecutorResult(
            final_text=context.expected_final_text,
            tool_path=tuple(context.case.tool_names),
            event_kind_path=("execution_started", "execution_completed"),
            metadata={
                "seeded_history_message_count": 2,
                "materialized_file_count": 1,
                "private_note": "private executor metadata",
            },
        )


class _CapturingExecutor:
    name = "capture"

    def __init__(self) -> None:
        self.seen_prompt = ""
        self.seen_expected = ""

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        self.seen_prompt = context.primary_prompt
        self.seen_expected = context.expected_final_text
        return EvalExecutorResult(
            final_text=context.expected_final_text,
            tool_path=tuple(context.case.tool_names),
            event_kind_path=("execution_started", "execution_completed"),
            metadata={"note": "private executor metadata"},
        )


class _MismatchExecutor:
    name = "mismatch"

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        return EvalExecutorResult(
            final_text="different final text",
            tool_path=(),
            event_kind_path=("execution_started", "execution_completed"),
        )


class _SequenceExecutor:
    name = "sequence"

    def __init__(self, outcomes: list[bool]) -> None:
        self._outcomes = outcomes
        self.call_count = 0

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        outcome = self._outcomes[self.call_count]
        self.call_count += 1
        return EvalExecutorResult(
            final_text=(
                context.expected_final_text
                if outcome
                else "different final text"
            ),
            tool_path=tuple(context.case.tool_names),
            event_kind_path=("execution_started", "execution_completed"),
        )


class _CoverageMismatchExecutor:
    name = "coverage_mismatch"

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        return EvalExecutorResult(
            final_text="different final text",
            tool_path=("shell",),
            event_kind_path=("execution_started", "execution_completed"),
            tool_calls=(
                EvalObservedCall(
                    tool_name="bash",
                    arguments={"command": "weather-cli forecast 'СПб'"},
                ),
            ),
        )


class _AlwaysPassScorer:
    name = "always-pass"

    def score(self, *, context, executor_result):
        del context, executor_result
        return EvalExecutionScorerResult(
            passed=True,
            score=0.75,
            scorer_name=self.name,
            metadata={"note": "private scorer note"},
        )


class _RecordingPassScorer:
    name = "recording-pass"

    def __init__(self) -> None:
        self.called = False

    def score(self, *, context, executor_result):
        del context, executor_result
        self.called = True
        return EvalExecutionScorerResult(
            passed=True,
            score=0.5,
            scorer_name=self.name,
            metadata={"override": True},
        )


class _ErrorExecutor:
    name = "erroring"

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        raise RuntimeError("private executor failure")


class _LeakyLabelExecutor:
    name = "leaky_labels"

    def run_case(self, context: EvalExecutionContext) -> EvalExecutorResult:
        return EvalExecutorResult(
            final_text=context.expected_final_text,
            tool_path=(context.primary_prompt,),
            event_kind_path=(context.primary_prompt,),
        )


class _RecordingReplayApiClient:
    def __init__(self) -> None:
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu-replay-1",
                            name="web_fetch",
                            input={"query": "private raw tool input"},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text="model final from replayed tool")],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _FinalOnlyModelApiClient:
    def __init__(self, *, final_text: str) -> None:
        self._final_text = final_text
        self.message_snapshots = []

    async def stream_message(self, request):
        self.message_snapshots.append(
            [(message.role, message.text) for message in request.messages]
        )
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._final_text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _HistorySegmentApiClient:
    def __init__(self, response: str | Exception) -> None:
        self._response = response
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._response)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _ScriptedBashModelApiClient:
    """Fake model: first calls ``bash`` with a fixed command, then answers.

    Lets a test pin exactly which shell command the model emits, so a capability
    regression (right tool, wrong command) can be reproduced deterministically.
    """

    def __init__(self, *, command: str, final_text: str) -> None:
        self._command = command
        self._final_text = final_text
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield ApiMessageCompleteEvent(
                message=ConversationMessage(
                    role="assistant",
                    content=[
                        ToolUseBlock(
                            id="toolu-bash-1",
                            name="bash",
                            input={"command": self._command},
                        )
                    ],
                ),
                usage=UsageSnapshot(input_tokens=1, output_tokens=1),
            )
            return
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[TextBlock(text=self._final_text)],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _LoopingBashModelApiClient:
    """Fake model that keeps requesting ``bash`` and never emits a final answer."""

    def __init__(self, *, command: str) -> None:
        self._command = command
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id=f"toolu-bash-{len(self.requests)}",
                        name="bash",
                        input={"command": self._command},
                    )
                ],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


class _PartialThenLoopApiClient:
    """Emits a partial text answer *and* a tool call on the first turn, then loops
    on the tool — so the run reaches MaxTurnsExceeded after producing a real
    partial answer whose final turn (a tool call) carries empty text."""

    def __init__(self, *, partial_text: str, command: str) -> None:
        self._partial_text = partial_text
        self._command = command
        self.requests = []

    async def stream_message(self, request):
        self.requests.append(request)
        content = [
            ToolUseBlock(
                id=f"toolu-bash-{len(self.requests)}",
                name="bash",
                input={"command": self._command},
            )
        ]
        if len(self.requests) == 1:
            content = [TextBlock(text=self._partial_text), *content]
        yield ApiMessageCompleteEvent(
            message=ConversationMessage(role="assistant", content=content),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _query_replay_context(
    tmp_path: Path,
    *,
    conversation_history: tuple[tuple[str, str], ...],
) -> EvalExecutionContext:
    store = EvalStore(tmp_path / "evals")
    case = EvalRunPackCase(
        gold_case_id="gold-1",
        case_id="case-1",
        episode_id="ep-1",
        case_kind="unit",
        rubric=["Answer the user."],
    )
    pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="cases/gold_cases.jsonl",
        cases=[case],
    )
    episode = EvalEpisode(
        episode_id="ep-1",
        source="gateway",
        app="ohmo",
        session_id="session-1",
        user_text="current turn",
    )
    return EvalExecutionContext(
        store=store,
        pack=pack,
        case=case,
        episode=episode,
        events=(),
        input_facets=(),
        expected_facets=(),
        tool_fixtures=(),
        primary_prompt="current turn",
        expected_final_text="expected final",
        resource_snapshot_status="absent",
        conversation_history=conversation_history,
    )


def _add_session_episode(
    store: EvalStore,
    *,
    episode_id: str,
    session_id: str,
    app: str,
    created_at: datetime,
    user_text: str,
    final_text: str,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app=app,
            session_id=session_id,
            created_at=created_at,
            user_text=user_text,
        )
    )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            payload={"text": final_text},
        )
    )


def _add_episode(
    store: EvalStore,
    *,
    episode_id: str,
    user_text: str,
    final_text: str,
    tool_name: str | None = None,
    tool_input: dict | None = None,
) -> None:
    store.append_episode(
        EvalEpisode(
            episode_id=episode_id,
            source="gateway",
            app="ohmo",
            session_id=f"session-{episode_id}",
            user_text=user_text,
        )
    )
    if tool_name:
        store.append_event(
            EvalEvent(
                episode_id=episode_id,
                kind="tool_started",
                tool_name=tool_name,
                tool_call_id="tool-1",
                payload={
                    "input_summary": "private tool input",
                    "input": (
                        {"query": "private raw tool input"}
                        if tool_input is None
                        else tool_input
                    ),
                },
            )
        )
        store.append_event(
            EvalEvent(
                episode_id=episode_id,
                kind="tool_completed",
                tool_name=tool_name,
                tool_call_id="tool-1",
                payload={
                    "output_summary": "private tool output",
                    "output": {"text": "private raw tool output"},
                },
            )
        )
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            payload={"text": final_text},
        )
    )

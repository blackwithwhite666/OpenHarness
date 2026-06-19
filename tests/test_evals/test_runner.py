from __future__ import annotations

import os
from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock, ToolResultBlock, ToolUseBlock
from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    EvalExecutionContext,
    EvalExecutorResult,
    QueryEngineEvalAgentRunner,
    ReplayToolsExecutor,
    EvalResource,
    EvalResourceSnapshot,
    EvalRunPack,
    EvalRunPackCase,
    EvalStore,
    build_case_candidates,
    build_case_drafts,
    promote_case_drafts,
    run_execution_report,
    run_replay_report,
    write_case_draft_pack,
    write_run_pack,
)


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

    serialized = result.path.read_text(encoding="utf-8")
    assert "private execution request" not in serialized
    assert "private execution answer" not in serialized
    assert "private raw tool input" not in serialized
    assert "private raw tool output" not in serialized
    assert "private tool input" not in serialized
    assert "private tool output" not in serialized


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


def _add_episode(
    store: EvalStore,
    *,
    episode_id: str,
    user_text: str,
    final_text: str,
    tool_name: str | None = None,
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
                    "input": {"query": "private raw tool input"},
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

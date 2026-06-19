from __future__ import annotations

from pathlib import Path

import pytest

from openharness.api.client import ApiMessageCompleteEvent
from openharness.api.usage import UsageSnapshot
from openharness.engine.messages import ConversationMessage, TextBlock, ToolUseBlock
from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    QueryEngineEvalAgentRunner,
    ReplayToolsExecutor,
    EvalRunPack,
    EvalStore,
    build_case_candidates,
    build_case_drafts,
    promote_case_drafts,
    run_execution_report,
    write_case_draft_pack,
    write_run_pack,
)


@pytest.mark.asyncio
async def test_replay_tools_runner_can_execute_inside_existing_event_loop(
    tmp_path: Path,
):
    store, pack = _prepare_pack(tmp_path)

    result = run_execution_report(store, pack=pack)

    assert result.report.passed_count == 1
    assert result.report.error_count == 0


@pytest.mark.asyncio
async def test_query_engine_runner_can_execute_inside_existing_event_loop(
    tmp_path: Path,
):
    store, pack = _prepare_pack(tmp_path)
    api_client = _ReplayApiClient()

    result = run_execution_report(
        store,
        pack=pack,
        executor=ReplayToolsExecutor(
            agent_runner=QueryEngineEvalAgentRunner(
                api_client=api_client,
                model="eval-model",
                cwd=tmp_path,
            )
        ),
    )

    assert result.report.passed_count == 1
    assert result.report.error_count == 0
    assert len(api_client.requests) == 2


class _ReplayApiClient:
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
                content=[TextBlock(text="private final answer")],
            ),
            usage=UsageSnapshot(input_tokens=1, output_tokens=1),
        )


def _prepare_pack(tmp_path: Path) -> tuple[EvalStore, EvalRunPack]:
    store = EvalStore(tmp_path / "evals")
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_started",
            tool_name="web_fetch",
            tool_call_id="tool-1",
            payload={
                "input_summary": "private tool input",
                "input": {"query": "private raw tool input"},
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_completed",
            tool_name="web_fetch",
            tool_call_id="tool-1",
            payload={
                "output_summary": "private tool output",
                "output": {"text": "private raw tool output"},
            },
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="gateway_final",
            payload={"text": "private final answer"},
        )
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])
    return store, write_run_pack(store).pack

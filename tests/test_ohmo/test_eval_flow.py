from __future__ import annotations

import asyncio
import json
from pathlib import Path

from openharness.evals import EvalEpisode, EvalEvent
from ohmo.evals import (
    build_ohmo_eval_pack,
    check_ohmo_eval_run_config,
    compare_ohmo_eval_reports,
    get_eval_store,
    promote_ohmo_eval_case_drafts,
    run_ohmo_eval_report,
    run_ohmo_eval_smoke,
    save_ohmo_eval_baseline,
    validate_ohmo_eval_review_manifest,
    write_ohmo_embedding_index,
    write_ohmo_eval_mine,
    write_ohmo_eval_review_manifest,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(
        self,
        texts: list[str],
        *,
        return_dense: bool = True,
        return_sparse: bool = False,
        batch_size: int | None = None,
    ) -> dict[str, object]:
        self.texts.extend(texts)
        return {
            "model": "BAAI/bge-m3",
            "count": len(texts),
            "dense": [[0.1, 0.2, 0.3] for _ in texts],
            "lexical_weights": None,
        }


def test_ohmo_eval_flywheel_golden_path_is_metadata_only(tmp_path: Path):
    workspace = tmp_path / "workspace"
    store = get_eval_store(workspace)
    store.append_episode(
        EvalEpisode(
            episode_id="ep-1",
            source="gateway",
            app="ohmo",
            session_id="session-1",
            user_text="private full path request",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_started",
            payload={"input_summary": "private tool input"},
            tool_name="web_fetch",
            tool_call_id="call-1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="tool_completed",
            payload={"output_summary": "private tool output"},
            tool_name="web_fetch",
            tool_call_id="call-1",
        )
    )
    store.append_event(
        EvalEvent(
            episode_id="ep-1",
            kind="gateway_final",
            payload={"text": "private final answer"},
        )
    )

    client = FakeEmbeddingClient()
    embedding = asyncio.run(
        write_ohmo_embedding_index(
            workspace=workspace,
            client=client,
            batch_size=2,
        )
    )
    mined = write_ohmo_eval_mine(workspace=workspace)
    review_manifest = write_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )
    _approve_first_review_item(review_manifest.path)
    validation = validate_ohmo_eval_review_manifest(
        workspace=workspace,
        filename="batch_review.json",
    )
    promoted = promote_ohmo_eval_case_drafts(
        workspace=workspace,
        manifest_filename="batch_review.json",
        reviewer="reviewer-1",
    )
    pack = build_ohmo_eval_pack(workspace=workspace)
    smoke = run_ohmo_eval_smoke(workspace=workspace, limit=1, report_only=True)
    config = check_ohmo_eval_run_config(workspace=workspace, limit=1)
    report = run_ohmo_eval_report(workspace=workspace, limit=1, report_only=True)
    baseline = save_ohmo_eval_baseline(
        workspace=workspace,
        source_report="eval_report.json",
        name="main",
    )
    comparison = compare_ohmo_eval_reports(
        workspace=workspace,
        baseline_report="baselines/main.json",
        candidate_report="eval_report.json",
        report_only=True,
    )

    assert client.texts == [
        "private full path request",
        "private tool input",
        "private tool output",
        "private final answer",
    ]
    assert embedding.manifest.embedding_count == 4
    assert mined.candidates.manifest.record_count == 1
    assert mined.cases.manifest.record_count == 1
    assert validation.approved_count == 1
    assert promoted.promoted_count == 1
    assert pack.case_count == 1
    assert pack.write.pack.cases[0].case_kind == "tool_workflow"
    assert pack.write.pack.cases[0].tool_names == ["web_fetch"]
    assert smoke.write.report.passed_count == 1
    assert config.replay_tools_only is True
    assert report.write.report.passed_count == 1
    assert baseline.name == "main"
    assert comparison.write.report.regression_count == 0

    _assert_artifacts_are_metadata_only(
        [
            embedding.manifest_path,
            embedding.records_path,
            mined.candidates.manifest_path,
            mined.candidates.records_path,
            mined.cases.manifest_path,
            mined.cases.records_path,
            promoted.manifest_path,
            promoted.records_path,
            pack.write.path,
            smoke.write.path,
            report.write.path,
            baseline.path,
            comparison.write.path,
        ],
        forbidden=[
            "private full path request",
            "private tool input",
            "private tool output",
            "private final answer",
            "private review comment",
        ],
    )


def _approve_first_review_item(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["items"][0]["decision"] = "approved"
    payload["items"][0]["reviewer"] = "manifest-reviewer"
    payload["items"][0]["comment"] = "private review comment"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _assert_artifacts_are_metadata_only(
    paths: list[Path],
    *,
    forbidden: list[str],
) -> None:
    for path in paths:
        serialized = path.read_text(encoding="utf-8")
        for text in forbidden:
            assert text not in serialized, path

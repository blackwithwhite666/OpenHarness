from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    EvalGoldCase,
    EvalResource,
    EvalResourceSnapshot,
    EvalRunPack,
    EvalRunPackCase,
    EvalStore,
    build_case_candidates,
    build_case_drafts,
    build_run_pack,
    promote_case_drafts,
    read_gold_cases,
    run_smoke_report,
    write_case_draft_pack,
    write_run_pack,
)


def test_run_pack_and_smoke_report_are_metadata_only(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private pack request",
        final_text="private pack answer",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    drafts[0] = drafts[0].model_copy(
        update={"capability_path": ["bash:weather-cli forecast"]}
    )
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id], reviewer="reviewer-1")

    pack_write = write_run_pack(store)
    smoke_write = run_smoke_report(store, pack=pack_write.pack)

    assert pack_write.relative_path == "packs/eval_pack.json"
    assert pack_write.pack.source_records_path == "cases/gold_cases.jsonl"
    assert pack_write.pack.metadata == {"privacy": "metadata_only", "case_count": 1}
    assert len(pack_write.pack.cases) == 1
    assert pack_write.pack.cases[0].case_id == drafts[0].case_id
    assert pack_write.pack.cases[0].capability_path == ["bash:weather-cli forecast"]
    assert pack_write.pack.cases[0].metadata["review_status"] == "approved"

    assert smoke_write.relative_path == "reports/smoke_report.json"
    assert smoke_write.report.report_kind == "smoke_report"
    assert smoke_write.report.case_count == 1
    assert smoke_write.report.passed_count == 1
    assert smoke_write.report.failed_count == 0
    assert smoke_write.report.metadata["mode"] == "smoke_report_only"

    serialized = (
        pack_write.path.read_text(encoding="utf-8")
        + smoke_write.path.read_text(encoding="utf-8")
    )
    assert "private pack request" not in serialized
    assert "private pack answer" not in serialized


def test_run_pack_can_select_gold_cases_and_rejects_missing(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(store, episode_id="ep-1", user_text="private one", final_text="private final")
    _add_episode(store, episode_id="ep-2", user_text="private two", final_text="private final")
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store)

    pack = build_run_pack(store, case_ids=[drafts[1].case_id])

    assert [case.case_id for case in pack.cases] == [drafts[1].case_id]
    with pytest.raises(ValueError, match="gold case not found: missing"):
        build_run_pack(store, case_ids=["missing"])


def test_run_pack_skips_unreplayable_cases_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-broken",
        user_text="[voice: download failed]",
        final_text="private broken final",
    )
    _add_episode(
        store,
        episode_id="ep-clean",
        user_text="private clean request",
        final_text="private clean final",
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store)

    clean_case_id = next(draft.case_id for draft in drafts if draft.episode_id == "ep-clean")
    broken_gold_case_id = next(
        gold.gold_case_id for gold in read_gold_cases(store) if gold.episode_id == "ep-broken"
    )
    with caplog.at_level("WARNING", logger="openharness.evals.pack"):
        pack = build_run_pack(store)

    assert [case.case_id for case in pack.cases] == [clean_case_id]
    assert pack.metadata["case_count"] == 1
    assert pack.metadata["skipped_unreplayable_count"] == 1
    assert pack.metadata["skipped_unreplayable"] == [
        {
            "gold_case_id": broken_gold_case_id,
            "reason": "unreplayable_input:download_failed",
        }
    ]
    assert (
        f"pack: skipped unreplayable case {broken_gold_case_id}: unreplayable_input:download_failed"
    ) in caplog.text
    assert "pack: skipped 1 unreplayable cases:" in caplog.text


def test_run_pack_carries_state_delta_metadata(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="private pack request",
        final_text="private pack answer",
    )
    _write_state_snapshot(
        store,
        episode_id="ep-1",
        phase="world_before",
        reminder_keys=("aaaaaaaaaaaaaaaa",),
    )
    _write_state_snapshot(
        store,
        episode_id="ep-1",
        phase="world_after",
        reminder_keys=("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"),
    )
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store, case_ids=[drafts[0].case_id])

    pack = build_run_pack(store)

    assert pack.cases[0].metadata["state_delta"]["changed"] is True
    assert pack.cases[0].metadata["state_delta"]["reminders"]["added_keys"] == [
        "bbbbbbbbbbbbbbbb"
    ]


def test_run_pack_ids_are_stable_and_report_ids_include_selected_cases(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    _add_episode(store, episode_id="ep-1", user_text="private one", final_text="private final")
    _add_episode(store, episode_id="ep-2", user_text="private two", final_text="private final")
    drafts = build_case_drafts(store, build_case_candidates(store))
    write_case_draft_pack(store, drafts)
    promote_case_drafts(store)

    full_pack = build_run_pack(store)
    repeat_pack = build_run_pack(store)
    limited_pack = full_pack.model_copy(update={"cases": full_pack.cases[:1]})
    full_report = run_smoke_report(store, pack=full_pack).report
    limited_report = run_smoke_report(store, pack=limited_pack).report

    assert repeat_pack.pack_id == full_pack.pack_id
    assert limited_report.report_id != full_report.report_id


def test_run_pack_and_smoke_report_validate_paths_and_empty_state(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    non_empty_pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="x",
        cases=[
            EvalRunPackCase(
                gold_case_id="gold-1",
                case_id="case-1",
                episode_id="ep-1",
                case_kind="conversation_replay",
            )
        ],
    )

    with pytest.raises(ValueError, match="no gold cases"):
        write_run_pack(store)
    with pytest.raises(ValueError, match="store.root/packs"):
        write_run_pack(store, pack=EvalRunPack(pack_id="pack-1", source_records_path="x"), pack_filename="../x.json")
    with pytest.raises(ValueError, match="store.root/reports"):
        run_smoke_report(
            store,
            pack=non_empty_pack,
            report_filename="../x.json",
        )
    with pytest.raises(ValueError, match="store.root/cases"):
        build_run_pack(store, gold_records_filename="../gold_cases.jsonl")


def test_run_pack_rejects_duplicate_gold_case_ids(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    gold = EvalGoldCase(
        gold_case_id="gold-1",
        case_id="case-1",
        candidate_id="candidate-1",
        episode_id="ep-1",
        case_kind="conversation_replay",
        input_facet_ids=["facet-input"],
        expected_facet_ids=["facet-output"],
        rubric=["be useful"],
    )
    (store.root / "cases" / "gold_cases.jsonl").write_text(
        gold.model_dump_json() + "\n" + gold.model_dump_json() + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate gold case ids: case-1"):
        build_run_pack(store)


def test_smoke_report_fails_cases_missing_required_refs(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")
    pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="cases/gold_cases.jsonl",
        cases=[
            EvalRunPackCase(
                gold_case_id="gold-1",
                case_id="case-1",
                episode_id="ep-1",
                case_kind="conversation_replay",
            )
        ],
    )

    result = run_smoke_report(store, pack=pack)

    assert result.report.case_count == 1
    assert result.report.passed_count == 0
    assert result.report.failed_count == 1
    assert result.report.cases[0].status == "failed"
    assert result.report.cases[0].warnings == [
        "has_input_facets",
        "has_expected_facets",
        "has_rubric",
    ]


def test_smoke_report_rejects_empty_pack(tmp_path: Path):
    store = EvalStore(tmp_path / "evals")

    with pytest.raises(ValueError, match="eval pack must contain cases"):
        run_smoke_report(
            store,
            pack=EvalRunPack(pack_id="pack-1", source_records_path="cases/gold_cases.jsonl"),
        )


def _add_episode(
    store: EvalStore,
    *,
    episode_id: str,
    user_text: str,
    final_text: str,
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
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            payload={"text": final_text},
        )
    )


def _write_state_snapshot(
    store: EvalStore,
    *,
    episode_id: str,
    phase: str,
    reminder_keys: tuple[str, ...],
) -> None:
    path = store.root / "states" / episode_id / f"{phase}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = EvalResourceSnapshot(
        episode_id=episode_id,
        resources=[
            EvalResource(
                resource_id="resource:reminders_json",
                kind="local_file",
                name="reminders_json",
                exists=True,
                metadata={
                    "entry_keys": list(reminder_keys),
                    "record_count": len(reminder_keys),
                    "status_counts": {"pending": len(reminder_keys)},
                },
            )
        ],
    )
    path.write_text(snapshot.model_dump_json(), encoding="utf-8")

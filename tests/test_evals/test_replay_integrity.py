from __future__ import annotations

from pathlib import Path

import pytest

from openharness.evals import (
    EvalEpisode,
    EvalEvent,
    EvalRunPack,
    EvalRunPackCase,
    EvalStore,
    EvalToolFixture,
    collect_text_facets,
    replay_integrity,
    run_execution_report,
)


def test_replay_integrity_blocks_download_failed_input() -> None:
    ok, reason = replay_integrity(
        _episode("[voice: download failed]"),
        (),
        (),
    )

    assert ok is False
    assert reason == "unreplayable_input:download_failed"


def test_replay_integrity_blocks_empty_media_placeholder_input() -> None:
    ok, reason = replay_integrity(
        _episode("  [voice] \n [image]  "),
        (),
        (),
    )

    assert ok is False
    assert reason == "unreplayable_input:empty"


def test_replay_integrity_blocks_telegram_media_reference() -> None:
    ok, reason = replay_integrity(
        _episode("Please process AgADabcdefghijklmnop from Telegram."),
        (),
        (),
    )

    assert ok is False
    assert reason == "unrecoverable_media_ref"


def test_replay_integrity_blocks_missing_read_file_absolute_path() -> None:
    ok, reason = replay_integrity(
        _episode("Read the captured file."),
        (),
        (
            EvalToolFixture(
                tool_name="read_file",
                call_key_hash="fixture-read",
                input_text='{"path": "/tmp/missing-input.txt"}',
                output_text="No such file or directory",
                is_error=True,
            ),
        ),
    )

    assert ok is False
    assert reason == "missing_input_file"


def test_replay_integrity_allows_notfound_text_in_successful_read() -> None:
    # A successful read whose CONTENT mentions "not found" / "ENOENT" (e.g. docs)
    # must NOT be flagged -- only a real read error counts (regression for the
    # false-positive that over-blocked a passing case).
    ok, reason = replay_integrity(
        _episode("Make the skill."),
        (),
        (
            EvalToolFixture(
                tool_name="read_file",
                call_key_hash="fixture-read",
                input_text='{"path": "/home/u/.ohmo/skills/x/SKILL.md"}',
                output_text="if the key is not found raise ENOENT ...",
                is_error=False,
            ),
        ),
    )

    assert ok is True
    assert reason is None


def test_replay_integrity_blocks_telegram_media_reference_in_gold() -> None:
    # The file-id lives in the gold reply (an event payload), not this turn's text.
    ok, reason = replay_integrity(
        _episode("Now verify the pdf skill works."),
        (
            EvalEvent(
                episode_id="ep-1",
                kind="gateway_final",
                payload={"text": "OCR ran on input: AgADabcdefghijklmnop"},
            ),
        ),
        (),
    )

    assert ok is False
    assert reason == "unrecoverable_media_ref"


def test_replay_integrity_allows_clean_text_and_normal_fixtures() -> None:
    ok, reason = replay_integrity(
        _episode("Review [draft] notes in report.txt and summarize them."),
        (),
        (
            EvalToolFixture(
                tool_name="read_file",
                call_key_hash="fixture-read",
                input_text='{"path": "report.txt"}',
                output_text="contents",
            ),
        ),
    )

    assert ok is True
    assert reason is None


def test_execution_report_blocks_unreplayable_input_without_model_failure(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = EvalStore(tmp_path / "evals")
    _add_episode(
        store,
        episode_id="ep-1",
        user_text="[voice: download failed]",
        final_text="private final",
    )
    facets = collect_text_facets(store)
    input_ids = [item.facet.facet_id for item in facets if item.facet.facet_kind == "user_request"]
    expected_ids = [
        item.facet.facet_id for item in facets if item.facet.facet_kind == "assistant_final"
    ]
    pack = EvalRunPack(
        pack_id="pack-1",
        source_records_path="cases/gold_cases.jsonl",
        cases=[
            EvalRunPackCase(
                gold_case_id="gold-1",
                case_id="case-1",
                episode_id="ep-1",
                case_kind="conversation_replay",
                input_facet_ids=input_ids,
                expected_facet_ids=expected_ids,
                rubric=["answer"],
            )
        ],
    )

    with caplog.at_level("WARNING", logger="openharness.evals.execution"):
        result = run_execution_report(store, pack=pack)

    assert result.report.failed_count == 0
    assert result.report.blocked_count == 1
    case = result.report.cases[0]
    assert case.status == "blocked"
    assert case.checks["replay_inputs_recoverable"] is False
    assert case.metadata["unreplayable_reason"] == "unreplayable_input:download_failed"
    assert case.context is not None
    assert case.context.metadata["unreplayable_reason"] == "unreplayable_input:download_failed"
    assert "eval case gold-1 unreplayable: unreplayable_input:download_failed" in caplog.text


def _episode(user_text: str) -> EvalEpisode:
    return EvalEpisode(episode_id="ep-1", user_text=user_text)


def _add_episode(
    store: EvalStore,
    *,
    episode_id: str,
    user_text: str,
    final_text: str,
) -> None:
    store.append_episode(EvalEpisode(episode_id=episode_id, user_text=user_text))
    store.append_event(
        EvalEvent(
            episode_id=episode_id,
            kind="gateway_final",
            payload={"text": final_text},
        )
    )

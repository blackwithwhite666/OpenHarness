"""Offline end-to-end tests for the GAIA runner (ADR §4, component #10).

Zero network, zero model, zero subprocess. The spawn boundary is INJECTED — a
canned-transcript stand-in passed as ``spawn_fn`` — so the K=3 orchestration,
the REPORT math, and the extraction-/infra-failure tagging are exercised fully
offline.

The three transcript shapes the ADR calls out:
  * a "good" transcript (correct <final_answer>),
  * a "sentinel-less" transcript (no sentinel -> exercises the tolerant
    extractor / a wrong last-line),
  * a spawn-failure (raises InfraFailure -> tagged distinct from wrong-answer).
"""

from __future__ import annotations

import json

import pytest

from tests.eval.gaia import run_subset
from tests.eval.gaia.run_subset import (
    InfraFailure,
    TaskRun,
    build_prompt,
    prepare_task_dir,
    run_subset as run_subset_fn,
)


def _task(task_id: str, gt: str, level: int, cwd) -> TaskRun:
    return TaskRun(
        task_id=task_id,
        question=f"question for {task_id}",
        ground_truth=gt,
        level=level,
        cwd=cwd,
    )


@pytest.mark.asyncio
async def test_run_subset_good_sentinel_passes(tmp_path):
    task = _task("t-good", "Paris", 1, tmp_path)

    async def good_spawn(t, model):
        assert model == "gpt-5.5"
        return f"Reasoning...\n<final_answer>{t.ground_truth}</final_answer>"

    report = await run_subset_fn(
        [task],
        spawn_fn=good_spawn,
        results_dir=tmp_path / "results",
        git_sha="sha1",
        k=3,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "results" / "sha1.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["score"] == 1.0
    assert row["extraction_failure"] is False
    assert row["infra_failure"] is False
    text = report.read_text()
    assert "sha1" in text
    assert "Overall" in text
    # 1/1 pass -> accuracy 1.000.
    assert "1.000" in text


@pytest.mark.asyncio
async def test_run_subset_sentinel_less_wrong(tmp_path):
    # No sentinel -> tolerant extractor falls back to the last line ("Berlin"),
    # which mismatches the ground truth "Paris" -> wrong (NOT extraction fail,
    # since a non-empty answer was extracted).
    task = _task("t-nosentinel", "Paris", 2, tmp_path)

    async def nosentinel_spawn(t, model):
        return "I reasoned about it.\nBerlin"

    await run_subset_fn(
        [task],
        spawn_fn=nosentinel_spawn,
        results_dir=tmp_path / "results",
        git_sha="sha2",
        k=3,
    )
    (row,) = [
        json.loads(line)
        for line in (tmp_path / "results" / "sha2.jsonl").read_text().splitlines()
    ]
    assert row["score"] == 0.0
    assert row["extraction_failure"] is False  # an answer WAS extracted
    assert row["infra_failure"] is False


@pytest.mark.asyncio
async def test_run_subset_extraction_failure_tagged(tmp_path):
    # Empty transcript -> no answer extractable -> extraction failure, score 0.
    task = _task("t-empty", "42", 1, tmp_path)

    async def empty_spawn(t, model):
        return "   \n  \n"

    await run_subset_fn(
        [task],
        spawn_fn=empty_spawn,
        results_dir=tmp_path / "results",
        git_sha="sha3",
        k=3,
    )
    (row,) = [
        json.loads(line)
        for line in (tmp_path / "results" / "sha3.jsonl").read_text().splitlines()
    ]
    assert row["score"] == 0.0
    assert row["extraction_failure"] is True
    assert row["extraction_failure_runs"] == 3
    assert row["infra_failure"] is False


@pytest.mark.asyncio
async def test_run_subset_infra_failure_tagged_distinct(tmp_path):
    # Spawn raises InfraFailure -> tagged as infra (NOT wrong-answer / extraction).
    task = _task("t-infra", "42", 3, tmp_path)

    async def failing_spawn(t, model):
        raise InfraFailure("spawn 403 / timeout simulation")

    await run_subset_fn(
        [task],
        spawn_fn=failing_spawn,
        results_dir=tmp_path / "results",
        git_sha="sha4",
        k=3,
    )
    (row,) = [
        json.loads(line)
        for line in (tmp_path / "results" / "sha4.jsonl").read_text().splitlines()
    ]
    assert row["score"] == 0.0
    assert row["infra_failure"] is True
    assert row["infra_failure_runs"] == 3
    # Crucially: an infra failure is NOT an extraction failure.
    assert row["extraction_failure"] is False


@pytest.mark.asyncio
async def test_run_subset_mixed_runs_report_math(tmp_path):
    # Three tasks: one all-good (L1), one all-wrong (L2), one infra (L3).
    tasks = [
        _task("g", "Paris", 1, tmp_path),
        _task("w", "Paris", 2, tmp_path),
        _task("i", "Paris", 3, tmp_path),
    ]

    async def router(t, model):
        if t.task_id == "g":
            return "<final_answer>Paris</final_answer>"
        if t.task_id == "w":
            return "<final_answer>London</final_answer>"
        raise InfraFailure("boom")

    report = await run_subset_fn(
        tasks,
        spawn_fn=router,
        results_dir=tmp_path / "results",
        git_sha="mix",
        k=3,
    )
    rows = {
        r["task_id"]: r
        for r in (
            json.loads(line)
            for line in (tmp_path / "results" / "mix.jsonl").read_text().splitlines()
        )
    }
    assert rows["g"]["score"] == 1.0
    assert rows["w"]["score"] == 0.0
    assert rows["i"]["infra_failure"] is True

    text = report.read_text()
    # Per-level rows present.
    assert "Level 1" in text and "Level 2" in text and "Level 3" in text
    # Overall accuracy = 1/3 passes.
    assert "0.333" in text
    # Infra-failure surfaced with its own line + rate.
    assert "Infra failures" in text
    # Wilson CI bracket rendered.
    assert "[" in text and "]" in text


@pytest.mark.asyncio
async def test_run_subset_k2_majority_pass(tmp_path):
    # K=2 with one good + one wrong run -> mean 0.5 -> counts as a pass
    # (aggregate threshold >= 0.5).
    task = _task("half", "Paris", 1, tmp_path)
    calls = {"n": 0}

    async def alternating(t, model):
        calls["n"] += 1
        return (
            "<final_answer>Paris</final_answer>"
            if calls["n"] == 1
            else "<final_answer>London</final_answer>"
        )

    await run_subset_fn(
        [task],
        spawn_fn=alternating,
        results_dir=tmp_path / "results",
        git_sha="k2",
        k=2,
    )
    (row,) = [
        json.loads(line)
        for line in (tmp_path / "results" / "k2.jsonl").read_text().splitlines()
    ]
    assert row["score"] == 0.5
    agg = run_subset.aggregate([row])
    assert agg["overall"]["k"] == 1  # 0.5 >= 0.5 -> pass


# --------------------------------------------------------------------------- #
# Attachment contract (ADR §2(c)): per-task dir + file copy + prompt augment.
# --------------------------------------------------------------------------- #
def test_prepare_task_dir_copies_attachment(tmp_path):
    src = tmp_path / "src" / "data.xlsx"
    src.parent.mkdir(parents=True)
    src.write_text("spreadsheet bytes", encoding="utf-8")

    run = prepare_task_dir(
        task_id="att1",
        question="read the file",
        ground_truth="42",
        level=2,
        source_file=src,
        work_root=tmp_path / "work",
    )
    # The per-task cwd exists, and the attachment landed in it (ADR §2(c)).
    assert run.cwd.is_dir()
    assert run.cwd.name == "att1"
    assert len(run.attachments) == 1
    landed = run.attachments[0]
    assert landed.exists()
    assert landed.parent == run.cwd
    assert landed.read_text() == "spreadsheet bytes"
    assert run.file_name == "data.xlsx"


def test_prepare_task_dir_no_attachment(tmp_path):
    run = prepare_task_dir(
        task_id="noatt",
        question="capital of France?",
        ground_truth="Paris",
        level=1,
        source_file=None,
        work_root=tmp_path / "work",
    )
    assert run.cwd.is_dir()
    assert run.attachments == []
    assert run.file_name is None


def test_build_prompt_includes_files_and_sentinel(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,b", encoding="utf-8")
    run = prepare_task_dir(
        task_id="p1",
        question="what is in the file?",
        ground_truth="x",
        level=2,
        source_file=src,
        work_root=tmp_path / "work",
    )
    prompt = build_prompt(run)
    assert "what is in the file?" in prompt
    assert "data.csv" in prompt
    assert "working directory" in prompt
    assert "<final_answer>" in prompt


def test_build_prompt_no_files(tmp_path):
    run = prepare_task_dir(
        task_id="p2",
        question="capital of France?",
        ground_truth="Paris",
        level=1,
        source_file=None,
        work_root=tmp_path / "work",
    )
    prompt = build_prompt(run)
    assert "capital of France?" in prompt
    assert "<final_answer>" in prompt
    # No file listing when there are no attachments.
    assert "working directory" not in prompt


# --------------------------------------------------------------------------- #
# The default spawn path pins the model (ADR §7) — assert it never uses inherit.
# --------------------------------------------------------------------------- #
def test_default_model_is_pinned_not_inherit():
    assert run_subset.DEFAULT_MODEL not in (None, "", "inherit")

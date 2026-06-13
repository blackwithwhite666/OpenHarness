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


@pytest.mark.asyncio
async def test_run_subset_writes_jsonl_incrementally(tmp_path):
    # Each task's row must be persisted BEFORE the next task runs — crash-safety +
    # live progress on a long (90-run) sweep. The spawn_fn observes the on-disk row
    # count as each task starts.
    results = tmp_path / "results"
    jsonl = results / "shaX.jsonl"
    tasks = [_task(f"t{i}", "Paris", 1, tmp_path) for i in range(3)]
    rows_on_disk_at_start: dict[str, int] = {}

    async def spawn(t, model):
        rows_on_disk_at_start[t.task_id] = (
            len(jsonl.read_text().splitlines()) if jsonl.exists() else 0
        )
        return "<final_answer>Paris</final_answer>"

    await run_subset_fn(
        tasks, spawn_fn=spawn, results_dir=results, git_sha="shaX", k=1
    )
    # t0 sees nothing yet; t1 sees t0's row; t2 sees t0+t1.
    assert rows_on_disk_at_start == {"t0": 0, "t1": 1, "t2": 2}
    assert len(jsonl.read_text().splitlines()) == 3


def test_parse_usage_strips_marker_and_sums():
    from tests.eval.gaia.run_subset import _parse_usage

    clean, tok = _parse_usage("answer here\n[[USAGE input_tokens=1200 output_tokens=300]]")
    assert tok == 1500
    assert "[[USAGE" not in clean and clean.strip() == "answer here"
    assert _parse_usage("no marker") == ("no marker", None)
    assert _parse_usage(None) == (None, None)


@pytest.mark.asyncio
async def test_run_subset_records_tokens_from_usage_marker(tmp_path):
    # The worker (opt-in OPENHARNESS_EMIT_USAGE) appends a [[USAGE …]] marker; the
    # runner must parse it into row["tokens"] AND strip it so the scorer still
    # extracts the answer (the marker must never be mistaken for the answer).
    task = _task("t-usage", "Paris", 1, tmp_path)

    async def spawn_with_usage(t, model):
        return (
            f"Reasoning...\n<final_answer>{t.ground_truth}</final_answer>\n"
            "[[USAGE input_tokens=1200 output_tokens=300]]"
        )

    await run_subset_fn(
        [task],
        spawn_fn=spawn_with_usage,
        results_dir=tmp_path / "results",
        git_sha="sha-usage",
        k=1,
    )
    (row,) = [
        json.loads(line)
        for line in (tmp_path / "results" / "sha-usage.jsonl").read_text().splitlines()
    ]
    assert row["score"] == 1.0   # marker stripped -> answer still extracted
    assert row["tokens"] == 1500  # input+output summed and recorded


@pytest.mark.asyncio
async def test_spawn_agent_kills_worker_on_timeout(monkeypatch):
    # A timed-out worker must be killed (manager.stop_task) before InfraFailure
    # propagates — otherwise heavy deep-research workers leak and saturate the host.
    stopped: list[str] = []

    class _Rec:
        status = "running"  # never terminal -> _wait_terminal times out

    class _Mgr:
        def get_task(self, tid):
            return _Rec()

        async def stop_task(self, tid):
            stopped.append(tid)

        def read_task_output(self, tid, max_bytes=0):
            return ""

    class _Res:
        success = True
        task_id = "t-leak"
        error = None

    class _Exec:
        async def spawn(self, config):
            return _Res()

    class _Reg:
        def get_executor(self, name):
            return _Exec()

    monkeypatch.setattr("openharness.tasks.get_task_manager", lambda: _Mgr())
    monkeypatch.setattr("openharness.swarm.registry.get_backend_registry", lambda: _Reg())
    monkeypatch.setattr(
        "openharness.coordinator.agent_definitions.get_agent_definition", lambda name: None
    )

    task = run_subset.TaskRun(
        task_id="x", question="q", ground_truth="a", level=1, cwd="/tmp"
    )
    with pytest.raises(InfraFailure):
        await run_subset._spawn_agent(
            task, "gpt-5.5", subagent_type="deep-research", timeout_s=0.05
        )
    assert stopped == ["t-leak"]  # the leaked worker was killed


@pytest.mark.asyncio
async def test_wait_terminal_kills_on_low_memory(monkeypatch):
    # Free memory below the floor -> InfraFailure immediately (host protection),
    # before the (large) timeout would fire. A ballooning worker must not OOM the box.
    monkeypatch.setattr(run_subset, "_free_mem_mb", lambda: 100.0)

    class _Rec:
        status = "running"

    class _Mgr:
        def get_task(self, tid):
            return _Rec()

    with pytest.raises(InfraFailure) as exc:
        await run_subset._wait_terminal(_Mgr(), "t1", timeout_s=999, min_free_mem_mb=800)
    assert "free memory" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_wait_terminal_no_mem_guard_when_unavailable(monkeypatch):
    # _free_mem_mb -> None (non-Linux / CI): the guard is a no-op; a terminal task
    # still returns normally.
    monkeypatch.setattr(run_subset, "_free_mem_mb", lambda: None)

    class _Rec:
        status = "completed"

    class _Mgr:
        def get_task(self, tid):
            return _Rec()

    assert await run_subset._wait_terminal(_Mgr(), "t1", timeout_s=5) == "completed"

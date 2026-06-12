"""Offline tests for the GAIA loader (ADR §4, component #3).

Zero network, zero HF token, zero real dataset. The gated ``snapshot_download``
is NEVER invoked — instead we build a FAKE snapshot dir (a tiny
``2023/validation/metadata.jsonl`` + a dummy attachment) in ``tmp_path`` and
assert the offline parser handles parsing, the ``0-0-0-0-0`` skip, and
``file_name`` absolutization correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval.gaia.loader import (
    EXAMPLE_TASK_ID,
    GAIA_CONFIG,
    GAIA_SPLIT,
    Task,
    absolutize_file_path,
    download_gaia_snapshot,
    get_hf_token,
    load_validation_tasks,
    split_dir,
)


def _write_fake_snapshot(root: Path, rows: list[dict], *, attachments: dict[str, str] | None = None) -> Path:
    """Materialize a fake GAIA snapshot: 2023/validation/{metadata.jsonl, files}."""
    vdir = root / GAIA_CONFIG / GAIA_SPLIT
    vdir.mkdir(parents=True, exist_ok=True)
    with (vdir / "metadata.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    for name, content in (attachments or {}).items():
        (vdir / name).write_text(content, encoding="utf-8")
    return root


# A realistic validation example (from the dataset spec) + the sentinel + a
# no-attachment task.
SENTINEL_ROW = {
    "task_id": EXAMPLE_TASK_ID,
    "Question": "example/demo task — must be skipped",
    "Level": 1,
    "Final answer": "n/a",
    "file_name": "",
}
FILE_ROW = {
    "task_id": "32102e3e-d12a-4209-9163-7b3a104efe5d",
    "Question": "The attached spreadsheet shows the inventory ... oldest Blu-Ray ...",
    "Level": 2,
    "Final answer": "Time-Parking 2: Parallel Universe",
    "file_name": "32102e3e-d12a-4209-9163-7b3a104efe5d.xlsx",
}
NOFILE_ROW = {
    "task_id": "aaaa-1111",
    "Question": "What is the capital of France?",
    "Level": 1,
    "Final answer": "Paris",
    "file_name": "",
}


def test_loads_and_skips_sentinel(tmp_path):
    root = _write_fake_snapshot(
        tmp_path,
        [SENTINEL_ROW, FILE_ROW, NOFILE_ROW],
        attachments={FILE_ROW["file_name"]: "dummy xlsx bytes"},
    )
    tasks = load_validation_tasks(root)

    # Sentinel skipped -> 2 real tasks.
    assert len(tasks) == 2
    ids = {t.task_id for t in tasks}
    assert EXAMPLE_TASK_ID not in ids
    assert ids == {FILE_ROW["task_id"], NOFILE_ROW["task_id"]}


def test_record_shape_and_field_mapping(tmp_path):
    root = _write_fake_snapshot(
        tmp_path, [NOFILE_ROW], attachments=None
    )
    (task,) = load_validation_tasks(root)
    assert isinstance(task, Task)
    assert task.task_id == "aaaa-1111"
    assert task.question == "What is the capital of France?"
    assert task.answer == "Paris"
    assert task.level == 1
    assert isinstance(task.level, int)
    assert task.file_path is None


def test_file_path_absolutized_and_exists(tmp_path):
    root = _write_fake_snapshot(
        tmp_path,
        [FILE_ROW],
        attachments={FILE_ROW["file_name"]: "dummy xlsx bytes"},
    )
    (task,) = load_validation_tasks(root)
    assert task.file_path is not None
    assert task.file_path.is_absolute()
    # Resolves under 2023/validation and the copied dummy actually exists.
    assert task.file_path.parent == split_dir(root).resolve()
    assert task.file_path.exists()
    assert task.file_path.name == FILE_ROW["file_name"]


def test_empty_file_name_yields_none(tmp_path):
    assert absolutize_file_path(tmp_path, "") is None
    assert absolutize_file_path(tmp_path, None) is None


def test_absolutize_joins_under_split_dir(tmp_path):
    p = absolutize_file_path(tmp_path, "foo.csv")
    assert p == (tmp_path / GAIA_CONFIG / GAIA_SPLIT / "foo.csv").resolve()


def test_blank_lines_in_metadata_are_skipped(tmp_path):
    vdir = tmp_path / GAIA_CONFIG / GAIA_SPLIT
    vdir.mkdir(parents=True)
    lines = [json.dumps(NOFILE_ROW), "", "   ", json.dumps(FILE_ROW)]
    (vdir / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (vdir / FILE_ROW["file_name"]).write_text("x", encoding="utf-8")
    tasks = load_validation_tasks(tmp_path)
    assert len(tasks) == 2


def test_missing_metadata_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_validation_tasks(tmp_path)


def test_level_coerced_from_string(tmp_path):
    row = {**NOFILE_ROW, "Level": "3"}
    root = _write_fake_snapshot(tmp_path, [row])
    (task,) = load_validation_tasks(root)
    assert task.level == 3


def test_missing_final_answer_defaults_empty(tmp_path):
    # Test-split rows have a blank "Final answer"; loader tolerates absence.
    row = {k: v for k, v in NOFILE_ROW.items() if k != "Final answer"}
    root = _write_fake_snapshot(tmp_path, [row])
    (task,) = load_validation_tasks(root)
    assert task.answer == ""


# --------------------------------------------------------------------------- #
# Token gating: the network path stays gated and is never invoked here.
# --------------------------------------------------------------------------- #
def test_get_hf_token_explicit_arg_wins():
    assert get_hf_token("explicit-tok") == "explicit-tok"


def test_get_hf_token_reads_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "env-tok")
    assert get_hf_token() == "env-tok"


def test_get_hf_token_raises_without_any(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        get_hf_token()


def test_download_is_gated_behind_token(monkeypatch):
    # Without a token, download_gaia_snapshot must raise BEFORE any network /
    # huggingface_hub import — the unit suite never hits the gated dataset.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        download_gaia_snapshot()

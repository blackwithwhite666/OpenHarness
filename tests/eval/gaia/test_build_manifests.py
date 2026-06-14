"""Offline tests for the stratified manifest builder (ADR §4, component #2).

Zero network, zero dataset. The selection logic is exercised against a SYNTHETIC
task list (not the real HF dataset). Asserts the disjoint-by-construction
property, the per-level shape, and determinism by seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.eval.gaia.build_manifests import (
    DEFAULT_PER_LEVEL,
    ManifestRow,
    manifest_payload,
    select_splits,
    write_manifest,
)
from tests.eval.gaia.loader import Task


def _synthetic_tasks(per_level: dict[int, int], *, with_files_level: int | None = None) -> list[Task]:
    """Build a synthetic task list with enough tasks per level for two splits."""
    tasks: list[Task] = []
    for level, want in per_level.items():
        for i in range(2 * want):  # exactly enough for dev+gate
            fp = None
            if with_files_level == level and i % 2 == 0:
                fp = Path(f"/snap/2023/validation/L{level}_{i}.xlsx")
            tasks.append(
                Task(
                    task_id=f"L{level}-task-{i:03d}",
                    question=f"q{level}-{i}",
                    answer=f"a{level}-{i}",
                    level=level,
                    file_path=fp,
                )
            )
    return tasks


def test_disjoint_and_per_level_shape():
    tasks = _synthetic_tasks(DEFAULT_PER_LEVEL)
    dev, gate = select_splits(tasks)

    dev_ids = {r.task_id for r in dev}
    gate_ids = {r.task_id for r in gate}

    # Disjoint by construction.
    assert dev_ids.isdisjoint(gate_ids)
    # Per-level shape L1=10 / L2=15 / L3=5 on BOTH splits.
    for split in (dev, gate):
        counts = {1: 0, 2: 0, 3: 0}
        for r in split:
            counts[r.level] += 1
        assert counts == DEFAULT_PER_LEVEL
    assert len(dev) == 30
    assert len(gate) == 30


def test_deterministic_by_seed():
    tasks = _synthetic_tasks(DEFAULT_PER_LEVEL)
    dev_a, gate_a = select_splits(tasks, seed=1729)
    dev_b, gate_b = select_splits(tasks, seed=1729)
    assert [r.task_id for r in dev_a] == [r.task_id for r in dev_b]
    assert [r.task_id for r in gate_a] == [r.task_id for r in gate_b]


def test_different_seed_changes_selection():
    tasks = _synthetic_tasks(DEFAULT_PER_LEVEL)
    dev_a, _ = select_splits(tasks, seed=1)
    dev_b, _ = select_splits(tasks, seed=2)
    # Overwhelmingly likely to differ; assert at least the ordering changed.
    assert [r.task_id for r in dev_a] != [r.task_id for r in dev_b]


def test_independent_of_input_order():
    tasks = _synthetic_tasks(DEFAULT_PER_LEVEL)
    shuffled = list(reversed(tasks))
    dev_a, gate_a = select_splits(tasks, seed=1729)
    dev_b, gate_b = select_splits(shuffled, seed=1729)
    # Pre-sort by task_id makes selection order-independent.
    assert {r.task_id for r in dev_a} == {r.task_id for r in dev_b}
    assert {r.task_id for r in gate_a} == {r.task_id for r in gate_b}


def test_raises_when_too_few_tasks():
    # Only enough for one split, not two disjoint ones.
    tasks = _synthetic_tasks({1: 10, 2: 15, 3: 5})
    # Drop one L3 task so we can't fill 2*5 = 10.
    tasks = [t for t in tasks if not (t.level == 3 and t.task_id.endswith("009"))]
    with pytest.raises(ValueError):
        select_splits(tasks)


def test_needs_file_and_file_name_propagate():
    tasks = _synthetic_tasks(DEFAULT_PER_LEVEL, with_files_level=2)
    dev, gate = select_splits(tasks)
    all_rows = dev + gate
    file_rows = [r for r in all_rows if r.needs_file]
    assert file_rows, "expected some file-bearing rows from level 2"
    for r in file_rows:
        assert r.file_name is not None
        assert r.file_name.endswith(".xlsx")
    # Non-file rows carry file_name=None.
    for r in all_rows:
        if not r.needs_file:
            assert r.file_name is None


def test_write_manifest_roundtrip(tmp_path):
    rows = [
        ManifestRow(task_id="t1", level=1, needs_file=False, file_name=None),
        ManifestRow(task_id="t2", level=2, needs_file=True, file_name="t2.xlsx"),
    ]
    path = write_manifest(tmp_path / "dev.yaml", "dev", rows)
    loaded = yaml.safe_load(path.read_text())
    assert loaded["split"] == "dev"
    assert loaded["target_counts"] == {"level_1": 1, "level_2": 1, "level_3": 0}
    assert [t["task_id"] for t in loaded["tasks"]] == ["t1", "t2"]
    assert loaded["tasks"][1]["needs_file"] is True
    assert loaded["tasks"][1]["file_name"] == "t2.xlsx"


def test_manifest_payload_capability_tags():
    rows = [
        ManifestRow(task_id="a", level=1, needs_file=False, file_name=None),
        ManifestRow(task_id="b", level=2, needs_file=False, file_name=None),
        ManifestRow(task_id="c", level=2, needs_file=True, file_name="c.csv"),
    ]
    payload = manifest_payload("gate", rows)
    caps = {t["task_id"]: t["capability"] for t in payload["tasks"]}
    assert caps["a"] == "web-browsing"
    assert caps["b"] == "multi-hop"
    assert caps["c"] == "file-attachment"


def test_select_single_split_shape_and_determinism():
    from tests.eval.gaia.build_manifests import select_single_split

    tasks = _synthetic_tasks({1: 10, 2: 35, 3: 15})  # 2x each -> enough
    per = {1: 10, 2: 35, 3: 15}
    rows = select_single_split(tasks, per_level=per, seed=2027)
    counts = {1: 0, 2: 0, 3: 0}
    for r in rows:
        counts[r.level] += 1
    assert counts == per
    assert len(rows) == 60
    # Deterministic by seed; different seed shifts selection.
    again = select_single_split(tasks, per_level=per, seed=2027)
    assert [r.task_id for r in rows] == [r.task_id for r in again]
    other = select_single_split(tasks, per_level=per, seed=99)
    assert [r.task_id for r in rows] != [r.task_id for r in other]


def test_select_single_split_raises_when_too_few():
    from tests.eval.gaia.build_manifests import select_single_split

    tasks = _synthetic_tasks({1: 3, 2: 3, 3: 3})  # only 6 per level
    with pytest.raises(ValueError):
        select_single_split(tasks, per_level={1: 10, 2: 10, 3: 10}, seed=1)

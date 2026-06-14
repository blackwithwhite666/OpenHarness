"""Stratified disjoint dev+gate manifest builder (ADR §4, component #2).

Given the loaded GAIA validation tasks, select two **disjoint** splits — dev and
gate, 30 tasks each, per-level ``L1=10 / L2=15 / L3=5`` (the §4
capability-coverage shape) — by a FIXED seed, and write ``dev.yaml`` /
``gate.yaml``.

Two surfaces, split by what touches the dataset:

* :func:`select_splits` — pure / offline stratified selection. Deterministic
  given the same ``tasks`` + ``seed``. dev and gate share no ``task_id``.
  Unit-tested against a synthetic task list (no HF token, no real dataset).
* :func:`build_and_write` / :func:`main` — the CLI entry point. Needs the
  dataset (HF_TOKEN) to load the real tasks, then calls :func:`select_splits`
  and :func:`write_manifest`.

Dependency direction (ADR §2½): imports the loader (which drives HF), never the
reverse.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml

from tests.eval.gaia.loader import (
    Task,
    download_gaia_snapshot,
    load_validation_tasks,
)

# Default split shape (ADR §1, §4): 30 tasks each, L1=10 / L2=15 / L3=5.
DEFAULT_PER_LEVEL = {1: 10, 2: 15, 3: 5}
DEFAULT_SEED = 1729

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ManifestRow:
    """One manifest row (subset of :class:`~tests.eval.gaia.loader.Task`)."""

    task_id: str
    level: int
    needs_file: bool
    file_name: str | None


def _to_row(task: Task) -> ManifestRow:
    fp = task.file_path
    return ManifestRow(
        task_id=task.task_id,
        level=task.level,
        needs_file=fp is not None,
        file_name=fp.name if fp is not None else None,
    )


def select_splits(
    tasks: Sequence[Task],
    *,
    per_level: dict[int, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Pick disjoint dev + gate splits, stratified by level, by a fixed seed.

    For each level the available tasks are shuffled with a seeded RNG, then the
    first ``per_level[level]`` go to dev and the *next* ``per_level[level]`` go
    to gate — so the two splits are DISJOINT by construction (no shared
    ``task_id``). Deterministic: same ``tasks`` + ``seed`` -> identical splits.

    Raises ``ValueError`` if any level lacks ``2 * per_level[level]`` tasks
    (can't fill two disjoint splits of that size).
    """
    per_level = dict(per_level or DEFAULT_PER_LEVEL)
    rng = random.Random(seed)

    by_level: dict[int, list[Task]] = {}
    for task in tasks:
        by_level.setdefault(task.level, []).append(task)

    dev: list[ManifestRow] = []
    gate: list[ManifestRow] = []
    for level, want in sorted(per_level.items()):
        pool = list(by_level.get(level, []))
        if len(pool) < 2 * want:
            raise ValueError(
                f"Level {level}: need {2 * want} tasks for two disjoint splits "
                f"of {want}, only {len(pool)} available."
            )
        # Sort by task_id first for a stable pre-shuffle order, then shuffle
        # with the seeded RNG — so the result depends only on (tasks, seed),
        # not on the input iteration order.
        pool.sort(key=lambda t: t.task_id)
        rng.shuffle(pool)
        dev.extend(_to_row(t) for t in pool[:want])
        gate.extend(_to_row(t) for t in pool[want : 2 * want])

    return dev, gate


def select_single_split(
    tasks: Sequence[Task],
    *,
    per_level: dict[int, int],
    seed: int = DEFAULT_SEED,
) -> list[ManifestRow]:
    """Pick ONE stratified split of ``per_level`` tasks per level, by a fixed seed.

    Same deterministic (sort by task_id -> seeded shuffle -> take first N) selection
    as :func:`select_splits`, but for a single split with an arbitrary per-level shape
    (e.g. a hard-weighted ``bench60`` = L1 10 / L2 35 / L3 15). Raises ``ValueError``
    if a level lacks enough tasks.
    """
    rng = random.Random(seed)
    by_level: dict[int, list[Task]] = {}
    for task in tasks:
        by_level.setdefault(task.level, []).append(task)

    rows: list[ManifestRow] = []
    for level, want in sorted(per_level.items()):
        pool = list(by_level.get(level, []))
        if len(pool) < want:
            raise ValueError(
                f"Level {level}: need {want} tasks, only {len(pool)} available."
            )
        pool.sort(key=lambda t: t.task_id)
        rng.shuffle(pool)
        rows.extend(_to_row(t) for t in pool[:want])
    return rows


def _classify_capability(row: ManifestRow) -> str:
    """Heuristic capability tag for the manifest (display/coverage only).

    The real capability taxonomy is hand-curated in the ADR; here we emit a
    coarse machine tag so the written manifest is self-describing. File-bearing
    tasks are ``file-attachment``; otherwise level is a proxy (L1 web-browsing,
    L2 multi-hop, L3 multi-hop).
    """
    if row.needs_file:
        return "file-attachment"
    if row.level == 1:
        return "web-browsing"
    return "multi-hop"


def manifest_payload(split: str, rows: Sequence[ManifestRow]) -> dict:
    """Build the YAML-serializable dict for one split."""
    counts = {1: 0, 2: 0, 3: 0}
    for row in rows:
        counts[row.level] = counts.get(row.level, 0) + 1
    return {
        "split": split,
        "seed": DEFAULT_SEED,
        "target_counts": {
            "level_1": counts.get(1, 0),
            "level_2": counts.get(2, 0),
            "level_3": counts.get(3, 0),
        },
        "tasks": [
            {
                "task_id": row.task_id,
                "level": row.level,
                "capability": _classify_capability(row),
                "needs_file": row.needs_file,
                "file_name": row.file_name,
            }
            for row in rows
        ],
    }


def write_manifest(path: Path | str, split: str, rows: Sequence[ManifestRow]) -> Path:
    """Write one split manifest to ``path`` as YAML. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest_payload(split, rows)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            f"# GAIA {split} split manifest — generated by build_manifests.py "
            f"(seed={DEFAULT_SEED}).\n"
            "# Disjoint from the other split by construction; do not hand-edit.\n\n"
        )
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)
    return path


def build_and_write(
    snapshot_root: Path | str,
    *,
    out_dir: Path | str = _HERE,
    per_level: dict[int, int] | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[Path, Path]:
    """Load tasks from a snapshot, select splits, and write both manifests."""
    tasks = load_validation_tasks(snapshot_root)
    dev, gate = select_splits(tasks, per_level=per_level, seed=seed)
    out_dir = Path(out_dir)
    dev_path = write_manifest(out_dir / "dev.yaml", "dev", dev)
    gate_path = write_manifest(out_dir / "gate.yaml", "gate", gate)
    return dev_path, gate_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: download (or reuse) the GAIA snapshot and write dev/gate manifests.

    Needs HF_TOKEN (the loader is HF-gated). Example::

        python -m tests.eval.gaia.build_manifests --snapshot /path/to/gaia
        python -m tests.eval.gaia.build_manifests           # downloads first
    """
    parser = argparse.ArgumentParser(description="Build GAIA dev/gate manifests.")
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Path to an already-downloaded GAIA snapshot root. If omitted, "
        "downloads it (HF_TOKEN required).",
    )
    parser.add_argument(
        "--out-dir", default=str(_HERE), help="Where to write dev.yaml / gate.yaml."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--single-name",
        default=None,
        help="Emit ONE split with this name (<name>.yaml) instead of dev/gate.",
    )
    parser.add_argument(
        "--per-level",
        default=None,
        help="Comma L1,L2,L3 counts for --single-name, e.g. '10,35,15'.",
    )
    args = parser.parse_args(argv)

    snapshot_root = (
        Path(args.snapshot) if args.snapshot else download_gaia_snapshot()
    )

    if args.single_name:
        l1, l2, l3 = (int(x) for x in (args.per_level or "10,35,15").split(","))
        tasks = load_validation_tasks(snapshot_root)
        rows = select_single_split(
            tasks, per_level={1: l1, 2: l2, 3: l3}, seed=args.seed
        )
        path = write_manifest(
            Path(args.out_dir) / f"{args.single_name}.yaml", args.single_name, rows
        )
        print(f"wrote {path} ({len(rows)} tasks: L1={l1} L2={l2} L3={l3}, seed={args.seed})")
        return 0

    dev_path, gate_path = build_and_write(
        snapshot_root, out_dir=args.out_dir, seed=args.seed
    )
    print(f"wrote {dev_path}")
    print(f"wrote {gate_path}")
    return 0


# Bare sys.exit(main()) — PY3_PROGRAM/entry-point friendly (no __main__ guard
# skip; ADR project conventions). Harmless for `python -m`.
if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())

"""GAIA dataset loader (SCAFFOLD — actual download is gated/stubbed).

Layer 3 of the deep-research architecture (ADR §4, component #3): read the GAIA
``2023/validation`` split, skip the example task ``0-0-0-0-0``, and absolutize
each task's ``file_path`` so the attachment contract (ADR §2(c)) can copy the
file into the per-task ``cwd``.

The download itself is STUBBED: GAIA is HF-gated (``snapshot_download(
"gaia-benchmark/GAIA")`` needs ``HF_TOKEN`` + accepted terms; an un-gated /
tokenless call 401s — ADR §7). That access is not configured in PR1, so the
real download raises ``NotImplementedError``; everything around it
(path-absolutization, the example-task skip, the record shape) is specified so
PR2 can drop the download in without reshaping callers.

Dependency direction (ADR §2½): imports OpenHarness/HF, never the reverse.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# The GAIA example/demo task — always skipped (ADR §4).
EXAMPLE_TASK_ID = "0-0-0-0-0"

# HF dataset repo + the split we evaluate (validation; test answers are
# leaderboard-hidden — ADR §7).
GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_CONFIG = "2023"
GAIA_SPLIT = "validation"


@dataclass(frozen=True)
class GaiaTask:
    """One GAIA validation task, post-absolutization.

    ``file_path`` is ``None`` when the task carries no attachment, else an
    absolute path into the downloaded snapshot. ~50% of validation tasks carry
    a file (ADR §2(c)); a missing file in ``cwd`` is a *harness bug*, not a
    model miss (ADR §4) — callers must assert presence before trusting a score.
    """

    task_id: str
    question: str
    ground_truth: str
    level: int
    file_name: str | None
    file_path: Path | None


def get_hf_token() -> str:
    """Return the HF token from the environment, or raise.

    Checks ``HF_TOKEN`` then ``HUGGING_FACE_HUB_TOKEN``. GAIA is gated; without
    a token + accepted terms the snapshot download 401s (ADR §7).
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "GAIA is HF-gated: set HF_TOKEN (and accept the dataset terms at "
            "https://huggingface.co/datasets/gaia-benchmark/GAIA). Not "
            "configured in PR1 — the loader download is stubbed."
        )
    return token


def download_gaia_snapshot(
    *,
    repo_id: str = GAIA_REPO_ID,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download the GAIA snapshot and return its local root.

    TODO(PR2): implement with::

        from huggingface_hub import snapshot_download
        local = snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            token=token or get_hf_token(),
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        return Path(local)

    STUBBED in PR1 — the gated dataset is not available here. Kept as an
    explicit seam so the loader's *shape* (and the tests around path handling)
    can land now and PR2 only fills this one function.
    """
    raise NotImplementedError(
        "GAIA snapshot download is stubbed (HF-gated, not configured in PR1). "
        "See docstring for the snapshot_download call to wire up in PR2."
    )


def absolutize_file_path(snapshot_root: Path, file_name: str | None) -> Path | None:
    """Resolve a task's relative ``file_name`` against the snapshot root.

    Returns ``None`` for tasks without an attachment. Real implementation
    joins ``snapshot_root / GAIA_CONFIG / GAIA_SPLIT / file_name`` (the layout
    PR2 confirms against the downloaded tree) and resolves to absolute. Pure /
    offline — safe to unit-test once the layout is pinned.
    """
    if not file_name:
        return None
    return (snapshot_root / GAIA_CONFIG / GAIA_SPLIT / file_name).resolve()


def load_validation_tasks(
    snapshot_root: Path | None = None,
) -> Iterator[GaiaTask]:
    """Yield ``GaiaTask`` records for ``2023/validation``, example skipped.

    TODO(PR2): read ``<snapshot_root>/<2023>/<validation>/metadata.jsonl``,
    for each record:
      * skip ``task_id == EXAMPLE_TASK_ID`` (the demo task);
      * map ``Question`` / ``Final answer`` / ``Level`` / ``file_name``;
      * ``file_path = absolutize_file_path(snapshot_root, file_name)``.

    STUBBED in PR1 (download gated). The manifests (``dev.yaml`` / ``gate.yaml``)
    pin the actual ``task_id``s once this is wired and the splits are frozen.
    """
    raise NotImplementedError(
        "load_validation_tasks is stubbed until HF access is configured (PR2). "
        "It will read 2023/validation metadata, skip 0-0-0-0-0, and absolutize "
        "file_path; the manifest task_ids are populated from its output."
    )

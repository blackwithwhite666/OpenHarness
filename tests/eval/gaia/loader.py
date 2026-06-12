"""GAIA dataset loader — gated download + offline metadata parsing.

Layer 3 of the deep-research architecture (ADR §4, component #3): read the GAIA
``2023/validation`` split, skip the example task ``0-0-0-0-0``, and absolutize
each task's ``file_name`` so the attachment contract (ADR §2(c)) can copy the
file into the per-task ``cwd``.

Two surfaces, split by what touches the network:

* :func:`download_gaia_snapshot` — the ONE network call. GAIA is HF-gated
  (``snapshot_download("gaia-benchmark/GAIA")`` needs ``HF_TOKEN`` + accepted
  terms; an un-gated / tokenless call 401s — ADR §7). ``huggingface_hub`` is
  imported lazily *inside* this function so the module imports (and the offline
  parser is testable) without the dependency installed.
* :func:`load_validation_tasks` — pure, offline metadata parsing. Reads
  ``<snapshot_root>/2023/validation/metadata.jsonl``, skips the sentinel,
  absolutizes ``file_name``. Unit-tested against a FAKE snapshot dir (no
  network, no HF token, no real dataset).

Dependency direction (ADR §2½): imports OpenHarness/HF, never the reverse.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Task",
    "EXAMPLE_TASK_ID",
    "GAIA_REPO_ID",
    "GAIA_CONFIG",
    "GAIA_SPLIT",
    "get_hf_token",
    "download_gaia_snapshot",
    "split_dir",
    "absolutize_file_path",
    "load_validation_tasks",
]

# The GAIA example/demo task — always skipped (ADR §4, owl/utils/gaia.py).
EXAMPLE_TASK_ID = "0-0-0-0-0"

# HF dataset repo + the config/split we evaluate (validation; test answers are
# leaderboard-hidden — ADR §7).
GAIA_REPO_ID = "gaia-benchmark/GAIA"
GAIA_CONFIG = "2023"
GAIA_SPLIT = "validation"

_METADATA_FILE = "metadata.jsonl"


@dataclass(frozen=True)
class Task:
    """One GAIA validation task, post-absolutization.

    ``file_path`` is ``None`` when the task carries no attachment, else an
    absolute path into the downloaded snapshot. ~50% of validation tasks carry
    a file (ADR §2(c)); a missing file in ``cwd`` is a *harness bug*, not a
    model miss (ADR §4) — callers must assert presence before trusting a score.
    """

    task_id: str
    question: str
    answer: str
    level: int
    file_path: Path | None = None


def get_hf_token(token: str | None = None) -> str:
    """Return an HF token (explicit arg, then env), or raise.

    Checks the explicit ``token`` argument first, then ``HF_TOKEN`` /
    ``HUGGING_FACE_HUB_TOKEN``. GAIA is gated; without a token + accepted terms
    the snapshot download 401s (ADR §7).
    """
    resolved = (
        token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not resolved:
        raise RuntimeError(
            "GAIA is HF-gated: set HF_TOKEN (and accept the dataset terms at "
            "https://huggingface.co/datasets/gaia-benchmark/GAIA), or pass an "
            "explicit token=."
        )
    return resolved


def download_gaia_snapshot(
    token: str | None = None,
    *,
    repo_id: str = GAIA_REPO_ID,
    local_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
) -> Path:
    """Materialize the GAIA dataset repo tree and return its local root.

    This is the ONE function that touches the network. It is GATED behind the
    HF token (:func:`get_hf_token` raises without one) and is never invoked from
    the unit tests — the offline parser :func:`load_validation_tasks` is tested
    against a synthetic snapshot dir instead.

    ``huggingface_hub`` is imported lazily so this module (and the offline
    parser) imports cleanly even when the dependency is absent.
    """
    resolved_token = get_hf_token(token)
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "huggingface_hub is required to download GAIA: "
            "`pip install huggingface_hub`."
        ) from exc

    local = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=resolved_token,
        local_dir=str(local_dir) if local_dir is not None else None,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return Path(local)


def split_dir(snapshot_root: Path | str, split: str = GAIA_SPLIT) -> Path:
    """Return ``<snapshot_root>/2023/<split>`` (where metadata + files live)."""
    return Path(snapshot_root) / GAIA_CONFIG / split


def absolutize_file_path(
    snapshot_root: Path | str, file_name: str | None, *, split: str = GAIA_SPLIT
) -> Path | None:
    """Resolve a task's bare ``file_name`` against the split dir.

    Returns ``None`` for tasks without an attachment (``""`` / ``None``). A
    row's ``file_name`` is a bare filename resolved relative to the split dir
    (``2023/validation``) — owl's loader does the same join + absolutize.
    """
    if not file_name:
        return None
    return (split_dir(snapshot_root, split) / file_name).resolve()


def load_validation_tasks(
    snapshot_root: Path | str,
    *,
    split: str = GAIA_SPLIT,
) -> list[Task]:
    """Read ``2023/<split>/metadata.jsonl`` into :class:`Task` records.

    Pure / offline (no network, no HF token). For each JSONL row:

    * skip blank lines and the sentinel ``task_id == "0-0-0-0-0"`` (the GAIA
      demo task — owl ``continue``s on it);
    * map ``Question`` / ``Final answer`` / ``Level`` / ``file_name``;
    * absolutize a bare ``file_name`` into an absolute ``file_path`` under the
      split dir (``None`` when the row has no attachment).

    ``Final answer`` is blank on the ``test`` split (private); on ``validation``
    it is the ground truth. ``Level`` is coerced to ``int`` (1/2/3).
    """
    metadata_path = split_dir(snapshot_root, split) / _METADATA_FILE
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"GAIA metadata not found: {metadata_path}. Did you run "
            "download_gaia_snapshot() first (HF_TOKEN required)?"
        )

    tasks: list[Task] = []
    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("task_id") == EXAMPLE_TASK_ID:
                continue
            tasks.append(
                Task(
                    task_id=data["task_id"],
                    question=data["Question"],
                    answer=data.get("Final answer", ""),
                    level=int(data["Level"]),
                    file_path=absolutize_file_path(
                        snapshot_root, data.get("file_name"), split=split
                    ),
                )
            )
    return tasks

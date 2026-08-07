from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

from ohmo.nutrition_ingest.models import candidate_id_for
from ohmo.nutrition_ingest.sidecars import NutritionResultStore
from ohmo.nutrition_ingest.watcher import NutritionArtifactScanner


def _write_candidate(root: Path, *, file_id: str, rev: str, discovery: str, data: bytes) -> Path:
    fixture = json.loads(
        (Path(__file__).parents[2] / "ohmo/nutrition_ingest/manifest_v1_fixture.json").read_text()
    )
    candidate_id = candidate_id_for(file_id, rev)
    directory = root / candidate_id
    directory.mkdir(parents=True)
    filename = f"original.{file_id[-3:]}.jpg"
    (directory / filename).write_bytes(data)
    fixture.update(
        candidate_id=candidate_id,
        file_id=file_id,
        rev=rev,
        discovery_time=discovery,
        original_filename=filename,
        original_size_bytes=len(data),
        original_sha256=hashlib.sha256(data).hexdigest(),
    )
    (directory / "manifest.json").write_text(json.dumps(fixture))
    return directory


def test_scanner_waits_for_complete_image_and_orders_deterministically(tmp_path: Path) -> None:
    root = tmp_path / "nutrition-assets"
    root.mkdir()
    first = _write_candidate(
        root, file_id="id:first", rev="rev:1", discovery="2026-08-05T10:01:00Z", data=b"first"
    )
    second = _write_candidate(
        root, file_id="id:second", rev="rev:1", discovery="2026-08-05T10:00:00Z", data=b"second"
    )
    first_image = next(first.glob("original.*.jpg"))
    first_image.write_bytes(b"partial")
    (root / "_producer").mkdir()
    (root / "_producer" / "manifest.json").write_text("not a candidate")
    (root / ".manifest.tmp").write_text("temporary")

    ready = NutritionArtifactScanner(root).scan_ready()
    assert [item.candidate_id for item in ready] == [second.name]

    first_image.write_bytes(b"first")
    ready = NutritionArtifactScanner(root).scan_ready()
    assert [item.candidate_id for item in ready] == [second.name, first.name]


def test_scanner_rejects_symlinked_candidate_directory(tmp_path: Path) -> None:
    root = tmp_path / "nutrition-assets"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    candidate = _write_candidate(
        outside,
        file_id="id:outside",
        rev="rev:outside",
        discovery="2026-08-05T10:00:00Z",
        data=b"outside",
    )
    os.symlink(candidate, root / candidate.name)

    assert NutritionArtifactScanner(root).scan_ready() == []
    assert candidate.is_dir()


def test_result_store_crash_before_replace_preserves_last_good_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "candidate" / "result.json"
    store = NutritionResultStore(path)
    first = _write_sidecar(store, revision=1, state="published")
    from ohmo.nutrition_ingest.models import ResultState, StateHistoryEntry

    second = first.model_validate(
        {
            **first.model_dump(mode="json"),
            "revision": 2,
            "state": ResultState.pending_confirmation,
            "state_history": [
                *[entry.model_dump(mode="json") for entry in first.state_history],
                StateHistoryEntry(
                    revision=2,
                    state=ResultState.pending_confirmation,
                    at="2026-08-05T10:01:00Z",
                ).model_dump(mode="json"),
            ],
        }
    )
    try:
        store.compare_and_replace(second, expected_revision=1, crash_after_flush=True)
    except OSError:
        pass
    assert store.load() == first
    assert not list(path.parent.glob("*.tmp"))


def test_result_store_lock_is_owner_only_under_umask(tmp_path: Path) -> None:
    path = tmp_path / "candidate" / "result.json"
    store = NutritionResultStore(path)
    previous_umask = os.umask(0o0002)
    try:
        _write_sidecar(store, revision=1, state="published")
    finally:
        os.umask(previous_umask)

    lock_path = path.with_name(".result.json.lock")
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def test_result_store_lock_repairs_pre_existing_mode(tmp_path: Path) -> None:
    path = tmp_path / "candidate" / "result.json"
    lock_path = path.with_name(".result.json.lock")
    lock_path.parent.mkdir(parents=True)
    lock_path.touch(mode=0o664)
    os.chmod(lock_path, 0o664)

    _write_sidecar(NutritionResultStore(path), revision=1, state="published")

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600


def _write_sidecar(store: NutritionResultStore, *, revision: int, state: str):
    from ohmo.nutrition_ingest.models import ResultState
    from tests.test_ohmo.test_nutrition_ingest_models import _result

    value = _result(ResultState(state), revision)
    return store.compare_and_replace(value, expected_revision=revision - 1)

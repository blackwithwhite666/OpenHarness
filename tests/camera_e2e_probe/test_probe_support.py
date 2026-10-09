"""Offline guards for the optional private finalizer path."""

from __future__ import annotations

import hashlib
import io
import stat
import subprocess
import sys
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_ohmo"))

from probe_support import (  # noqa: E402
    create_storage_run_dir,
    select_finalizer_event,
    source_jpeg,
    call_wellness_with_synthetic_self,
    unique_honcho_scope,
    verify_source_worktree,
)


class _ProbeWellnessAuthorizationContext:
    def __init__(self):
        self._context = ContextVar("probe_wellness_authorized_read", default=None)

    def get(self):
        return self._context.get()

    def set(self, value):
        return self._context.set(value)

    def reset(self, token):
        self._context.reset(token)


@pytest.mark.asyncio
async def test_direct_probe_wellness_read_resets_authority_after_success_and_failure():
    authorization_context = _ProbeWellnessAuthorizationContext()
    authorized_read = object()

    class App:
        async def call_tool(self, name, arguments):
            assert name == "get_wellness_data"
            assert arguments == {"params": {"start": "synthetic"}}
            assert authorization_context.get() is authorized_read
            return "synthetic result"

    app = App()
    arguments = {"params": {"start": "synthetic"}}
    assert await call_wellness_with_synthetic_self(
        app, authorization_context, lambda _body: authorized_read, arguments
    ) == "synthetic result"
    assert authorization_context.get() is None

    class FailingApp(App):
        async def call_tool(self, name, arguments):
            await super().call_tool(name, arguments)
            raise RuntimeError("synthetic tool failure")

    with pytest.raises(RuntimeError, match="synthetic tool failure"):
        await call_wellness_with_synthetic_self(
            FailingApp(), authorization_context, lambda _body: authorized_read, arguments
        )
    assert authorization_context.get() is None


def test_storage_run_is_persistent_restricted_and_ignored():
    root = Path(__file__).resolve().parents[2]
    parent = root / "tmp" / "camera-normal-chat-docker" / "storage-runs"
    run = create_storage_run_dir(root)
    marker = run / "nutrition.db"
    marker.mkdir()
    (marker / "keep").write_text("synthetic", encoding="ascii")

    assert run.parent == parent
    assert run.is_dir() and marker.is_dir()
    assert (marker / "keep").read_text(encoding="ascii") == "synthetic"
    assert stat.S_IMODE(parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    assert (
        subprocess.run(
            ["git", "check-ignore", "--quiet", "--", str(marker)], cwd=root, check=False
        ).returncode
        == 0
    )


def _synthetic_jpeg(*, trailer: bytes = b"") -> bytes:
    output = io.BytesIO()
    exif = Image.Exif()
    exif[0x010F] = "Synthetic Camera"
    Image.new("RGB", (8, 6), color=(12, 34, 56)).save(output, format="JPEG", exif=exif)
    return output.getvalue() + trailer


def _synthetic_png() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color=(12, 34, 56)).save(output, format="PNG")
    return output.getvalue()


def test_private_source_accepts_appended_data_and_preserves_original_bytes(tmp_path, monkeypatch):
    path = tmp_path / "private.jpg"
    data = _synthetic_jpeg(trailer=b"synthetic appended motion-photo data")
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        "probe_support.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    returned = source_jpeg(str(path), sha, tmp_path)
    assert returned == data
    assert hashlib.sha256(returned).hexdigest() == sha
    assert returned.endswith(b"synthetic appended motion-photo data")
    with Image.open(io.BytesIO(returned)) as decoded:
        assert decoded.format == "JPEG"
        assert decoded.getexif()[0x010F] == "Synthetic Camera"
    with pytest.raises(ValueError, match="requires a path"):
        source_jpeg(str(path), None, tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        source_jpeg(str(path), "0" * 64, tmp_path)
    monkeypatch.setattr(
        "probe_support.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    with pytest.raises(ValueError, match="Git-ignored"):
        source_jpeg(str(path), sha, tmp_path)


@pytest.mark.parametrize(
    "data",
    (
        b"\xff\xd8synthetic\xff\xd9",
        _synthetic_jpeg()[:-40],
        _synthetic_png(),
    ),
    ids=("marker-only-junk", "truncated-jpeg", "non-jpeg"),
)
def test_private_source_rejects_bytes_that_do_not_decode_as_jpeg(tmp_path, monkeypatch, data):
    path = tmp_path / "private.jpg"
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        "probe_support.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    with pytest.raises(ValueError, match="decodable bounded JPEG"):
        source_jpeg(str(path), sha, tmp_path)


def test_private_source_still_rejects_path_outside_worktree(tmp_path, monkeypatch):
    root = tmp_path / "worktree"
    outside = tmp_path / "outside.jpg"
    root.mkdir()
    data = _synthetic_jpeg()
    outside.write_bytes(data)
    monkeypatch.setattr(
        "probe_support.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    with pytest.raises(ValueError, match="inside the worktree"):
        source_jpeg(str(outside), hashlib.sha256(data).hexdigest(), root)


def test_each_model_run_has_distinct_honcho_workspace_and_session():
    first = unique_honcho_scope()
    second = unique_honcho_scope()
    assert first[0] != second[0] and first[1] != second[1]


def test_source_worktree_requires_full_sha_and_offline_allows_candidate_diffs(tmp_path):
    root = tmp_path / "synthetic-source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    source = root / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Camera Probe",
            "-c",
            "user.email=camera-probe@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert verify_source_worktree(root, head, require_clean=True) == (head, "")
    with pytest.raises(ValueError, match="lowercase full 40-character"):
        verify_source_worktree(root, head[:12], require_clean=False)
    with pytest.raises(ValueError, match="revision changed"):
        verify_source_worktree(root, "0" * 40, require_clean=False)

    source.write_text("value = 2\n", encoding="utf-8")
    candidate_head, dirty = verify_source_worktree(root, head, require_clean=False)
    assert (candidate_head, dirty) == (head, "M source.py")
    with pytest.raises(ValueError, match="tracked or non-ignored untracked"):
        verify_source_worktree(root, head, require_clean=True)


def test_source_acceptance_rejects_dirty_untracked_and_content_drift(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Camera Probe"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "camera-probe@example.invalid"],
        check=True,
    )
    source = root / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    pinned = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert verify_source_worktree(root, pinned, require_clean=True) == (pinned, "")

    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked or non-ignored untracked"):
        verify_source_worktree(root, pinned, require_clean=True)
    source.write_text("value = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked or non-ignored untracked"):
        verify_source_worktree(root, pinned, require_clean=True)
    source.write_text("value = 1\n", encoding="utf-8")
    (root / "runtime.py").write_text("source = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tracked or non-ignored untracked"):
        verify_source_worktree(root, pinned, require_clean=True)
    (root / "runtime.py").unlink()
    with pytest.raises(ValueError, match="revision changed"):
        verify_source_worktree(root, "0" * 40, require_clean=False)


def test_source_acceptance_accepts_a_different_full_pinned_commit(tmp_path, monkeypatch):
    import probe_support

    root = tmp_path / "source"
    root.mkdir()
    candidate = "f" * 40
    commands = iter((candidate + "\n", ""))

    def fake_git(*_args, **_kwargs):
        from types import SimpleNamespace

        return SimpleNamespace(stdout=next(commands))

    monkeypatch.setattr(probe_support.subprocess, "run", fake_git)
    assert verify_source_worktree(root, candidate, require_clean=True) == (candidate, "")


def test_source_acceptance_allows_ignored_runtime_output(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "Camera Probe"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "camera-probe@example.invalid"],
        check=True,
    )
    (root / ".gitignore").write_text("tmp/\n", encoding="utf-8")
    (root / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", ".gitignore", "source.py"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    pinned = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    output = root / "tmp" / "run.db"
    output.parent.mkdir()
    output.write_text("synthetic", encoding="ascii")
    assert verify_source_worktree(root, pinned, require_clean=True) == (pinned, "")




def _ordinary_camera_event(event_id="event-1", source_id="owner-turn-1", candidate="candidate-1"):
    return SimpleNamespace(
        id=event_id,
        metadata={
            "role": "assistant",
            "nutrition_annotation_status": "recorded",
            "source_message_id": source_id,
            "source_principal": "telegram:123",
            "tenant_id": "synthetic_owner",
            "ingest_source": "dropbox_camera",
            "gateway_session_id": "current-session",
            "source_image_attachment_count": 0,
            "photo_occurrence_source": {
                "source_origin": "dropbox_camera",
                "origin_principal": "telegram:__camera__",
                "camera_candidate_id": candidate,
                "native_photo_message_id": "77",
            },
            "decision_trace": {"annotations": {"nutrition": {
                "meal_at": "2026-10-01T12:00:00+00:00",
            }}},
        },
    )


def test_finalizer_selection_uses_unique_ordinary_owned_source_receipt():
    event = _ordinary_camera_event()
    fixture = SimpleNamespace(id="fixture", metadata={"role": "assistant"})
    assert select_finalizer_event(
        [fixture, event], "candidate-1", "owner-turn-1", 77,
        expected_event_id="event-1",
    ) is event
    for changed in (
        {"source_message_id": "other-owner-turn"},
        {"source_principal": "telegram:foreign"},
        {"tenant_id": "foreign-tenant"},
        {"ingest_source": "telegram"},
        {"camera_route": "context"},
    ):
        bad = SimpleNamespace(id=event.id, metadata={**event.metadata, **changed})
        with pytest.raises(AssertionError, match="finalizer meal event"):
            select_finalizer_event([bad], "candidate-1", "owner-turn-1", 77)


def test_finalizer_selection_rejects_ambiguous_or_forged_photo_occurrence():
    event = _ordinary_camera_event()
    duplicate = _ordinary_camera_event(event_id="event-2")
    with pytest.raises(AssertionError, match="exactly one"):
        select_finalizer_event(
            [event, duplicate], "candidate-1", "owner-turn-1", 77,
        )
    wrong_source = _ordinary_camera_event()
    wrong_source.metadata["photo_occurrence_source"] = {
        **wrong_source.metadata["photo_occurrence_source"],
        "native_photo_message_id": "999",
    }
    with pytest.raises(AssertionError, match="exactly one"):
        select_finalizer_event(
            [wrong_source], "candidate-1", "owner-turn-1", 77,
        )

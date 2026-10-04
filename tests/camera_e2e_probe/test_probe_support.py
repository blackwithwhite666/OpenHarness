"""Offline guards for the optional private finalizer path."""

from __future__ import annotations

import asyncio
import hashlib
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_ohmo"))

from test_camera_ingress import _admit, _candidate, _ingress  # noqa: E402
from openharness.channels.bus.events import InboundMessage  # noqa: E402

from probe_support import (  # noqa: E402
    create_storage_run_dir,
    require_bound_answer,
    select_finalizer_event,
    source_jpeg,
    unique_honcho_scope,
    verify_source_worktree,
)


def test_storage_run_is_persistent_restricted_and_ignored():
    root = Path(__file__).resolve().parents[2]
    parent = root / "tmp" / "camera-native-docker" / "storage-runs"
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


def test_private_source_requires_explicit_hash_and_ignored_jpeg(tmp_path, monkeypatch):
    path = tmp_path / "private.jpg"
    data = b"\xff\xd8synthetic\xff\xd9"
    path.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(
        "probe_support.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    assert source_jpeg(str(path), sha, tmp_path) == data
    with pytest.raises(ValueError, match="requires a path"):
        source_jpeg(str(path), None, tmp_path)
    with pytest.raises(ValueError, match="mismatch"):
        source_jpeg(str(path), "0" * 64, tmp_path)
    monkeypatch.setattr(
        "probe_support.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=1)
    )
    with pytest.raises(ValueError, match="Git-ignored"):
        source_jpeg(str(path), sha, tmp_path)


def test_each_model_run_has_distinct_honcho_workspace_and_session():
    first = unique_honcho_scope()
    second = unique_honcho_scope()
    assert first[0] != second[0] and first[1] != second[1]


def test_source_worktree_requires_full_sha_and_offline_allows_candidate_diffs():
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert verify_source_worktree(root, head, require_clean=False)[0] == head
    with pytest.raises(ValueError, match="tracked or non-ignored untracked"):
        verify_source_worktree(root, head, require_clean=True)
    with pytest.raises(ValueError, match="lowercase full 40-character"):
        verify_source_worktree(root, head[:12], require_clean=False)
    with pytest.raises(ValueError, match="revision changed"):
        verify_source_worktree(root, "0" * 40, require_clean=False)


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


@pytest.mark.asyncio
async def test_real_inbound_binding_and_no_fixture_authored_event(tmp_path):
    ingress, root, bus, _ = _ingress(tmp_path)
    request = _candidate(root)
    try:
        assert (await _admit(ingress, root, "Bearer " + "s" * 40, request))[0] == 202
        await asyncio.wait_for(bus.consume_inbound(), timeout=1)
        answer = InboundMessage(
            channel="telegram",
            sender_id="123",
            chat_id="123",
            content="Я съела 4 сливы",
            metadata={"message_id": 78, "reply_to_message_id": 77},
        )
        ingress.process_real_inbound(answer)
        assert require_bound_answer(answer, request["candidate_id"])
        assert answer.session_key == "telegram:123"
        fixture = SimpleNamespace(
            id="fixture", metadata={"role": "assistant", "nutrition_annotation_status": "recorded"}
        )
        with pytest.raises(AssertionError, match="finalizer meal event"):
            select_finalizer_event([fixture], request["candidate_id"], "78", 77)
        real = SimpleNamespace(
            id="server-event",
            metadata={
                "role": "assistant",
                "nutrition_annotation_status": "recorded",
                "camera_candidate_id": request["candidate_id"],
                "camera_answer_bound": "yes",
                "camera_reply_to_native_message_id": "77",
                "source_message_id": "78",
            },
        )
        assert (
            select_finalizer_event([fixture, real], request["candidate_id"], "78", 77).id
            == "server-event"
        )
        with pytest.raises(AssertionError, match="finalizer meal event"):
            select_finalizer_event([real], request["candidate_id"], "79", 77)
    finally:
        await ingress.close()

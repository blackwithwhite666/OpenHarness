"""Small offline-checkable guards for the optional private Camera probe."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from ohmo.gateway.camera import CAMERA_AUTHORITY

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def create_storage_run_dir(root: Path) -> Path:
    """Keep each projection under the ignored worktree root after the run ends."""
    parent = root / "tmp" / "camera-e2e" / "storage-runs"
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if parent.resolve() != root.resolve() / "tmp" / "camera-e2e" / "storage-runs":
        raise ValueError("storage root escapes the worktree task directory")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(parent)],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("storage root must be Git-ignored")
    os.chmod(parent, 0o700)
    return Path(tempfile.mkdtemp(prefix="finalizer-", dir=parent))


def source_jpeg(path_text: str | None, expected_sha: str | None, root: Path) -> bytes | None:
    if path_text is None and expected_sha is None:
        return None
    if not path_text or not expected_sha or not _SHA256.fullmatch(expected_sha):
        raise ValueError("private source requires a path and lowercase SHA-256")
    try:
        path = Path(path_text).resolve(strict=True)
    except OSError:
        raise ValueError("private source path is unavailable") from None
    if not path.is_file() or not path.is_relative_to(root.resolve()):
        raise ValueError("private source must be a file inside the worktree")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(path)],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ignored.returncode != 0:
        raise ValueError("private source must be Git-ignored")
    try:
        data = path.read_bytes()
    except OSError:
        raise ValueError("private source cannot be read") from None
    if not (
        4 <= len(data) <= 10 * 1024 * 1024 and data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9"
    ):
        raise ValueError("private source must be a bounded JPEG")
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("private source SHA-256 mismatch")
    return data


def unique_honcho_scope() -> tuple[str, str]:
    run_id = uuid4().hex
    return f"camera-joined-{run_id}", f"camera-joined-session-{run_id}"


def require_bound_answer(message, candidate_id: str) -> str:
    metadata = message.metadata
    turn_id = metadata.get("_camera_turn_id")
    if not (
        metadata.get("_camera_authority") is CAMERA_AUTHORITY
        and metadata.get("_camera_answer") == "yes"
        and metadata.get("_camera_candidate_id") == candidate_id
        and metadata.get("reply_to_message_id") == 77
        and isinstance(turn_id, str)
        and turn_id
        and len(message.media) == 1
    ):
        raise AssertionError("Camera owner reply was not bound to the native photo")
    return turn_id


def select_finalizer_event(messages, candidate_id: str, answer_message_id: str):
    """Reject fixture events and unrelated assistant turns before sync."""
    matches = [
        item
        for item in messages
        if item.metadata.get("role") == "assistant"
        and item.metadata.get("camera_candidate_id") == candidate_id
        and item.metadata.get("camera_answer_bound") == "yes"
        and item.metadata.get("camera_reply_to_native_message_id") == "77"
        and item.metadata.get("source_message_id") == answer_message_id
        and item.metadata.get("nutrition_annotation_status") == "recorded"
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one validated finalizer meal event")
    return matches[0]

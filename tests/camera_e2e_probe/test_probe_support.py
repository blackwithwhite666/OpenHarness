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
)


def test_storage_run_is_persistent_restricted_and_ignored():
    root = Path(__file__).resolve().parents[2]
    parent = root / "tmp" / "camera-e2e" / "storage-runs"
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
            select_finalizer_event([fixture], request["candidate_id"], "78")
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
            select_finalizer_event([fixture, real], request["candidate_id"], "78").id
            == "server-event"
        )
        with pytest.raises(AssertionError, match="finalizer meal event"):
            select_finalizer_event([real], request["candidate_id"], "79")
    finally:
        await ingress.close()

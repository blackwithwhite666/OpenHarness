"""Per-chat work dir: every unbound chat runs in its own ~/.ohmo/work/<token>/
cwd, so transient output (diagrams, downloads, scratch) is isolated per chat and
reaped on /new — instead of all chats sharing the workspace root.
"""

import os
import time
from pathlib import Path

from openharness.channels.bus.events import InboundMessage

from ohmo.gateway.runtime import OhmoSessionRuntimePool
from ohmo.session_storage import (
    clear_session_work_dir,
    get_session_work_dir,
    reap_stale_work_dirs,
)
from ohmo.workspace import get_work_dir, initialize_workspace


# --------------------------- helpers ---------------------------


def test_session_work_dir_created_and_isolated(tmp_path):
    a = get_session_work_dir("telegram:111", workspace=tmp_path)
    b = get_session_work_dir("telegram:222", workspace=tmp_path)
    assert a.is_dir() and b.is_dir()
    assert a != b  # distinct per chat
    assert a.parent == get_work_dir(tmp_path)
    assert get_session_work_dir("telegram:111", workspace=tmp_path) == a  # idempotent


def test_clear_session_work_dir(tmp_path):
    d = get_session_work_dir("telegram:c", workspace=tmp_path)
    (d / "diagram.html").write_text("x")
    clear_session_work_dir("telegram:c", workspace=tmp_path)
    assert not d.exists()
    d2 = get_session_work_dir("telegram:c", workspace=tmp_path)  # recreated lazily, empty
    assert d2.is_dir() and not any(d2.iterdir())


def test_reap_stale_work_dirs(tmp_path):
    fresh = get_session_work_dir("telegram:fresh", workspace=tmp_path)
    stale = get_session_work_dir("telegram:stale", workspace=tmp_path)
    old = time.time() - 10 * 86400
    os.utime(stale, (old, old))
    reaped = reap_stale_work_dirs(workspace=tmp_path, max_age_s=7 * 86400)
    assert reaped == 1
    assert fresh.is_dir() and not stale.exists()


def test_reap_missing_root_is_zero(tmp_path):
    assert reap_stale_work_dirs(workspace=tmp_path / "nope") == 0


# --------------------------- routing ---------------------------


def test_unbound_chat_cwd_is_per_session_work_dir(tmp_path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    pool = OhmoSessionRuntimePool(cwd=tmp_path, workspace=workspace, provider_profile="codex")

    msg_a = InboundMessage(channel="telegram", sender_id="u1", chat_id="111", content="hi", metadata={})
    msg_b = InboundMessage(channel="telegram", sender_id="u2", chat_id="222", content="hi", metadata={})
    cwd_a = pool._cwd_for_message(msg_a, "telegram:111")
    cwd_b = pool._cwd_for_message(msg_b, "telegram:222")

    work_root = str(get_work_dir(pool._workspace))
    assert cwd_a.startswith(work_root) and cwd_b.startswith(work_root)  # under ~/.ohmo/work
    assert cwd_a != cwd_b  # isolated per chat
    assert Path(cwd_a).is_dir()
    assert cwd_a == str(get_session_work_dir("telegram:111", pool._workspace))
    assert cwd_a != pool._cwd  # not the shared workspace root anymore

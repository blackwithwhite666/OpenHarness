"""Offline tests for removing the retired Dropbox Camera config before deploy."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "ci" / "migrate_camera_gateway_config.py"


def _run_migration(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_migration_removes_only_the_retired_camera_root_and_preserves_mode(tmp_path: Path):
    config_path = tmp_path / "gateway.json"
    raw = {
        "provider_profile": "codex",
        "camera_ingress": {
            "enabled": True,
            "synchronized_root": "/home/user/Dropbox/nutrition-assets",
            "listen_port": 18751,
        },
    }
    config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    config_path.chmod(0o600)

    result = _run_migration(config_path)

    assert result.returncode == 0, result.stderr
    assert "removed retired Dropbox Camera config" in result.stdout
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "provider_profile": "codex",
        "camera_ingress": {"enabled": True, "listen_port": 18751},
    }
    assert stat.S_IMODE(config_path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_migration_is_idempotent_and_leaves_current_config_bytes_unchanged(tmp_path: Path):
    config_path = tmp_path / "gateway.json"
    raw = b'{"camera_ingress":{"enabled":true,"listen_port":18751}}\n'
    config_path.write_bytes(raw)

    result = _run_migration(config_path)

    assert result.returncode == 0, result.stderr
    assert "needs no migration" in result.stdout
    assert config_path.read_bytes() == raw


def test_migration_refuses_malformed_json_without_rewriting_it(tmp_path: Path):
    config_path = tmp_path / "gateway.json"
    raw = b'{"camera_ingress":'
    config_path.write_bytes(raw)

    result = _run_migration(config_path)

    assert result.returncode != 0
    assert config_path.read_bytes() == raw


def test_migration_refuses_symlink_without_touching_target(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text('{"camera_ingress":{"synchronized_root":"/private"}}\n')
    config_path = tmp_path / "gateway.json"
    config_path.symlink_to(target)

    result = _run_migration(config_path)

    assert result.returncode != 0
    assert target.read_text() == '{"camera_ingress":{"synchronized_root":"/private"}}\n'

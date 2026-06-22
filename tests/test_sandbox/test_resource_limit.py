"""Tests for per-task sandbox resource limits."""

from __future__ import annotations

import pytest

from openharness.config.settings import SandboxResourceSettings, SandboxSettings, Settings
from openharness.sandbox.adapter import SandboxUnavailableError
from openharness.sandbox.resource_limit import kill_resource_scope, wrap_command_with_resource_limit


def test_wrap_resource_limit_returns_original_when_disabled():
    argv = ["bash", "-lc", "echo hi"]
    settings = Settings(
        sandbox=SandboxSettings(resources=SandboxResourceSettings(enabled=False))
    )

    wrapped, unit = wrap_command_with_resource_limit(argv, settings=settings)

    assert wrapped == argv
    assert unit is None


def test_wrap_resource_limit_returns_original_when_backend_off():
    argv = ["bash", "-lc", "echo hi"]
    settings = Settings(
        sandbox=SandboxSettings(resources=SandboxResourceSettings(enabled=True, backend="off"))
    )

    wrapped, unit = wrap_command_with_resource_limit(argv, settings=settings)

    assert wrapped == argv
    assert unit is None


def test_wrap_resource_limit_prefixes_when_available(monkeypatch):
    argv = ["bash", "-lc", "echo hi"]
    settings = Settings(
        sandbox=SandboxSettings(resources=SandboxResourceSettings(enabled=True))
    )
    monkeypatch.setattr("openharness.sandbox.resource_limit.get_platform", lambda: "linux")
    monkeypatch.setattr(
        "openharness.sandbox.resource_limit.shutil.which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None,
    )

    wrapped, unit = wrap_command_with_resource_limit(argv, settings=settings)

    assert wrapped[:3] == ["systemd-run", "--user", "--scope"]
    assert isinstance(unit, str)
    assert unit.endswith(".scope")
    assert f"--unit={unit}" in wrapped
    assert ["-p", "MemoryMax=70%"] == wrapped[6:8]
    assert "-p" in wrapped
    assert "MemorySwapMax=2G" in wrapped
    separator_idx = wrapped.index("--")
    assert wrapped[separator_idx + 1 :] == argv


def test_wrap_resource_limit_raises_when_required_and_unavailable(monkeypatch):
    settings = Settings(
        sandbox=SandboxSettings(
            resources=SandboxResourceSettings(enabled=True, fail_if_unavailable=True)
        )
    )
    monkeypatch.setattr("openharness.sandbox.resource_limit.get_platform", lambda: "linux")
    monkeypatch.setattr("openharness.sandbox.resource_limit.shutil.which", lambda name: None)

    with pytest.raises(SandboxUnavailableError):
        wrap_command_with_resource_limit(["bash", "-lc", "echo hi"], settings=settings)


def test_wrap_resource_limit_returns_original_when_unavailable_and_not_required(monkeypatch):
    argv = ["bash", "-lc", "echo hi"]
    settings = Settings(
        sandbox=SandboxSettings(
            resources=SandboxResourceSettings(enabled=True, fail_if_unavailable=False)
        )
    )
    monkeypatch.setattr("openharness.sandbox.resource_limit.get_platform", lambda: "linux")
    monkeypatch.setattr("openharness.sandbox.resource_limit.shutil.which", lambda name: None)

    wrapped, unit = wrap_command_with_resource_limit(argv, settings=settings)

    assert wrapped == argv
    assert unit is None


def test_wrap_resource_limit_returns_original_when_no_cap_configured(monkeypatch):
    argv = ["bash", "-lc", "echo hi"]
    settings = Settings(
        sandbox=SandboxSettings(
            resources=SandboxResourceSettings(
                enabled=True,
                memory_max="",
                memory_high="",
                pids_max=0,
            )
        )
    )
    monkeypatch.setattr("openharness.sandbox.resource_limit.get_platform", lambda: "linux")
    monkeypatch.setattr(
        "openharness.sandbox.resource_limit.shutil.which",
        lambda name: "/usr/bin/systemd-run" if name == "systemd-run" else None,
    )

    wrapped, unit = wrap_command_with_resource_limit(argv, settings=settings)

    assert wrapped == argv
    assert unit is None


def test_settings_resource_limit_round_trips_from_dict():
    settings = Settings(
        sandbox={
            "resources": {
                "enabled": True,
                "memory_max": "3G",
                "memory_high": "2G",
                "memory_swap_max": "0",
                "pids_max": 128,
                "cpu_weight": 50,
                "backend": "systemd-run",
                "fail_if_unavailable": True,
            }
        }
    )

    dumped = settings.model_dump()
    loaded = Settings.model_validate(dumped)

    assert loaded.sandbox.resources.enabled is True
    assert loaded.sandbox.resources.memory_max == "3G"
    assert loaded.sandbox.resources.memory_high == "2G"
    assert loaded.sandbox.resources.memory_swap_max == "0"
    assert loaded.sandbox.resources.pids_max == 128
    assert loaded.sandbox.resources.cpu_weight == 50
    assert loaded.sandbox.resources.backend == "systemd-run"
    assert loaded.sandbox.resources.fail_if_unavailable is True


def test_kill_resource_scope_calls_systemctl(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs

    monkeypatch.setattr("openharness.sandbox.resource_limit.subprocess.run", fake_run)

    kill_resource_scope("oh-task-test.scope")

    assert captured["argv"] == [
        "systemctl",
        "--user",
        "kill",
        "--signal=SIGKILL",
        "oh-task-test.scope",
    ]
    assert captured["kwargs"] == {
        "check": False,
        "capture_output": True,
        "timeout": 5,
    }


def test_kill_resource_scope_swallows_errors(monkeypatch):
    def fake_run(*args, **kwargs):
        raise RuntimeError("systemctl failed")

    monkeypatch.setattr("openharness.sandbox.resource_limit.subprocess.run", fake_run)

    kill_resource_scope("oh-task-test.scope")

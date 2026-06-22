"""Per-task resource limit wrapper using systemd-run scopes."""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid

from openharness.config import Settings, load_settings
from openharness.config.settings import SandboxResourceSettings
from openharness.platforms import get_platform
from openharness.sandbox.adapter import SandboxUnavailableError


logger = logging.getLogger(__name__)


def make_scope_unit_name() -> str:
    """Return a transient systemd scope unit name for one task."""
    return f"oh-task-{uuid.uuid4().hex[:12]}.scope"


def resource_limit_availability(settings: Settings) -> tuple[bool, str | None]:
    """Return whether per-task resource limits can be applied."""
    backend = settings.sandbox.resources.backend
    if backend == "off":
        return False, "disabled"

    if get_platform() not in {"linux", "wsl"}:
        return False, "resource limits require Linux cgroup v2"

    if backend in {"auto", "systemd-run"}:
        if shutil.which("systemd-run") is None:
            return False, "systemd-run not found"
        return True, None

    return False, f"unknown resource backend {backend!r}"


def build_resource_limit_argv(
    rs: SandboxResourceSettings,
    *,
    unit: str | None = None,
) -> list[str]:
    """Build the systemd-run argv prefix for resource-limited commands."""
    argv = ["systemd-run", "--user", "--scope", "-q", "--expand-environment=no"]
    if unit is not None:
        argv.insert(3, f"--unit={unit}")
    if rs.memory_max:
        argv.extend(["-p", f"MemoryMax={rs.memory_max}"])
    if rs.memory_high:
        argv.extend(["-p", f"MemoryHigh={rs.memory_high}"])
    if rs.memory_swap_max:
        argv.extend(["-p", f"MemorySwapMax={rs.memory_swap_max}"])
    if rs.pids_max:
        argv.extend(["-p", f"TasksMax={rs.pids_max}"])
    if rs.cpu_weight:
        argv.extend(["-p", f"CPUWeight={rs.cpu_weight}"])
    argv.append("--")
    return argv


def wrap_command_with_resource_limit(
    argv: list[str],
    *,
    settings: Settings | None = None,
) -> tuple[list[str], str | None]:
    """Wrap an argv list with systemd-run when per-task resource limits are enabled."""
    resolved_settings = settings or load_settings()
    rs = resolved_settings.sandbox.resources
    if not rs.enabled or rs.backend == "off":
        return argv, None

    available, reason = resource_limit_availability(resolved_settings)
    if not available:
        if rs.fail_if_unavailable:
            raise SandboxUnavailableError(reason or "resource limits unavailable")
        logger.warning(
            "resource limit requested but unavailable: %s; running task UNBOUNDED",
            reason,
        )
        return argv, None

    if not (rs.memory_max or rs.memory_high or rs.pids_max):
        logger.warning("resource limit enabled but no cap configured; running task UNBOUNDED")
        return argv, None

    unit = make_scope_unit_name()
    return build_resource_limit_argv(rs, unit=unit) + argv, unit


def kill_resource_scope(unit: str) -> None:
    """Best-effort cleanup for a transient systemd scope."""
    try:
        subprocess.run(
            ["systemctl", "--user", "kill", "--signal=SIGKILL", unit],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except Exception as exc:
        logger.debug("failed to kill resource scope %s: %s", unit, exc)

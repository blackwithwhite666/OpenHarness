"""Offline contract tests for the deploy gateway stop-and-wait helper."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "ci" / "restart_gateway.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_restart_helper(
    tmp_path: Path,
    *,
    live: bool = False,
    pid_file: bool = False,
    systemd_stops: bool = False,
    stop_detached: bool = False,
    stop_requires_poll: bool = False,
    identity: str = "gateway",
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    workspace = tmp_path / ".ohmo"
    workspace.mkdir()
    pid = os.getpid()
    if pid_file:
        (workspace / "gateway.pid").write_text(f"{pid}\n", encoding="utf-8")
    state = tmp_path / "gateway-live"
    state.write_text("1" if live else "0", encoding="utf-8")
    log = tmp_path / "calls.log"

    _write_executable(
        bin_dir / "systemctl",
        """#!/usr/bin/env bash
set -eu
printf 'systemctl %s\\n' "$*" >> "$OHMO_FAKE_LOG"
if [[ "$1 $2" == "--user stop" && "$OHMO_FAKE_SYSTEMD_STOPS" == "1" ]]; then
  printf '0' > "$OHMO_FAKE_STATE"
  rm -f "$OHMO_FAKE_PID_FILE"
fi
""",
    )
    _write_executable(
        bin_dir / "ohmo",
        """#!/usr/bin/env bash
set -eu
printf 'ohmo %s\\n' "$*" >> "$OHMO_FAKE_LOG"
if [[ "$1 $2" == "gateway stop" && "$OHMO_FAKE_STOP_DETACHED" == "1" ]]; then
  if [[ "$OHMO_FAKE_STOP_REQUIRES_POLL" == "1" ]]; then
    printf '2' > "$OHMO_FAKE_STATE"
  else
    printf '0' > "$OHMO_FAKE_STATE"
  fi
fi
""",
    )
    _write_executable(
        bin_dir / "ps",
        """#!/usr/bin/env bash
set -eu
args="python -m ohmo gateway run --cwd /synthetic --workspace $OHMO_GATEWAY_WORKSPACE --no-console-log"
if [[ "$OHMO_FAKE_IDENTITY" == "console" ]]; then
  args="/synthetic/.local/bin/ohmo gateway run --cwd /synthetic"
elif [[ "$OHMO_FAKE_IDENTITY" == "unknown" ]]; then
  args="python -m unrelated-worker"
fi
state="$(cat "$OHMO_FAKE_STATE")"
if [[ "$state" == "0" ]]; then
  exit 0
fi
if [[ "$1" == "-p" ]]; then
  printf '%s\\n' "$args"
elif [[ "$1" == "-eo" ]]; then
  printf '%s %s\\n' "$OHMO_FAKE_PID" "$args"
  if [[ "$state" == "2" ]]; then
    printf '0' > "$OHMO_FAKE_STATE"
  fi
fi
""",
    )
    _write_executable(bin_dir / "journalctl", "#!/usr/bin/env bash\nset -eu\n")

    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "OHMO_FAKE_LOG": str(log),
        "OHMO_FAKE_STATE": str(state),
        "OHMO_FAKE_PID": str(pid),
        "OHMO_FAKE_PID_FILE": str(workspace / "gateway.pid"),
        "OHMO_FAKE_SYSTEMD_STOPS": "1" if systemd_stops else "0",
        "OHMO_FAKE_STOP_DETACHED": "1" if stop_detached else "0",
        "OHMO_FAKE_STOP_REQUIRES_POLL": "1" if stop_requires_poll else "0",
        "OHMO_FAKE_IDENTITY": identity,
        "OHMO_GATEWAY_WORKSPACE": str(workspace),
        "OHMO_GATEWAY_STOP_MAX_WAIT": "1" if stop_requires_poll else "0",
        "OHMO_GATEWAY_STOP_INTERVAL": "0",
        "OHMO_GATEWAY_START_DELAY": "0",
        "HOME": str(tmp_path / "home"),
        "XDG_RUNTIME_DIR": str(tmp_path / "runtime"),
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/synthetic-bus",
    }
    result = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    result.calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
    return result


def test_restart_helper_stops_managed_gateway_before_starting_unit(tmp_path: Path):
    result = _run_restart_helper(
        tmp_path,
        live=True,
        pid_file=True,
        systemd_stops=True,
        identity="console",
    )

    assert result.returncode == 0, result.stderr
    assert result.calls == [
        "systemctl --user stop ohmo-gateway.service",
        f"ohmo gateway stop --workspace {tmp_path / '.ohmo'}",
        "systemctl --user start ohmo-gateway.service",
        "systemctl --user is-active ohmo-gateway.service",
    ]


def test_restart_helper_starts_cleanly_when_no_gateway_process_exists(tmp_path: Path):
    result = _run_restart_helper(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.calls[0] == "systemctl --user stop ohmo-gateway.service"
    assert result.calls[-2:] == [
        "systemctl --user start ohmo-gateway.service",
        "systemctl --user is-active ohmo-gateway.service",
    ]


def test_restart_helper_stops_and_waits_for_detached_workspace_gateway(tmp_path: Path):
    result = _run_restart_helper(tmp_path, live=True, pid_file=True, stop_detached=True)

    assert result.returncode == 0, result.stderr
    assert result.calls.index(f"ohmo gateway stop --workspace {tmp_path / '.ohmo'}") < result.calls.index(
        "systemctl --user start ohmo-gateway.service"
    )


def test_restart_helper_polls_after_detached_stop_before_starting_unit(tmp_path: Path):
    result = _run_restart_helper(
        tmp_path,
        live=True,
        pid_file=True,
        stop_detached=True,
        stop_requires_poll=True,
    )

    assert result.returncode == 0, result.stderr
    assert "waiting for detached workspace gateway" in result.stdout
    assert result.calls.index("systemctl --user start ohmo-gateway.service") > result.calls.index(
        f"ohmo gateway stop --workspace {tmp_path / '.ohmo'}"
    )


def test_restart_helper_refuses_to_start_when_detached_gateway_survives(tmp_path: Path):
    result = _run_restart_helper(tmp_path, live=True, pid_file=True)

    assert result.returncode == 1
    assert "workspace gateway still running" in result.stderr
    assert "systemctl --user start ohmo-gateway.service" not in result.calls


def test_restart_helper_refuses_live_pid_file_with_unknown_identity(tmp_path: Path):
    result = _run_restart_helper(tmp_path, live=True, pid_file=True, identity="unknown")

    assert result.returncode == 1
    assert "refusing unverified live PID" in result.stderr
    assert result.calls == ["systemctl --user stop ohmo-gateway.service"]

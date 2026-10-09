import os
import subprocess
from pathlib import Path


def test_wait_for_idle_counts_runtime_start_and_complete_markers(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "journalctl-count"
    journalctl = bin_dir / "journalctl"
    journalctl.write_text(
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "counter = Path(os.environ['JOURNALCTL_COUNT'])\n"
        "calls = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(calls + 1))\n"
        "if calls < 2:\n"
        "    print('ohmo runtime processing start channel=telegram session_id=s1')\n"
        "    print('ohmo runtime processing start channel=telegram session_id=s2')\n"
        "    if calls == 1:\n"
        "        print('ohmo runtime processing complete channel=telegram session_id=s1')\n"
        "else:\n"
        "    print('ohmo runtime processing start channel=telegram session_id=s1')\n"
        "    print('ohmo runtime processing start channel=telegram session_id=s2')\n"
        "    print('ohmo runtime processing complete channel=telegram session_id=s1')\n"
        "    print('ohmo runtime processing complete channel=telegram session_id=s2')\n",
        encoding="utf-8",
    )
    journalctl.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    sleep.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "JOURNALCTL_COUNT": str(counter),
            "XDG_RUNTIME_DIR": str(tmp_path),
            "OHMO_IDLE_INTERVAL": "1",
            "OHMO_IDLE_MAX_WAIT": "2",
        }
    )
    script = Path(__file__).parents[1] / "ci" / "wait_for_idle.sh"
    result = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "a user turn is in-flight" in result.stdout
    assert "gateway idle — safe to restart" in result.stdout
    assert counter.read_text(encoding="utf-8") == "4"

"""Bind an approved native settings root before subscription resolution."""

from __future__ import annotations

import os
from pathlib import Path

from probe_support import native_person_source_clients


def resolve_native_report_clients(config_dir: Path):
    """Use the selected settings and external-auth store for both clients."""
    from openharness.config.settings import load_settings

    selected = config_dir.resolve(strict=True)
    settings_path = selected / "settings.json"
    if not settings_path.is_file():
        raise ValueError("lead must supply existing read-only native subscription settings")
    os.environ["OPENHARNESS_CONFIG_DIR"] = str(selected)
    os.environ["OPENHARNESS_PROFILE"] = "codex"
    settings = load_settings(settings_path)
    return native_person_source_clients(settings, scenario="ordinary balance report only")

"""Isolation for reminder tests: redirect reminders.json into a temp dir.

The store imports ``get_reminders_path`` by name, so the binding in
``ohmo.reminders.store`` must be patched (the workspace source is patched too).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _tmp_reminders_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "reminders.json"
    monkeypatch.setattr(
        "ohmo.workspace.get_reminders_path", lambda workspace=None: target
    )
    monkeypatch.setattr(
        "ohmo.reminders.store.get_reminders_path", lambda workspace=None: target
    )

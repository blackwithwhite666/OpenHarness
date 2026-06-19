"""Store construction helpers for ohmo eval/flywheel data."""

from __future__ import annotations

from pathlib import Path

from openharness.evals import EvalStore
from ohmo.workspace import get_evals_dir


def get_eval_store(workspace: str | Path | None = None) -> EvalStore:
    """Return the generic eval store rooted at ``<ohmo-workspace>/evals``."""
    return EvalStore(get_evals_dir(workspace))

from pathlib import Path

from ohmo.evals import get_eval_store
from ohmo.workspace import get_evals_dir


def test_get_evals_dir_resolves_under_explicit_workspace(tmp_path: Path):
    workspace = tmp_path / "workspace"

    assert get_evals_dir(workspace) == workspace.resolve() / "evals"


def test_get_evals_dir_uses_workspace_environment(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "from-env"
    monkeypatch.setenv("OHMO_WORKSPACE", str(workspace))

    assert get_evals_dir() == workspace.resolve() / "evals"


def test_get_eval_store_uses_ohmo_workspace_evals_dir(tmp_path: Path):
    workspace = tmp_path / "workspace"

    store = get_eval_store(workspace)

    assert store.root == workspace.resolve() / "evals"
    assert (workspace / "evals" / "evals.sqlite").is_file()

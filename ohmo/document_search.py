"""Best-effort integration with the document search index."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_DOCUMENT_SEARCH_CLI = os.environ.get("OHMO_DOCUMENT_SEARCH_CLI", "document_search-cli")


def reindex(path: str | Path, *, collection: str = "memory") -> None:
    """Start a detached index update without delaying or breaking the caller."""
    if os.environ.get("OHMO_MEMORY_AUTOINDEX", "1") != "1":
        return
    try:
        subprocess.Popen(
            [_DOCUMENT_SEARCH_CLI, "index", str(path), "--collection", collection],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        return

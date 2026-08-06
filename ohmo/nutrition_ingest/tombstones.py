"""Atomic, owner-only suppression tombstones for nutrition candidates."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path

from .models import SeenTombstoneV1, validate_candidate_id


class SeenTombstoneStore:
    """Write only validated direct children of ``<root>/_seen``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.directory = self.root / "_seen"

    def path_for(self, candidate_id: str) -> Path:
        validated = validate_candidate_id(candidate_id)
        path = self.directory / f"{validated}.json"
        if path.parent != self.directory or path.name != f"{validated}.json":
            raise ValueError("tombstone path escapes the _seen directory")
        return path

    def replace(self, value: SeenTombstoneV1) -> Path:
        value = SeenTombstoneV1.model_validate(value.model_dump(mode="json"))
        path = self.path_for(value.candidate_id)
        self._ensure_directory()
        payload = json.dumps(
            value.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
        )
        payload += "\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.directory
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600, follow_symlinks=False)
            self._fsync(self.directory)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def _ensure_directory(self) -> None:
        if self.directory.exists():
            mode = self.directory.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError("nutrition _seen path must be a real directory")
            if self.directory.resolve() != self.directory:
                raise ValueError("nutrition _seen directory escapes configured root")
            os.chmod(self.directory, 0o700)
            return
        self.directory.mkdir(mode=0o700)
        os.chmod(self.directory, 0o700)
        self._fsync(self.root)

    @staticmethod
    def _fsync(directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["SeenTombstoneStore"]

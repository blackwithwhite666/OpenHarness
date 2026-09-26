#!/usr/bin/env python3
"""Remove the retired Dropbox path before installing the strict Camera API config."""

from __future__ import annotations

import argparse
import json
import os
import stat
import uuid
from pathlib import Path
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def remove_retired_dropbox_root(config_path: Path) -> bool:
    """Remove only camera_ingress.synchronized_root, preserving all other config."""
    try:
        metadata = config_path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise ValueError("gateway config must be a regular file owned by the deploy user")

    config = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(config, dict):
        raise ValueError("gateway config must contain a JSON object")
    camera = config.get("camera_ingress")
    if camera is None:
        return False
    if not isinstance(camera, dict):
        raise ValueError("camera_ingress config must contain a JSON object")
    if "synchronized_root" not in camera:
        return False

    del camera["synchronized_root"]
    mode = stat.S_IMODE(metadata.st_mode)
    temporary = config_path.with_name(f".{config_path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config_path)
        directory_fd = os.open(config_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".ohmo" / "gateway.json",
    )
    args = parser.parse_args()
    try:
        changed = remove_retired_dropbox_root(args.config)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"cannot migrate gateway config: {exc}")
    print(
        "removed retired Dropbox Camera config" if changed else "gateway config needs no migration"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

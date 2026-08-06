"""Deterministic readiness scanner for synchronized nutrition artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import asyncio
from collections.abc import Awaitable, Callable

from .models import ManifestV1, parse_manifest


@dataclass(frozen=True)
class ReadyNutritionArtifact:
    candidate_id: str
    directory: Path
    manifest_path: Path
    image_path: Path
    manifest: ManifestV1

    @property
    def candidate(self) -> ManifestV1:
        return self.manifest


class NutritionArtifactScanner:
    """Find only complete, self-verifying published candidates under ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def scan_ready(self) -> list[ReadyNutritionArtifact]:
        if not self.root.exists():
            return []
        candidates: list[ReadyNutritionArtifact] = []
        for directory in sorted(self.root.iterdir(), key=lambda path: path.name):
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            if directory.name.startswith(".") or directory.name.endswith(".tmp"):
                continue
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file() or manifest_path.name.startswith("."):
                continue
            try:
                manifest = parse_manifest(manifest_path, self.root)
                image_path = self._image_path(directory, manifest.original_filename)
                self._verify_image(image_path, manifest)
            except (OSError, ValueError, TypeError):
                continue
            candidates.append(
                ReadyNutritionArtifact(
                    candidate_id=manifest.candidate_id,
                    directory=directory,
                    manifest_path=manifest_path,
                    image_path=image_path,
                    manifest=manifest,
                )
            )
        candidates.sort(key=lambda item: (item.manifest.discovery_time, item.candidate_id))
        return candidates

    def _image_path(self, directory: Path, filename: str) -> Path:
        relative = Path(filename)
        if relative.name != filename or relative.is_absolute() or filename in {"", ".", ".."}:
            raise ValueError("manifest image filename escapes candidate directory")
        canonical_image_path = (directory / f"original{relative.suffix}").resolve()
        named_image_path = (directory / relative).resolve()
        image_path = (
            canonical_image_path
            if canonical_image_path.is_file()
            else named_image_path
        )
        if image_path.parent != directory.resolve():
            raise ValueError("manifest image filename escapes candidate directory")
        if not image_path.is_file():
            raise ValueError("candidate image is not synchronized")
        return image_path

    @staticmethod
    def _verify_image(image_path: Path, manifest: ManifestV1) -> None:
        size = image_path.stat().st_size
        if size != manifest.original_size_bytes:
            raise ValueError("candidate image size does not match manifest")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != manifest.original_sha256:
            raise ValueError("candidate image hash does not match manifest")


class NutritionIngestWatcher:
    """Small cancellable polling loop around a coordinator's ``poll_once``."""

    def __init__(
        self,
        poll_once: Callable[[], Awaitable[object] | object],
        *,
        interval_seconds: float = 10.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._poll_once = poll_once
        self._interval = interval_seconds
        self._running = False

    async def run(self) -> None:
        self._running = True
        try:
            while self._running:
                result = self._poll_once()
                if asyncio.iscoroutine(result):
                    await result
                await asyncio.sleep(self._interval)
        except asyncio.CancelledError:
            raise

    def stop(self) -> None:
        self._running = False

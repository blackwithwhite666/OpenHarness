"""Materialize captured read fixtures into a confined live-read workspace."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


def _safe_join(sandbox: Path, abs_or_rel: str) -> Path | None:
    """Return ``abs_or_rel`` mirrored under ``sandbox``, or ``None`` on escape."""
    try:
        root = sandbox.resolve()
        candidate = root / abs_or_rel.lstrip("/")
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def remap_in(path_str: str, sandbox: Path) -> str | None:
    """Map an original path into the sandbox mirror."""
    target = _safe_join(sandbox, path_str)
    return str(target) if target is not None else None


def remap_out(text: str, sandbox: Path) -> str:
    """Map sandbox paths in tool output back to their original absolute paths."""
    return text.replace(str(sandbox.resolve()), "")


def materialize_read_fixtures(fixtures: Iterable[Any], sandbox: Path) -> int:
    """Write captured ``read_file`` outputs into the sandbox mirror.

    Captured ``read_file`` output may already include display formatting such as
    headers or line numbers. The output is written verbatim as a best-effort file
    reconstruction for later live local reads.
    """
    written: set[Path] = set()
    count = 0
    for fixture in fixtures:
        if getattr(fixture, "tool_name", None) != "read_file":
            continue
        try:
            payload = json.loads(getattr(fixture, "input_text", "") or "")
        except json.JSONDecodeError:
            continue
        path = payload.get("path") if isinstance(payload, dict) else None
        if not isinstance(path, str) or not path:
            continue
        target = _safe_join(sandbox, path)
        if target is None or target in written or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(getattr(fixture, "output_text", ""), encoding="utf-8")
        written.add(target)
        count += 1
    return count


class LiveLocalReadTool(BaseTool):
    """Run a real local read tool against a materialized sandbox, with replay fallback."""

    def __init__(
        self,
        *,
        real_tool: BaseTool,
        mock_tool: BaseTool,
        sandbox: Path,
        path_fields: tuple[str, ...] = ("path", "root"),
    ) -> None:
        self.name = real_tool.name
        self.description = real_tool.description
        self.input_model = real_tool.input_model
        self._real_tool = real_tool
        self._mock_tool = mock_tool
        self._sandbox = sandbox.resolve()
        self._path_fields = path_fields

    async def execute(
        self,
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> ToolResult:
        try:
            data = arguments.model_dump()
            for field in self._path_fields:
                value = data.get(field)
                if not isinstance(value, str):
                    continue
                remapped = remap_in(value, self._sandbox)
                if remapped is None:
                    return await self._mock_tool.execute(arguments, context)
                data[field] = remapped
            remapped_args = self._real_tool.input_model(**data)
            result = await self._real_tool.execute(
                remapped_args,
                ToolExecutionContext(
                    cwd=self._sandbox,
                    metadata=context.metadata,
                    hook_executor=context.hook_executor,
                ),
            )
        except Exception:
            return await self._mock_tool.execute(arguments, context)
        return ToolResult(
            output=remap_out(result.output, self._sandbox),
            is_error=result.is_error,
            metadata={**result.metadata, "lane": "live-local", "tool": self.name},
        )

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

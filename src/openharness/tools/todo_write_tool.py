"""Tool for maintaining a project TODO file."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult


class TodoWriteToolInput(BaseModel):
    """Arguments for TODO writes."""

    item: str = Field(default="", description="TODO item text (omit only for clear_completed)")
    checked: bool = Field(default=False, description="Mark the item done")
    remove: bool = Field(
        default=False,
        description="Remove the line matching `item` (whether done or not) instead of adding it",
    )
    clear_completed: bool = Field(
        default=False,
        description="Prune ALL completed [x] items from the list (housekeeping; ignores `item`)",
    )
    path: str = Field(default="TODO.md")


def _todo_lines(text: str) -> list[str]:
    return text.splitlines()


class TodoWriteTool(BaseTool):
    """Add, complete, remove, or prune items in a TODO markdown checklist."""

    name = "todo_write"
    description = (
        "Maintain a markdown TODO checklist. Add a new item, mark one done "
        "(checked=true), remove one (remove=true), or prune every completed item "
        "(clear_completed=true — use this to drop a previous task's finished items "
        "so the list only reflects the current request)."
    )
    input_model = TodoWriteToolInput

    async def execute(self, arguments: TodoWriteToolInput, context: ToolExecutionContext) -> ToolResult:
        path = Path(context.cwd) / arguments.path
        existing = path.read_text(encoding="utf-8") if path.exists() else "# TODO\n"

        # Housekeeping: drop every completed item (e.g. leftovers from a prior task).
        if arguments.clear_completed:
            kept = [ln for ln in _todo_lines(existing) if not ln.lstrip().startswith("- [x]")]
            removed = len(_todo_lines(existing)) - len(kept)
            path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
            return ToolResult(output=f"Pruned {removed} completed item(s) in {path}")

        if not arguments.item.strip():
            return ToolResult(
                output="item is required (unless clear_completed=true)", is_error=True
            )

        unchecked_line = f"- [ ] {arguments.item}"
        checked_line = f"- [x] {arguments.item}"

        # Remove the item entirely (in either state).
        if arguments.remove:
            kept = [
                ln for ln in _todo_lines(existing)
                if ln.strip() not in (unchecked_line, checked_line)
            ]
            if len(kept) == len(_todo_lines(existing)):
                return ToolResult(output=f"Item not found in {path}")
            path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
            return ToolResult(output=f"Removed item from {path}")

        target_line = checked_line if arguments.checked else unchecked_line

        if unchecked_line in existing and arguments.checked:
            # Mark existing unchecked item as done (in-place update)
            updated = existing.replace(unchecked_line, checked_line, 1)
        elif target_line in existing:
            # Item already in desired state — no-op
            return ToolResult(output=f"No change needed in {path}")
        else:
            # New item — append
            updated = existing.rstrip() + f"\n{target_line}\n"

        path.write_text(updated, encoding="utf-8")
        return ToolResult(output=f"Updated {path}")

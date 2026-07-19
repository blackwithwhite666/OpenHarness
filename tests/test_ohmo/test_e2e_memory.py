"""End-to-end acceptance coverage for durable remember-to-recall behavior."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openharness.tools.base import ToolExecutionContext

from ohmo.evals.memory.provisioning import BackendKind, provision_backend
from ohmo.evals.memory.recall_judge import answer_turn
from ohmo.memory_catalog import MemoryCatalog
from ohmo.memory_store import MemoryStore
from ohmo.memory_tool import OhmoMemoryTool, OhmoMemoryToolInput
from ohmo.prompt_seam import prepare_turn

_OLD_FACT = "I prefer trains over planes."
_NEW_FACT = "I now prefer flying."
_MEMORY_RE = re.compile(r"<memory>\n(?P<memory>.*?)\n</memory>", re.DOTALL)
_QUERY_RE = re.compile(r"QUERY:\n(?P<query>.*?)\n\nReturn only", re.DOTALL)


class _MemoryOnlyCompleter:
    """Deterministically answer only when the requested fact is in memory."""

    async def complete(self, prompt: str) -> str:
        memory_match = _MEMORY_RE.search(prompt)
        query_match = _QUERY_RE.search(prompt)
        assert memory_match is not None
        assert query_match is not None
        memory = memory_match.group("memory")
        query = query_match.group("query").casefold()
        if "prefer" in query and "travel" in query:
            for fact in (_NEW_FACT, _OLD_FACT):
                if fact in memory:
                    return fact
        return "I don't know."


def _context(workspace: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=workspace)


def _usage_count(kind: str, workspace: Path, name: str) -> int:
    if kind == "file":
        usage = MemoryStore(workspace).usage(name)
        return int(usage["use_count"]) if usage is not None else 0
    record = MemoryCatalog(workspace).get("owner", name)
    return record.usage if record is not None else 0


@pytest.mark.parametrize("kind", ["file", "catalog"])
async def test_remember_late_recall_update_forget_and_no_fabrication(kind: BackendKind):
    provisioned = await provision_backend(
        kind,
        run="e2e-memory",
        case="remember-recall",
        sample=0,
        seed_entries=[],
    )
    backend = provisioned.backend
    tool = OhmoMemoryTool(backend)
    completer = _MemoryOnlyCompleter()
    context = _context(provisioned.workspace)
    memory_name = "travel_preference.md"

    try:
        remember_turn = f"remember: {_OLD_FACT}"
        remembered_content = remember_turn.removeprefix("remember: ")
        added = await tool.execute(
            OhmoMemoryToolInput(
                action="add",
                title="Travel preference",
                content=remembered_content,
            ),
            context,
        )
        assert not added.is_error, added.output

        for user_turn, assistant_turn in (
            ("Can you summarize today's weather?", "Weather was discussed."),
            ("Help me name a project folder.", "Project naming was discussed."),
        ):
            await prepare_turn(backend, latest_user_prompt=user_turn)
            await backend.append_turn("user", user_turn)
            await backend.append_turn("assistant", assistant_turn)

        recall_query = "What do I prefer for travel?"
        usage_before_recall = _usage_count(kind, provisioned.workspace, memory_name)
        recalled = await prepare_turn(backend, latest_user_prompt=recall_query)
        assert _OLD_FACT in recalled.memory_block
        answer = await answer_turn(recall_query, recalled.memory_block, complete=completer)
        assert _OLD_FACT in answer
        assert _usage_count(kind, provisioned.workspace, memory_name) > usage_before_recall
        assert [entry.name for entry in await backend.list()][0] == memory_name

        update_turn = f"actually, {_NEW_FACT}"
        updated_content = update_turn.removeprefix("actually, ")
        updated = await tool.execute(
            OhmoMemoryToolInput(
                action="update",
                name=memory_name,
                content=updated_content,
            ),
            context,
        )
        assert not updated.is_error, updated.output
        await prepare_turn(backend, latest_user_prompt="Let's discuss something else first.")

        updated_recall = await prepare_turn(backend, latest_user_prompt=recall_query)
        assert _NEW_FACT in updated_recall.memory_block
        assert _OLD_FACT not in updated_recall.memory_block
        updated_answer = await answer_turn(
            recall_query,
            updated_recall.memory_block,
            complete=completer,
        )
        assert _NEW_FACT in updated_answer
        assert _OLD_FACT not in updated_answer

        removed = await tool.execute(
            OhmoMemoryToolInput(action="remove", name=memory_name),
            context,
        )
        assert not removed.is_error, removed.output
        forgotten_recall = await prepare_turn(backend, latest_user_prompt=recall_query)
        assert _OLD_FACT not in forgotten_recall.memory_block
        assert _NEW_FACT not in forgotten_recall.memory_block
        forgotten_answer = await answer_turn(
            recall_query,
            forgotten_recall.memory_block,
            complete=completer,
        )
        assert "don't know" in forgotten_answer.casefold()

        unknown_query = "What is my preferred hotel in Kyoto?"
        unknown_recall = await prepare_turn(backend, latest_user_prompt=unknown_query)
        assert "Kyoto" not in unknown_recall.memory_block
        assert "## travel_preference.md" not in unknown_recall.memory_block
        unknown_answer = await answer_turn(
            unknown_query,
            unknown_recall.memory_block,
            complete=completer,
        )
        assert "don't know" in unknown_answer.casefold()
        assert "Kyoto" not in unknown_answer
    finally:
        await provisioned.teardown()

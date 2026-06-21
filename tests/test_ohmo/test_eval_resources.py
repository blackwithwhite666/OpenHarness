from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ohmo.evals.resources as eval_resources
from ohmo.evals import get_eval_store, write_ohmo_resource_snapshot
from ohmo.workspace import (
    get_attachments_dir,
    get_contacts_path,
    get_gateway_config_path,
    get_groups_dir,
    get_memory_dir,
    get_plugins_dir,
    get_reminders_path,
    get_sessions_dir,
    get_skills_dir,
    initialize_workspace,
)


class FakeTool:
    name = "fake_search"
    description = "Search already registered local metadata"

    def to_api_schema(self):
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }


class FakeToolRegistry:
    def list_tools(self):
        return [FakeTool()]


def test_ohmo_resource_snapshot_writes_metadata_only_manifest(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    (workspace / "todos").mkdir()
    (workspace / "todos" / "private-todo.md").write_text(
        "# TODO\n- [ ] send Alice the private plan\n",
        encoding="utf-8",
    )
    (get_memory_dir(workspace) / "Private_Project.md").write_text(
        "# Private Project\nsecret memory body\n",
        encoding="utf-8",
    )
    (get_sessions_dir(workspace) / "session-secret.json").write_text(
        json.dumps({"session_key": "telegram:secret", "summary": "private summary"}),
        encoding="utf-8",
    )
    (get_attachments_dir(workspace) / "passport-preview.png").write_bytes(b"\x89PNG secret")
    (get_skills_dir(workspace) / "secret-skill.md").write_text(
        "run private command",
        encoding="utf-8",
    )
    (get_plugins_dir(workspace) / "secret-plugin.md").write_text(
        "plugin command token",
        encoding="utf-8",
    )
    group_dir = get_groups_dir(workspace) / "feishu"
    group_dir.mkdir(parents=True)
    (group_dir / "secret-chat.json").write_text(
        json.dumps({"name": "Secret Group", "owner_open_id": "ou_secret", "cwd": "/private"}),
        encoding="utf-8",
    )
    get_gateway_config_path(workspace).write_text(
        json.dumps({"channel_configs": {"telegram": {"token": "secret-token"}}}),
        encoding="utf-8",
    )
    get_reminders_path(workspace).write_text(
        json.dumps(
            [
                {
                    "text": "call Alice",
                    "next_fire_at": 999999,
                    "chat_id": "chat-secret",
                }
            ]
        ),
        encoding="utf-8",
    )
    get_contacts_path(workspace).write_text(
        json.dumps(
            {
                "telegram:chat-secret": {
                    "display_name": "Alice",
                    "username": "alice_secret",
                    "chat_id": "chat-secret",
                }
            }
        ),
        encoding="utf-8",
    )

    store = get_eval_store(workspace)
    result = write_ohmo_resource_snapshot(
        store=store,
        episode_id="episode-1",
        workspace=workspace,
        bundle=SimpleNamespace(tool_registry=FakeToolRegistry()),
    )

    assert result.relative_path == "states/episode-1/world_before.json"
    assert result.path == store.root / result.relative_path
    assert result.path.is_file()
    assert result.resource_count == len(result.manifest.resources)
    assert result.tool_count == 1
    assert result.local_resource_count == result.resource_count - 1

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    resources = {resource["resource_id"]: resource for resource in payload["resources"]}
    for resource_id in (
        "ohmo.workspace.soul_md",
        "ohmo.workspace.user_md",
        "ohmo.workspace.identity_md",
        "ohmo.workspace.bootstrap_md",
        "ohmo.workspace.memory_dir",
        "ohmo.workspace.memory_index",
        "ohmo.workspace.sessions_dir",
        "ohmo.workspace.attachments_dir",
        "ohmo.workspace.skills_dir",
        "ohmo.workspace.plugins_dir",
        "ohmo.workspace.groups_dir",
        "ohmo.workspace.state_json",
        "ohmo.workspace.gateway_json",
        "ohmo.workspace.gateway_restart_notice_json",
        "ohmo.workspace.reminders_json",
        "ohmo.workspace.contacts_json",
        "ohmo.workspace.todos_dir",
        "ohmo.runtime_tool.fake_search",
    ):
        assert resource_id in resources
    assert "ohmo.workspace.logs_dir" not in resources

    memory_dir = resources["ohmo.workspace.memory_dir"]
    assert memory_dir["path"] == "memory"
    assert memory_dir["exists"] is True
    assert memory_dir["metadata"]["file_count"] >= 2
    assert memory_dir["metadata"]["total_size_bytes"] > 0
    assert isinstance(memory_dir["metadata"]["newest_mtime_ns"], int)

    contacts = resources["ohmo.workspace.contacts_json"]
    assert contacts["metadata"]["parse_status"] == "ok"
    assert contacts["metadata"]["json_type"] == "object"
    assert contacts["metadata"]["record_count"] == 1
    reminders = resources["ohmo.workspace.reminders_json"]
    assert reminders["metadata"]["parse_status"] == "ok"
    assert reminders["metadata"]["json_type"] == "array"
    assert reminders["metadata"]["record_count"] == 1
    gateway = resources["ohmo.workspace.gateway_json"]
    assert gateway["metadata"]["parse_status"] == "ok"

    tool = resources["ohmo.runtime_tool.fake_search"]
    assert tool["kind"] == "runtime_tool"
    assert tool["name"] == "fake_search"
    assert tool["metadata"]["description"] == FakeTool.description
    assert len(tool["metadata"]["input_schema_hash"]) == 64
    assert tool["metadata"]["input_schema"]["required"] == ["query"]

    manifest_text = result.path.read_text(encoding="utf-8")
    for sensitive_fragment in (
        "Private_Project.md",
        "private-todo.md",
        "passport-preview.png",
        "session-secret.json",
        "telegram:secret",
        "private summary",
        "secret-token",
        "call Alice",
        "chat-secret",
        "alice_secret",
        "Secret Group",
        "ou_secret",
        "run private command",
        "plugin command token",
    ):
        assert sensitive_fragment not in manifest_text


def test_directory_snapshot_caps_recursive_scan_without_path_leaks(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(eval_resources, "_DIRECTORY_AGGREGATE_ENTRY_LIMIT", 3)
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    secret_nested = get_memory_dir(workspace) / "secret-nested-dir"
    secret_nested.mkdir()
    for index in range(8):
        (secret_nested / f"private-memory-{index}.md").write_text(
            f"secret memory {index}",
            encoding="utf-8",
        )

    store = get_eval_store(workspace)
    result = write_ohmo_resource_snapshot(
        store=store,
        episode_id="episode-capped",
        workspace=workspace,
    )

    payload = json.loads(result.path.read_text(encoding="utf-8"))
    resources = {resource["resource_id"]: resource for resource in payload["resources"]}
    memory_metadata = resources["ohmo.workspace.memory_dir"]["metadata"]

    assert memory_metadata["truncated"] is True
    assert memory_metadata["entry_limit"] == 3
    assert memory_metadata["visited_count"] == 3

    manifest_text = result.path.read_text(encoding="utf-8")
    assert "secret-nested-dir" not in manifest_text
    assert "private-memory-" not in manifest_text
    assert "secret memory" not in manifest_text


def test_state_key_metadata_for_reminders_is_private(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    get_reminders_path(workspace).write_text(
        json.dumps(
            [
                {
                    "id": "reminder-1",
                    "channel": "telegram",
                    "chat_id": "chat-1",
                    "mode": "static",
                    "tz": "UTC",
                    "dtstart": "2026-01-02T09:00:00+00:00",
                    "rrule": None,
                    "status": "active",
                    "summary": "secret dentist appointment",
                },
                {
                    "id": "reminder-2",
                    "channel": "telegram",
                    "chat_id": "chat-1",
                    "mode": "static",
                    "tz": "UTC",
                    "dtstart": "2026-01-03T09:00:00+00:00",
                    "rrule": None,
                    "status": "done",
                    "summary": "secret renewal note",
                },
            ]
        ),
        encoding="utf-8",
    )

    manifest = eval_resources.build_ohmo_resource_snapshot(
        episode_id="episode-reminders",
        workspace=workspace,
    )

    resources = {resource.resource_id: resource for resource in manifest.resources}
    metadata = resources["ohmo.workspace.reminders_json"].metadata
    assert metadata["record_count"] == 2
    assert metadata["status_counts"] == {"active": 1, "done": 1}
    assert len(metadata["entry_keys"]) == 2
    assert all(_is_hex16(key) for key in metadata["entry_keys"])

    serialized = manifest.model_dump_json()
    assert "secret dentist appointment" not in serialized
    assert "secret renewal note" not in serialized


def test_state_key_metadata_for_memory_uses_stems_only(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    memory_dir = get_memory_dir(workspace)
    (memory_dir / "project_alpha.md").write_text("private alpha body", encoding="utf-8")
    (memory_dir / "project_beta.md").write_text("private beta body", encoding="utf-8")

    manifest = eval_resources.build_ohmo_resource_snapshot(
        episode_id="episode-memory",
        workspace=workspace,
    )

    resources = {resource.resource_id: resource for resource in manifest.resources}
    metadata = resources["ohmo.workspace.memory_dir"].metadata
    expected_stems = sorted(
        child.stem
        for child in memory_dir.iterdir()
        if child.is_file() and child.suffix == ".md"
    )
    assert metadata["entry_keys"] == expected_stems
    assert metadata["entry_count"] == len(expected_stems)


def test_state_key_metadata_for_todos_hashes_items(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    todos_dir = workspace / "todos"
    todos_dir.mkdir()
    (todos_dir / "list-1.md").write_text(
        "# TODO\n- [ ] Call Alice about the private plan\n- [x] Buy secret supplies\n",
        encoding="utf-8",
    )

    manifest = eval_resources.build_ohmo_resource_snapshot(
        episode_id="episode-todos",
        workspace=workspace,
    )

    resources = {resource.resource_id: resource for resource in manifest.resources}
    metadata = resources["ohmo.workspace.todos_dir"].metadata
    assert metadata["entry_count"] == 2
    assert len(metadata["entry_keys"]) == 2
    assert all(_is_hex16(key) for key in metadata["entry_keys"])

    serialized = manifest.model_dump_json()
    assert "Call Alice about the private plan" not in serialized
    assert "Buy secret supplies" not in serialized


def test_resource_snapshot_phase_paths_and_validation(tmp_path: Path):
    workspace = tmp_path / ".ohmo-home"
    initialize_workspace(workspace)
    store = get_eval_store(workspace)

    before = write_ohmo_resource_snapshot(
        store=store,
        episode_id="episode-phases",
        workspace=workspace,
        phase="world_before",
    )
    after = write_ohmo_resource_snapshot(
        store=store,
        episode_id="episode-phases",
        workspace=workspace,
        phase="world_after",
    )

    assert before.relative_path == "states/episode-phases/world_before.json"
    assert after.relative_path == "states/episode-phases/world_after.json"
    assert before.path.is_file()
    assert after.path.is_file()

    with pytest.raises(ValueError, match="unknown snapshot phase"):
        write_ohmo_resource_snapshot(
            store=store,
            episode_id="episode-phases",
            workspace=workspace,
            phase="../unsafe",
        )


def _is_hex16(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(char in "0123456789abcdef" for char in value)
    )

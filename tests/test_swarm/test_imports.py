"""Import regression tests for swarm startup."""

from __future__ import annotations

import importlib
import sys


def _is_tools_or_swarm(module_name: str) -> bool:
    return (
        module_name == "openharness.tools"
        or module_name.startswith("openharness.tools.")
        or module_name == "openharness.swarm"
        or module_name.startswith("openharness.swarm.")
    )


def test_create_default_tool_registry_does_not_import_mailbox_eagerly():
    # Snapshot the modules we evict so we can restore the *original* objects
    # afterwards. Reimporting openharness.tools creates a fresh TraceTool class;
    # if we left it in sys.modules, later tests holding the original class would
    # see isinstance() fail across the two duplicate class identities.
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if _is_tools_or_swarm(name)}
    try:
        tools = importlib.import_module("openharness.tools")
        registry = tools.create_default_tool_registry()

        assert registry.get("bash") is not None
        assert "openharness.swarm.mailbox" not in sys.modules
        assert "openharness.swarm.lockfile" not in sys.modules
    finally:
        for name in [name for name in list(sys.modules) if _is_tools_or_swarm(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)

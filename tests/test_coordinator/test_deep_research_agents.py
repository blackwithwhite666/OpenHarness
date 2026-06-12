"""Tests for the deep-research and research-verification agent definitions.

Covers ADR adrs/deep-research-openharness.md §3 #8/#9, §5 M1:

* both defs load via ``get_agent_definition``;
* both pin a CONCRETE (non-inherit, non-None) model so the eval path is
  deterministic (ADR §5/§7 — ``build_inherited_cli_flags`` drops ``--model`` for
  "inherit");
* the deep-research toolset partitions to Serper-MCP search + fetch + bash
  (browser) + todo + read + agent (to spawn the verifier);
* the research-verification def is READ/FETCH only and is DISTINCT from the
  pre-existing code/test ``verification`` def, which must stay untouched;
* the bundled ``deep-research.md`` skill parses (frontmatter + body).

All offline. No live agent run (no model in this env).
"""

from __future__ import annotations

from openharness.coordinator.agent_definitions import (
    get_agent_definition,
    get_builtin_agent_definitions,
)
from openharness.skills.bundled import get_bundled_skills

#: Concrete model id the eval runner pins (tests/eval/gaia/run_subset.py
#: DEFAULT_MODEL). The deep-research defs must pin the same kind of concrete id.
_EXPECTED_MODEL = "claude-opus-4-8"

#: Provider-specific aliases / inherit sentinels that are NOT deterministic for
#: evals — the deep-research defs must avoid all of these (ADR §5/§7).
_NON_PINNED_MODELS = {None, "inherit", "haiku", "sonnet", "opus", "default"}


# ---------------------------------------------------------------------------
# deep-research AgentDefinition
# ---------------------------------------------------------------------------


def test_deep_research_def_loads():
    agent = get_agent_definition("deep-research")
    assert agent is not None
    assert agent.name == "deep-research"
    assert agent.subagent_type == "deep-research"
    assert agent.source == "builtin"
    assert agent.system_prompt


def test_deep_research_model_is_pinned_not_inherit():
    agent = get_agent_definition("deep-research")
    assert agent.model == _EXPECTED_MODEL
    assert agent.model not in _NON_PINNED_MODELS


def test_deep_research_toolset_partition():
    """Toolset routes to the Serper MCP + fetch + bash(browser) + verifier (ADR §2f)."""
    agent = get_agent_definition("deep-research")
    assert agent.tools is not None
    tools = set(agent.tools)
    # Serper search MCP (mcp__<server>__<tool> for the google_search server).
    assert "mcp__google_search__search" in tools
    # Breadth/triage fetch + bash (drives `browser-cli md` for depth).
    assert "web_fetch" in tools
    assert "bash" in tools
    # Plan ledger + attachment read + spawning the research-verification agent.
    assert "todo_write" in tools
    assert "read_file" in tools
    assert "agent" in tools
    # Must NOT silently expose file-mutation tools.
    assert "file_write" not in tools
    assert "edit_file" not in tools


def test_deep_research_requires_serper_mcp():
    agent = get_agent_definition("deep-research")
    assert agent.required_mcp_servers == ["google_search"]


def test_deep_research_prompt_has_loop_and_final_answer():
    agent = get_agent_definition("deep-research")
    prompt = agent.system_prompt or ""
    # The parallel-ReAct loop + the <final_answer> sentinel (ADR §2f).
    assert "<final_answer>" in prompt
    assert "mcp__google_search__search" in prompt
    assert "browser-cli md" in prompt
    assert "research-verification" in prompt
    assert "cwd" in prompt  # attachment contract


# ---------------------------------------------------------------------------
# research-verification AgentDefinition
# ---------------------------------------------------------------------------


def test_research_verification_def_loads():
    agent = get_agent_definition("research-verification")
    assert agent is not None
    assert agent.name == "research-verification"
    assert agent.subagent_type == "research-verification"
    assert agent.source == "builtin"
    assert agent.background is True


def test_research_verification_model_is_pinned_not_inherit():
    agent = get_agent_definition("research-verification")
    assert agent.model == _EXPECTED_MODEL
    assert agent.model not in _NON_PINNED_MODELS


def test_research_verification_is_read_fetch_only():
    agent = get_agent_definition("research-verification")
    # Allowed: read + fetch + bash + search. No write / spawn.
    assert agent.tools is not None
    assert set(agent.tools) == {"read_file", "web_fetch", "bash", "mcp__google_search__search"}
    # Denylist blocks mutation + recursive agent spawning + plan exit.
    assert agent.disallowed_tools is not None
    deny = set(agent.disallowed_tools)
    for blocked in ("agent", "file_edit", "file_write", "notebook_edit"):
        assert blocked in deny


def test_research_verification_prompt_is_citation_checker_not_code_verifier():
    """Distinct prompt: claims->sources, verdict — explicitly NOT a code verifier."""
    agent = get_agent_definition("research-verification")
    prompt = (agent.system_prompt or "").lower()
    # It checks claims against cited sources and emits a verdict.
    assert "claim" in prompt
    assert "cited source" in prompt
    assert "verdict:" in prompt
    assert "supported" in prompt and "unsupported" in prompt
    # It explicitly disclaims being the code/test verifier (it tells the agent to
    # IGNORE builds/linters/test runners — the opposite of the code verifier,
    # which is built around running them).
    assert "not a code/test verifier" in prompt
    assert "ignore builds" in prompt
    # And its critical reminder is citation-scoped, not project-edit-scoped.
    reminder = (agent.critical_system_reminder or "").lower()
    assert "citation-verification" in reminder


# ---------------------------------------------------------------------------
# The pre-existing code/test `verification` def must be DISTINCT and UNTOUCHED
# ---------------------------------------------------------------------------


def test_code_verification_def_is_distinct_and_unchanged():
    code_verifier = get_agent_definition("verification")
    research_verifier = get_agent_definition("research-verification")
    assert code_verifier is not None
    assert research_verifier is not None
    # Distinct names / subagent types.
    assert code_verifier.name != research_verifier.name
    # The code verifier keeps its original "inherit" model (NOT modified).
    assert code_verifier.model == "inherit"
    # The code verifier's prompt is still the build/test verifier.
    code_prompt = (code_verifier.system_prompt or "").lower()
    assert "test suite" in code_prompt or "builds, tests" in code_prompt
    # The research verifier's prompt is NOT the same object.
    assert code_verifier.system_prompt != research_verifier.system_prompt


def test_both_deep_research_defs_are_builtin():
    names = {a.name for a in get_builtin_agent_definitions()}
    assert "deep-research" in names
    assert "research-verification" in names
    # The original built-ins are all still present.
    for original in ("general-purpose", "Explore", "Plan", "worker", "verification"):
        assert original in names


# ---------------------------------------------------------------------------
# Bundled deep-research skill parses
# ---------------------------------------------------------------------------


def test_deep_research_skill_parses():
    skills = {s.name: s for s in get_bundled_skills()}
    assert "deep-research" in skills, sorted(skills)
    skill = skills["deep-research"]
    assert skill.source == "bundled"
    assert skill.command_name == "deep-research"
    # Frontmatter description survived YAML block-scalar parsing.
    assert skill.description
    assert "research" in skill.description.lower()


def test_deep_research_skill_body_documents_loop_and_tools():
    skills = {s.name: s for s in get_bundled_skills()}
    body = skills["deep-research"].content
    # Loop + the real tool names + the <final_answer> sentinel.
    assert "mcp__google_search__search" in body
    assert "browser-cli md" in body
    assert "web_fetch" in body
    assert "<final_answer>" in body
    assert "research-verification" in body
    # Body sections (frontmatter form, ADR §3 — skill-creator shape).
    assert "## When to use" in body
    assert "## Workflow" in body

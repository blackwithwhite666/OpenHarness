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
_EXPECTED_MODEL = "gpt-5.5"  # deployed gateway model (Codex); pinned, not "inherit"

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
    assert "mcp__google_search__google_search" in tools
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
    assert "mcp__google_search__google_search" in prompt
    assert "browser-cli md" in prompt
    assert "research-verification" in prompt
    assert "cwd" in prompt  # attachment contract


def test_deep_research_prompt_uses_native_or_internal_image_handling():
    agent = get_agent_definition("deep-research")
    prompt = agent.system_prompt or ""
    assert "provided natively" in prompt
    assert "internally described before your model call" in prompt
    assert "do NOT call `image_to_text`" in prompt
    assert "Pillow/OpenCV" in prompt


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
    assert set(agent.tools) == {"read_file", "web_fetch", "bash", "mcp__google_search__google_search"}
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
    assert "deep-research-noverify" in names
    # The original built-ins are all still present.
    for original in ("general-purpose", "Explore", "Plan", "worker", "verification"):
        assert original in names


# ---------------------------------------------------------------------------
# deep-research-noverify — research-only ablation arm (no in-loop verifier)
# ---------------------------------------------------------------------------


def test_deep_research_noverify_def_loads_and_model_pinned():
    agent = get_agent_definition("deep-research-noverify")
    assert agent is not None
    assert agent.name == "deep-research-noverify"
    assert agent.subagent_type == "deep-research-noverify"
    assert agent.source == "builtin"
    assert agent.system_prompt
    assert agent.model == _EXPECTED_MODEL
    assert agent.model not in _NON_PINNED_MODELS
    assert agent.required_mcp_servers == ["google_search"]


def test_deep_research_noverify_drops_agent_tool():
    """Same research partition as deep-research MINUS `agent` (cannot spawn the verifier)."""
    agent = get_agent_definition("deep-research-noverify")
    tools = set(agent.tools or [])
    assert "agent" not in tools  # the whole point — no verifier spawn
    # The rest of the research loop is intact.
    for kept in ("mcp__google_search__google_search", "web_fetch", "bash", "todo_write", "read_file"):
        assert kept in tools


def test_deep_research_noverify_prompt_has_no_verify_step():
    """The verify step (6) is actually gone — guards against a no-op .replace that
    would silently ship verification in the 'noverify' arm."""
    noverify = get_agent_definition("deep-research-noverify").system_prompt or ""
    assert "research-verification" not in noverify
    assert "VERIFY / CITE" not in noverify
    # But it is still the deep-research loop with the sentinel + parallel search.
    assert "<final_answer>" in noverify
    assert "mcp__google_search__google_search" in noverify
    assert "ANSWER." in noverify


def test_deep_research_keeps_its_verify_step_unchanged():
    """Regression guard: deriving noverify via .replace must NOT mutate the original."""
    verifying = get_agent_definition("deep-research").system_prompt or ""
    assert "VERIFY / CITE" in verifying
    assert "research-verification" in verifying


# ---------------------------------------------------------------------------
# M1-regression fixes (ADR §5 iteration 1): correct tool name, hard budget,
# browser demoted to last-resort, terse exact-match answer format.
# ---------------------------------------------------------------------------


def test_deep_research_prompt_has_budget_and_terse_format():
    prompt = get_agent_definition("deep-research").system_prompt or ""
    # #1 correct Serper tool name + web_search fallback documented.
    assert "mcp__google_search__google_search" in prompt
    assert "web_search" in prompt
    # #2 hard budget on turns/waves/fetches so runs stop instead of timing out.
    assert "HARD BUDGET" in prompt
    assert "14 assistant turns" in prompt
    assert "2 search waves" in prompt
    # #3 browser demoted to a last resort (default is web_fetch).
    assert "LAST-RESORT" in prompt
    # #6 terse, exact answer; banned qualifier words listed.
    assert "MINIMAL exact value" in prompt
    assert "number of" in prompt


def test_deep_research_tools_include_web_search_fallback():
    for name in ("deep-research", "deep-research-noverify"):
        tools = set(get_agent_definition(name).tools or [])
        assert "web_search" in tools
        assert "mcp__google_search__google_search" in tools


def test_deep_research_noverify_inherits_budget_and_terse_format():
    # The fixes live in the shared base prompt -> the noverify arm gets them too.
    noverify = get_agent_definition("deep-research-noverify").system_prompt or ""
    assert "HARD BUDGET" in noverify
    assert "MINIMAL exact value" in noverify
    assert "LAST-RESORT" in noverify


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
    assert "mcp__google_search__google_search" in body
    assert "browser-cli md" in body
    assert "web_fetch" in body
    assert "<final_answer>" in body
    assert "research-verification" in body
    # Body sections (frontmatter form, ADR §3 — skill-creator shape).
    assert "## When to use" in body
    assert "## Workflow" in body

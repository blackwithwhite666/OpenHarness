"""Tests for runtime system-prompt assembly — local-rules token cap."""

from __future__ import annotations

from openharness.config.settings import Settings
from openharness.prompts import build_runtime_system_prompt
from openharness.prompts.context import (
    DEFAULT_LOCAL_RULES_MAX_TOKENS,
    _truncate_to_token_budget,
)
from openharness.services.token_estimation import estimate_tokens


def test_truncate_to_token_budget():
    assert _truncate_to_token_budget("hello", 10_000) == "hello"  # under budget, unchanged
    big = "x" * 100_000  # ~25k tokens
    out = _truncate_to_token_budget(big, 1_000)
    assert "truncated to ~1000 tokens" in out
    assert estimate_tokens(out) < 1_200  # ~budget + marker overhead
    assert _truncate_to_token_budget(big, 0) == big  # cap disabled (<= 0)


def test_build_runtime_caps_runaway_local_rules(tmp_path, monkeypatch):
    # A runaway auto-generated rules.md (~350k tokens) must not dominate the prompt.
    huge = "".join(f"- /home/x/artifact_{i}.txt\n" for i in range(50_000))
    monkeypatch.setattr("openharness.prompts.context.load_local_rules", lambda: huge)

    prompt = build_runtime_system_prompt(
        Settings(system_prompt="BASE"),
        cwd=tmp_path,
        latest_user_prompt="hi",
        include_project_memory=False,
    )

    assert "# Local Environment Rules" in prompt
    assert f"truncated to ~{DEFAULT_LOCAL_RULES_MAX_TOKENS} tokens" in prompt
    assert prompt.count("artifact_") < 50_000          # not all lines survived
    assert estimate_tokens(prompt) < 50_000            # bounded far below the ~350k input


def test_build_runtime_local_rules_cap_env_override(tmp_path, monkeypatch):
    huge = "".join(f"- line {i}\n" for i in range(50_000))
    monkeypatch.setattr("openharness.prompts.context.load_local_rules", lambda: huge)
    monkeypatch.setenv("OPENHARNESS_LOCAL_RULES_MAX_TOKENS", "500")

    prompt = build_runtime_system_prompt(
        Settings(system_prompt="BASE"),
        cwd=tmp_path,
        include_project_memory=False,
    )

    assert "truncated to ~500 tokens" in prompt


def test_build_runtime_local_rules_cap_disabled(tmp_path, monkeypatch):
    rules = "- only a few rules\n- second rule\n"
    monkeypatch.setattr("openharness.prompts.context.load_local_rules", lambda: rules)
    monkeypatch.setenv("OPENHARNESS_LOCAL_RULES_MAX_TOKENS", "0")

    prompt = build_runtime_system_prompt(
        Settings(system_prompt="BASE"),
        cwd=tmp_path,
        include_project_memory=False,
    )

    assert "only a few rules" in prompt
    assert "truncated to" not in prompt

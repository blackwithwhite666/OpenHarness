# Plan: ohmo-kvcache-telemetry

Measure-first slice of the KV-cache ADR (`adrs/kv-cache-economics.md`). On the
Codex/Responses prod path the base prompt is already byte-stable (no date) so
OpenAI auto-caches it — but we currently **drop `cached_tokens`** on parse, so the
cache hit-rate is invisible. This plan makes it observable and adds a
session-stable `prompt_cache_key`. NO prompt-layout refactor (deferred until the
numbers justify it). NO Anthropic breakpoints (that route is not prod).

Defaults: model=gpt-5.6-sol reasoning=xhigh workdir=/Users/dldmitry/tmp/OpenHarness timeout=45m
Branch: kv-cache-telemetry (agents do code+tests+local checks only; never commit/push/PR)

## Step: s1-usage-telemetry — Parse & record provider cache tokens
Verify: cd /Users/dldmitry/tmp/OpenHarness && uv run ruff check src/openharness/api ohmo/evals/recorder.py tests/test_api/test_cache_usage.py && uv run pytest tests/test_api/test_cache_usage.py -q && uv run python -c "from openharness.api.usage import UsageSnapshot; f=UsageSnapshot.model_fields; assert 'cached_input_tokens' in f and 'cache_write_input_tokens' in f"
Contract: see ~/.todovan/ohmo-kvcache-telemetry/s1-usage-telemetry.md

## Step: s2-prompt-cache-key — Thread a session-stable prompt_cache_key
Verify: cd /Users/dldmitry/tmp/OpenHarness && uv run ruff check src/openharness/api src/openharness/engine ohmo/gateway/runtime.py tests/test_api/test_prompt_cache_key.py && uv run pytest tests/test_api/test_prompt_cache_key.py -q && uv run python -c "from openharness.api.client import ApiMessageRequest; assert 'cache_key' in ApiMessageRequest.__dataclass_fields__"
Contract: see ~/.todovan/ohmo-kvcache-telemetry/s2-prompt-cache-key.md

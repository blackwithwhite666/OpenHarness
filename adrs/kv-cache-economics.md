# ADR: KV-cache economics — stable prompt prefix + provider cache hints

- **Status:** Proposed
- **Date:** 2026-07-21
- **Owner:** ohmo / OpenHarness
- **Scope:** `ohmo/` gateway prompt seam + `src/openharness/api/*` client paths + `src/openharness/prompts/*`

## Context

The Hermes Agent harness treats the provider prompt cache as a first-class cost
lever: it keeps a **byte-stable prompt prefix** across a session, pushes all
volatile data into the user message, and sets explicit cache breakpoints. On
long, tool-heavy trajectories that yields on the order of ~75% input-token
savings. ohmo currently does **none** of this, so we re-pay full input price on
every turn. Verified in code:

1. **No cache breakpoints on the Anthropic path.** `src/openharness/api/client.py:211`
   passes `params["system"] = request.system_prompt` as a plain string, and
   `ConversationMessage.to_api_param` (`src/openharness/engine/messages.py:101`)
   serializes content blocks with no `cache_control`. Anthropic does **not**
   cache without explicit `cache_control` breakpoints → the Claude path gets
   **zero** prompt caching.

2. **No `prompt_cache_key` on the Codex/OpenAI paths.** The prod provider is
   Codex (`ohmo/gateway/models.py:18` `provider_profile = "codex"`). The Codex
   Responses request (`src/openharness/api/codex_client.py:284-295`) sets
   `instructions` + `input` but no `prompt_cache_key`; `openai_client.py`
   likewise. OpenAI auto-caches on the `instructions`+`input` **prefix**, so a
   prefix that changes every turn defeats it.

3. **The system prompt is recomposed every turn and is not byte-stable.** The
   gateway calls `set_system_prompt(...)` per turn (`ohmo/gateway/runtime.py:770`).
   The memory-free base embeds a **daily UTC date**
   (`src/openharness/prompts/environment.py:128` →
   `src/openharness/prompts/system_prompt.py:71` `- Date: {env.date}`) and
   re-reads SOUL/USER/IDENTITY from disk each turn — so the cached prefix is
   invalidated at least once per day and on any persona-file touch.

4. **Volatile data lives in the cached prefix, not the user message.** The one
   dynamic datum in the prompt (date) sits in the system prompt — the exact
   opposite of Hermes, which injects the timestamp into the user message via a
   pre-call seam.

### What already helps us

- `ohmo/prompt_seam.py:114` `compose_runtime_prompt` already separates a
  `memory_free_base` from the volatile memory `snapshot` and appends memory
  **last** (`prompt_seam.py:150`). So the volatile tail is already positioned
  correctly — we mainly need to make the base stable and mark the boundary.
- There is already a per-turn **user-message assembly seam**,
  `_build_inbound_user_message` (`ohmo/gateway/runtime.py:2121`), which prepends
  a `[Speaker]` header and attachment notes to the user text. That is the
  natural injection point for volatile hints.

### Verified against Hermes source

This ADR was cross-checked against the real Hermes implementation
(`~/tmp/hermes-agent`, `NousResearch/hermes-agent`) — not just the talk. What
the code actually does:

- **Whole system prompt is frozen per session.** `agent/system_prompt.py:3` —
  *"built once per session and reused across all turns — only context
  compression triggers a rebuild"* — with a documented invariant and the exact
  `stable / context / volatile` layer contract (`system_prompt.py:12-18`). The
  `volatile` layer (memory snapshot, USER.md) is frozen too; a mid-session
  memory write only takes effect next session. This is stronger than "freeze the
  base."
- **Cache strategy = `system_and_3`.** `agent/prompt_caching.py:84` places **4**
  `cache_control` breakpoints — system prompt + the **last 3** non-system
  messages (rolling window), TTL `5m` or `1h`. It is native-Anthropic vs
  envelope (OpenRouter) aware and deliberately skips empty messages so no
  breakpoint is wasted (`_can_carry_marker`, `prompt_caching.py:52`).
- **Caching is Anthropic-only; there is NO `prompt_cache_key`.** Auto-enabled
  only for Claude models on native-Anthropic/OpenRouter
  (`website/docs/developer-guide/context-compression-and-caching.md:402-406`).
  For OpenAI/Codex-format routes Hermes relies **purely on prefix stability** —
  it sets no explicit cache key.
- **Model identity is part of the cache key.** A mid-session model or credential
  swap yields zero cache hits (caching doc §"Cache-Aware Design Patterns" #5).
- Talk claims that did **not** survive code review: compaction does **not**
  spawn a new session (head + summary + tail are reassembled in place); and the
  cache logic is not "dumb" — the empty-message skip and native/envelope
  handling are deliberate.

## Decision

Adopt a provider-agnostic core (stable prefix + volatile-in-user-message) plus a
thin per-provider cache-hint layer.

- **D1 — Freeze the session prefix.** The system prompt becomes byte-stable for
  the life of a session (rebuilt only on compaction, like Hermes): (a) remove the
  `- Date:` line; (b) snapshot SOUL/USER/IDENTITY once per session and reuse the
  rendered base verbatim instead of re-reading per turn.
  - *Deliberate divergence from Hermes:* Hermes freezes the **whole** prompt
    including the memory snapshot, so fresh facts only land next session. ohmo
    keeps its per-turn memory value but moves the volatile memory delta **and
    derived recall out of the system prompt into the user message** (D2), instead
    of appending them to the system-prompt tail. Net: the prefix stays cacheable
    *and* fresh facts still reach the model this turn.

- **D2 — Volatile data goes in the user message.** Inject the current timestamp,
  the per-turn memory delta, and derived recall into the user message through the
  existing `_build_inbound_user_message` seam (e.g. `[Current time]` + `[Memory]`
  / `[Recall]` sections) — never into the frozen system prompt.

- **D3 — Anthropic cache breakpoints (`system_and_3`, mirror Hermes).** In
  `api/client.py`, render `system` as a content-block list and place **4**
  `cache_control: {"type": "ephemeral"}` breakpoints: the system prompt + the
  **last 3** non-system messages (rolling window). TTL `5m` default, `1h`
  configurable. Skip empty / marker-incapable messages so no breakpoint is
  wasted, and honor native-Anthropic vs envelope (OpenRouter) placement. Near
  port of Hermes `agent/prompt_caching.py`. Requires `messages.to_api_param` to
  carry `cache_control` on a block.

- **D4 — `prompt_cache_key` on Codex/OpenAI (ohmo extension, beyond Hermes).**
  Hermes sets no cache key and relies on prefix stability for OpenAI-format
  routes. Because ohmo's prod route is the Codex **Responses API**, add
  `prompt_cache_key = <stable per-session key>` (`api/codex_client.py`,
  `api/openai_client.py`) as an additive routing win on top of the now-stable
  prefix. Explicitly an ohmo addition, not a Hermes match.

- **D5 — Measure it.** Surface provider cache counters
  (`cache_read_input_tokens` / `cached_tokens`) in usage logging and add a
  session-level cache-hit metric so the win is observed, not assumed.

## Consequences and non-goals

- **The system prompt stops changing mid-session.** Fresh facts still reach the
  model the same turn — but via the user message (D2), not the frozen system
  prompt. Memory/recall gating is unchanged; only its placement moves.
- **Model / credential identity is part of the cache key.** A mid-session model
  swap (`/model`, primary-model fallback, credential-pool rotation) yields zero
  cache hits and re-reads the full conversation at undiscounted price (Hermes
  caching doc §Design-Patterns #5). Do not add features that silently swap the
  model/credentials mid-session; if ohmo has provider fallback, warn on switch.
- **Non-goal: compaction.** The compaction-side gaps confirmed against the
  Hermes doc (summary-update-in-place via `_previous_summary`; first-message
  anchor `protect_first_n=3`; token-budget tail `protect_last_n=20` +
  `target_ratio=0.20`) are a separate P2 ADR. Note the talk's "new session on
  compaction" is **not** real and is dropped.
- **Risk — persona edits mid-session.** If a user edits `soul.md`/`user.md`
  during a session, the change applies next session, not immediately. Acceptable
  and consistent with the freeze model; document in the runbook.
- **Risk — over-breakpointing on Anthropic.** Cap at 4 breakpoints (Anthropic
  limit) and only mark stable boundaries; a breakpoint inside volatile text is
  wasted spend.

## Implementation plan

Each stage leaves the tree green and is independently verifiable. Run from repo
root with `uv run`.

### Stage 1 — Move the timestamp out of the system prompt into the user message

- **What:** Delete the `- Date: {env.date}` line from the rendered system
  prompt; add a `[Current time]` line (ISO-8601 + tz) to the assembled user
  message.
- **Files:** `src/openharness/prompts/system_prompt.py:71` (remove the date
  line; keep `EnvironmentInfo.date` available for callers that still want it);
  `ohmo/gateway/runtime.py:2121` `_build_inbound_user_message` (prepend the
  time line alongside the `[Speaker]` header).
- **Verify:**
  - `uv run pytest tests/test_ohmo -q -k "user_message or speaker"`
  - New assertions: system prompt contains no `Date:` line; user message
    contains a `[Current time]` line.

### Stage 2 — Freeze the full system prompt; route memory to the user message

- **What:** Compute the whole system prompt (persona + SOUL/USER/IDENTITY +
  skills + platform hints) **once per session** and reuse it byte-for-byte. Stop
  appending the per-turn memory snapshot / derived recall to the system-prompt
  tail (`compose_runtime_prompt`); instead render them into the user-message seam
  from Stage 1 (`[Memory]` / `[Recall]` sections).
- **Files:** `ohmo/gateway/runtime.py` (cache the frozen prompt on the session
  object; stop calling `set_system_prompt` per turn); `ohmo/prompt_seam.py:114`
  (`compose_runtime_prompt` no longer appends `snapshot` to the base — the
  snapshot text is returned for the user-message seam); `ohmo/gateway/runtime.py:2121`
  (`_build_inbound_user_message` gains the memory/recall sections).
- **Verify:**
  - `uv run pytest tests/test_ohmo -q -k "prompt_seam or compose or base"`
  - New test: across two turns of one session with no persona edits, the system
    prompt is byte-identical; a fresh memory write appears in the *user* message,
    not the system prompt.

### Stage 3 — Anthropic cache breakpoints (`system_and_3`)

- **What:** Emit `system` as a content-block list; apply the `system_and_3`
  strategy — 4 `ephemeral` breakpoints = system prompt + last 3 non-system
  messages (rolling), TTL `5m`/`1h`, skip empty/marker-incapable messages, honor
  native-Anthropic vs envelope placement. Port `agent/prompt_caching.py`. Extend
  `to_api_param` to optionally attach `cache_control` to a block.
- **Files:** `src/openharness/api/client.py:203-233` (`_stream_once`);
  `src/openharness/engine/messages.py:101` (`to_api_param` cache_control option);
  new `src/openharness/api/prompt_caching.py` (the strategy, ported).
- **Verify:**
  - `uv run pytest tests/test_api -q -k "cache or system_and_3"`
  - New test: system block marked; the last 3 marker-capable non-system messages
    marked; total breakpoints == 4; empty messages skipped.

### Stage 4 — `prompt_cache_key` on Codex + OpenAI

- **What:** Add `prompt_cache_key = <session-stable key>` to the Codex Responses
  body and the OpenAI chat body.
- **Files:** `src/openharness/api/codex_client.py:284-295` (add to `body`);
  `src/openharness/api/openai_client.py` (add to request params); thread a
  stable session key through `ApiMessageRequest` if not already present.
- **Verify:**
  - `uv run pytest tests/test_api -q -k "codex or cache_key"`
  - New test: Codex body includes a stable, non-empty `prompt_cache_key` that is
    identical across two turns of the same session and differs across sessions.

### Stage 5 — Measurement

- **What:** Log `cache_read_input_tokens` (Anthropic) / `cached_tokens` (OpenAI)
  from response usage; add a per-session cache-hit-ratio metric to the eval
  recorder.
- **Files:** `src/openharness/api/client.py` + `codex_client.py` (usage parse);
  `ohmo/gateway/runtime.py` (`GatewayEvalRecorder` resource snapshot).
- **Verify:**
  - `uv run pytest tests/test_ohmo -q -k "usage or eval_record"`
  - Manual: a ≥10-turn shadow session reports cache-read tokens > 0 and rising.

## Acceptance criteria (DoD)

1. System prompt contains no per-turn-changing data; the memory-free base is
   byte-identical across turns of a session (Stage 1–2 tests green).
2. Anthropic requests carry the `system_and_3` breakpoints — system prompt +
   last 3 non-system messages, 4 total (Stage 3 test green).
3. Codex/OpenAI requests carry a stable per-session `prompt_cache_key`
   (Stage 4 test green).
4. Usage logging surfaces provider cache counters; a ≥10-turn session shows
   `cache_read`/`cached` tokens > 0 (Stage 5).
5. Target: **≥50% input-token reduction** on sessions of ≥10 turns, measured via
   the new metric on a shadow run.
6. `uv run pytest`, `uv run ruff check src tests scripts`, and `uv run mypy` all
   pass.

## Rollout

Ship behind the existing prod deploy flow (pipx from the `deploy` branch;
`systemctl --user restart ohmo-gateway.service`). Land Stages 1–2 first (pure
prefix stability — benefits Codex auto-caching with no API-shape change),
observe the metric, then Stages 3–4 (explicit hints), then confirm the target on
a multi-turn shadow session before declaring done.

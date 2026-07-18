# Plan: h4-phase0 — OpenHarness memory file-seam (H4 Phase 0)

Source ADR: `/Users/dldmitry/agents-playgroud/adrs/honcho-openharness-backend.md` (rev 6, converged over
6 adversarial gpt-5.6-sol reviews → `CRITICAL ISSUES REMAINING: no`). This plan implements **only
rollout step 1** — the file seam — with **no honcho in the hot path** and **owner-path behaviour
byte-identical to today**. The gate *enforcement*, tool-confinement / out-of-process memory service,
and the honcho client are later phases (separate plans), gated on product approvals + infra.

Repo: `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam` (off HEAD, which has memory PRs
#52 `search` + #53 auto-reindex). Tests: `uv` venv, `pytest` (`asyncio_mode=auto`, `testpaths=["tests"]`).
Run with `.venv/bin/python -m pytest …`.

Defaults: model=gpt-5.6-sol reasoning=xhigh workdir=/Users/dldmitry/tmp/OpenHarness timeout=45m

Global constraints for EVERY step (restate in each contract — agents boot stateless):
- Do **not** `git commit`, `git push`, branch, or deploy. Code + tests only. The orchestrator handles git.
- Do **not** add honcho, HTTP, network, or SQLite code in Phase 0 — this is a pure file-backed refactor.
- The **owner-path observable behaviour must not change**: the `memory` tool outputs, the injected
  memory prompt bytes, `memory_judge` proposal semantics, write guarantees, and archive behaviour stay
  identical. The characterization suite (s1) is the regression net; keep it green in every later step.
- Use `.venv/bin/python -m pytest …` for tests. Don't weaken or delete existing assertions to pass.
- Keep new files < 600 LOC; match the surrounding code style.

---

## Step: s1-characterize — pin current memory behaviour (regression net)
Verify: cd /Users/dldmitry/tmp/OpenHarness && .venv/bin/python -m pytest tests/test_ohmo/test_memory_characterization.py -q
Contract:
Repo `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam`. This is a **test-only** step: add
`tests/test_ohmo/test_memory_characterization.py` that pins the CURRENT observable behaviour of ohmo's
curated memory so a later refactor can't change it silently. **Change no source files.**

First READ these to learn the real API: `ohmo/memory_tool.py` (the 6 `memory` actions + the
`document_search-cli` `search`), `ohmo/memory.py` (`load_ohmo_memory_prompt` + the 4000/12000 budgets +
first-200-line index + overflow), `ohmo/memory_store.py` (`MemoryStore` — `add`/`update`/`add_legacy`/
`remove`/`list`/`get`/`record_use`, `MemoryEntry`, `MemoryOpResult`, soft-archive), `ohmo/memory_judge.py`
(3-turn cadence, judge-`remove`→`removal_proposals.json`, `consolidate` auto-removes sources). Look at the
existing `tests/test_ohmo/test_memory_tool.py`, `test_memory_judge.py`, `test_memory_autoreindex.py` for
the fixture/tmpdir style and reuse it.

Write characterization tests (build a `MemoryStore` in a tmp dir, exercise real code, assert EXACT current
output — run the code first to capture the real strings, then assert them):
1. `memory` tool: `add(title,content)`, `list`, `get(name)`, `update`, `remove`, and the `search` action —
   assert the exact returned `MemoryOpResult.message` / rendered strings and error strings (e.g. get of a
   missing name).
2. Injected prompt: for a store with a few entries, assert the `load_ohmo_memory_prompt` output shape —
   the `# ohmo Memory` scaffold, the fenced `MEMORY.md` index (first 200 lines), usage-ranked bodies,
   ≤4000-char body cap, the 12000-char budget cutoff (first body always injected), and the overflow note.
3. Write guarantees: `add` exact-dedup + per-entry(4000) + store(24000) budget + threat-scan; `update`
   budget+scan but **no** exact dedup; `add_legacy` skips dedup + whole-store budget. Assert each.
4. `memory_judge`: a judge op named `remove` becomes a `removal_proposals.json` proposal (not an immediate
   delete); a `consolidate` removes its source entries; foreground `store.remove` soft-archives to
   `archive/`. Assert these.

DoD: `tests/test_ohmo/test_memory_characterization.py` passes on the CURRENT code and covers all four
areas. No source file changed (verify `git status` shows only the new test file). Run the Verify command
yourself and confirm green before declaring done.

---

## Step: s2-backend — async MemoryBackend Protocol + FileMemoryBackend
Verify: cd /Users/dldmitry/tmp/OpenHarness && .venv/bin/python -m pytest tests/test_ohmo/test_memory_backend_file.py tests/test_ohmo/test_memory_characterization.py -q
Contract:
Repo `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam`. A characterization suite already
exists at `tests/test_ohmo/test_memory_characterization.py` (keep it green). Add the seam abstraction —
**no call-site changes yet**.

Create `ohmo/memory_backend.py`:
- A `MemoryBackend` `typing.Protocol` with **async** methods mirroring today's surface over
  storage-neutral DTOs (reuse `MemoryEntry`/`MemoryOpResult` from `ohmo/memory_store.py`; do NOT put
  `pathlib.Path` in the Protocol signatures): `async def list(...)`, `get(name)`, `search(query, top_k)`
  → a `list[MemoryHit]` (define a small `MemoryHit` dataclass `{name, title, snippet, rank}` — **no score
  field**, matching the ADR D3), `add(title, content)`, `update(name, content)`, `remove(name)`,
  `render_prompt(budget) -> str`, and `append_turn(role, text) -> None`.
- `FileMemoryBackend(MemoryBackend)` that wraps the existing synchronous file implementation: delegate to
  `MemoryStore` for CRUD/list/get, to `ohmo/memory_tool`'s existing `document_search-cli` search path for
  `search`, and to `ohmo/memory.load_ohmo_memory_prompt` for `render_prompt`. `append_turn` is a **no-op**
  on the file backend. Wrap the sync calls so the async methods don't block (e.g. `asyncio.to_thread`), but
  keep semantics identical.

READ `ohmo/memory_store.py`, `ohmo/memory_tool.py`, `ohmo/memory.py` first for the exact signatures.

Add `tests/test_ohmo/test_memory_backend_file.py` (async tests, `asyncio_mode=auto`): assert
`FileMemoryBackend` CRUD/list/get/search/render_prompt produce the SAME results as calling the underlying
`MemoryStore`/`load_ohmo_memory_prompt` directly, and `append_turn` is a no-op.

DoD: `ohmo/memory_backend.py` + `tests/test_ohmo/test_memory_backend_file.py` added; **no other source
file changed** (the Protocol is not yet wired anywhere). Both the new test file and the characterization
suite pass. Run the Verify command yourself and confirm green.

---

## Step: s3-seam — 3-step prompt seam + memory-free persona
Verify: cd /Users/dldmitry/tmp/OpenHarness && .venv/bin/python -m pytest tests/test_prompts tests/test_ohmo/test_memory_characterization.py tests/test_ohmo/test_prompt_seam.py -q
Contract:
Repo `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam`. `ohmo/memory_backend.py`
(FileMemoryBackend) and the characterization suite exist. Implement the exact prompt seam from the ADR §8.

Today `build_ohmo_system_prompt` (`ohmo/prompts.py`, around the `load_ohmo_memory_prompt` call ~line
199-200) unconditionally appends the memory block, and it is frozen into `settings.system_prompt`, then
recomposed each turn via `_runtime_system_prompt` in `ohmo/gateway/runtime.py` (~1017-1031) WITHOUT
re-reading memory. Change to the three-step seam:
1. Add a keyword param `include_ohmo_memory: bool = True` to `build_ohmo_system_prompt`; when `False`, build
   the persona **without** the memory block AND without the workspace-root / "memory lives here" lines
   (`ohmo/prompts.py` ~168-175). Store this memory-free base **separately** from `settings.system_prompt`
   so `current_settings()` (`src/openharness/ui/runtime.py` ~141-150) stops reapplying a composed memory
   override.
2. Add `async def prepare_turn(backend, turn_ctx) -> str` (the memory snapshot) — on the file backend it
   calls `FileMemoryBackend.render_prompt(...)`; and a pure `compose_runtime_prompt(memory_free_base,
   snapshot) -> str` that appends **exactly one** memory block (guard against the old-file block + the new
   snapshot both being present).
3. Call `prepare_turn` before every submitted turn (ordinary submit, command submit, continuation,
   `_refresh_bundle`) so recall is read-your-writes rather than the frozen block.

**The owner-path final prompt bytes must be identical to today** — the characterization suite + existing
`tests/test_prompts` enforce this; if they change, you changed behaviour and must fix the seam, not the
tests.

Add `tests/test_ohmo/test_prompt_seam.py`: (a) `include_ohmo_memory=False` yields a persona with no memory
block and no workspace-path lines; (b) `compose_runtime_prompt` injects exactly one block; (c) an `add`
followed by a fresh `prepare_turn` shows the new entry (read-your-writes), whereas the frozen
`settings.system_prompt` would not.

DoD: the three changes landed; `tests/test_prompts`, the characterization suite, and the new
`test_prompt_seam.py` all pass. Run the Verify command yourself and confirm green.

---

## Step: s4-swap — make_memory_backend factory + route swap sites
Verify: cd /Users/dldmitry/tmp/OpenHarness && .venv/bin/python -m pytest tests/test_ohmo tests/test_prompts tests/test_commands -q
Contract:
Repo `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam`. The `MemoryBackend`/
`FileMemoryBackend` (s2) and the prompt seam (s3) exist. Wire the seam through the real call sites, still
**file-default, no behaviour change**.

- Add a `memory_backend: str = "file"` field to `GatewayConfig` (`ohmo/gateway/models.py`) plus optional
  placeholders `honcho_base_url/honcho_api_key/honcho_workspace` (unused in Phase 0). `load_gateway_config`
  (`ohmo/gateway/config.py`) parses them if present; default `file`.
- Add `make_memory_backend(cfg, workspace) -> MemoryBackend` returning `FileMemoryBackend` for
  `memory_backend == "file"` (raise `NotImplementedError("honcho backend not built in Phase 0")` for
  `"honcho"`).
- Route the swap sites through the factory / backend instead of constructing `MemoryStore` directly for
  the model-facing paths: `ohmo/gateway/runtime.py` (the memory-tool wiring + the `prepare_turn` call),
  `ohmo/memory.py` helpers, `ohmo/memory_tool.py`. Keep the file `MemoryStore` as the FileMemoryBackend's
  internal — don't delete it.
- Gate `memory_judge` scheduling/execution to the **file backend only** (skip when
  `cfg.memory_backend != "file"`).

READ `ohmo/gateway/runtime.py`, `ohmo/gateway/models.py`, `ohmo/gateway/config.py`, `ohmo/memory.py`,
`ohmo/memory_tool.py`, `ohmo/memory_judge.py` first. Do not change honcho anything; there is none yet.

DoD: factory + config field added; the model-facing memory paths go through `make_memory_backend`;
`memory_judge` is file-gated; the **whole** `tests/test_ohmo tests/test_prompts tests/test_commands`
suite passes unchanged. Run the Verify command yourself and confirm green.

---

## Step: s5-command-adapter — file-only /memory + /dream adapter + honcho-disabled dispatch
Verify: cd /Users/dldmitry/tmp/OpenHarness && .venv/bin/python -m pytest tests/test_commands tests/test_ohmo/test_command_backend_kind.py -q
Contract:
Repo `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam`. Keep `/memory` and `/dream` behaving
**exactly as today on the file backend**, and make the dispatch backend-aware so a future honcho backend is
cleanly refused (the honcho backend itself is NOT built in Phase 0).

- READ `src/openharness/commands/registry.py` (`MemoryCommandBackend` — the `Path`-returning callbacks,
  `/memory show|add|remove|edit|migrate`, `/dream`) and `tests/test_commands/test_registry.py` +
  `test_command_flows.py`.
- Keep the filesystem `MemoryCommandBackend` as the **file-only** adapter, unchanged for
  `memory_backend == "file"`.
- Add a backend-kind guard: when `cfg.memory_backend == "honcho"`, `/memory` (show/add/remove/edit/migrate)
  and `/dream` return a clear error `"not supported on the honcho backend"` instead of touching the
  filesystem. Since no honcho backend exists yet, this is a dispatch-level guard you can unit-test by
  passing the kind.
- Add `tests/test_ohmo/test_command_backend_kind.py`: with kind `file`, the commands behave as today
  (reuse an existing flow); with kind `honcho`, they return the not-supported error and perform no
  filesystem write.

DoD: file-backend command behaviour unchanged (`tests/test_commands` green); the new kind-guard test
passes. Run the Verify command yourself and confirm green.

---

## Step: s6-turncontext — TurnContext + gateway canonical-principal identity (plumbing only)
Verify: cd /Users/dldmitry/tmp/OpenHarness && .venv/bin/python -m pytest tests/test_ohmo -q
Contract:
Repo `/Users/dldmitry/tmp/OpenHarness`, branch `h4/phase0-file-seam`. Land the gateway-identity
**plumbing** the later confidentiality gate needs, **without yet gating memory off for non-owners** (that
enforcement is a Phase-1 named behaviour change — do NOT change who currently gets memory).

- Add a `TurnContext` dataclass (`ohmo/gateway/` — new module or into `router.py`):
  `{principal: str, is_owner: bool, is_private: bool, channel: str, chat_id: str, session_id: str}`.
- Add a per-channel **canonical-principal function**: for Telegram, derive the principal from the
  **immutable numeric id** portion of `sender_id` (which is `"<numeric-id>|<mutable-username>"` — see
  `src/openharness/channels/impl/telegram.py` ~735-739), never the username.
- Add owner principals + config to `GatewayConfig` (a list of canonical owner principals); compute
  `is_owner` (principal ∈ owners) and `is_private` (channel reports a non-group/private chat; treat
  absent/unknown group metadata as **not** private) in the gateway, and thread a `TurnContext` into the
  `prepare_turn` call and the backend ops (they may ignore it in Phase 0).
- Do **NOT** change memory injection behaviour based on the gate yet — only compute + plumb + optionally
  log. The characterization suite must stay green.

READ `ohmo/gateway/router.py`, `ohmo/gateway/runtime.py`, `ohmo/gateway/models.py`,
`src/openharness/channels/impl/telegram.py` first.

Add `tests/test_ohmo/test_turncontext.py`: canonical principal is the numeric id (stable across a username
change); `is_owner` true only for configured owners; `is_private` false for group / forwarded / unknown
metadata.

DoD: `TurnContext` + canonical-principal fn + owner config + gateway computation landed and threaded into
`prepare_turn`; memory behaviour for the owner unchanged; `tests/test_ohmo` (incl. characterization +
the new `test_turncontext.py`) passes. Run the Verify command yourself and confirm green.

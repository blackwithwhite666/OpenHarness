# Plan: h4-phase4 — прокачка / eval-lane (memory recall vs file-baseline)

**Status: DESIGN ONLY — not executed.** The quality gate **between** Phase 1 (off-path stack) and
Phase 2 (user-visible recall). Source ADR: `/Users/dldmitry/agents-playgroud/adrs/honcho-openharness-backend.md`
(rev 6). This phase ships **nothing** to the owner; it proves — and iterates — that honcho/catalog recall
**beats the file baseline end-to-end**, which is the precondition that unlocks Phase 2. Shadow mode
(Phase 1 step 7) gives an *offline* comparison; Phase 4 gives the *end-to-end* quality signal.

Deploy target for live sweeps: the 93.77 evals flywheel against `https://memory.worfalomey.top/`.
Repo `/Users/dldmitry/tmp/OpenHarness`. Tests: `.venv/bin/python -m pytest`.
Defaults: model=gpt-5.6-sol reasoning=xhigh workdir=/Users/dldmitry/tmp/OpenHarness timeout=45m

## What Phase 4 delivers (and what it does NOT)
- ✅ per-case **memory-backend provisioning** in the eval harness (isolated `ohmo-eval-<run>-<case>-<sample>`
  workspaces via the eval-only admin credential from `p1-honcho-client`), a **memory-specific multi-turn
  benchmark**, a **recall dimension** in the meta/grounding judge, and an **A/B report** (file vs catalog vs
  shadow-honcho) with a machine-readable **go/no-go gate**.
- ⛔ **Not**: any user-visible change; any prod default change; the **tuning decisions** themselves — those
  are orchestrator-run sweeps (ranking weights, bootstrap knobs), never an agent contract.

## Global constraints (restate per step)
- Agents write **eval code + tests + fixtures ONLY**. The orchestrator runs the real sweeps on the 93.77
  flywheel + against `memory.worfalomey.top` and **owns the tuning decisions** (catalog↔honcho blend
  weights, `observe_others`, deriver batching, prompt-seam budget). Never let an agent commit/deploy/tune-live.
- The **eval-only admin/provisioning credential** lives ONLY in the eval harness, **never** in the gateway.
- Isolated per-case honcho workspace + fresh catalog per sample; deterministic teardown.
- Match the existing ohmo eval style; new files < 600 LOC; `asyncio_mode=auto`.

---

## Step: p4-eval-provisioning — per-case memory-backend provisioning
**What to read first:** locate ohmo's **faithful multi-turn eval lane** (search the repo for the
fs-sandbox jail-per-turn harness + the Claude-sim driver + the intent/grounding judge) and how it builds
per-case fixtures; the eval-only admin credential + `ohmo-eval-<run>-<case>-<sample>` workspace bootstrap
from `p1-honcho-client` (`ohmo/memory_service/bootstrap.py`).
**What:** an eval fixture that, per case/sample, provisions an isolated memory backend selected by
`--memory-backend {file,catalog,shadow}`: `file` → temp workspace dir; `catalog` → fresh SQLite catalog;
`shadow` → eval-only admin credential mints an `ohmo-eval-<run>-<case>-<sample>` workspace + short-lived
token. Deterministic teardown after each case.
**Files:** the eval provisioning module + `tests/test_ohmo/test_eval_provisioning.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_eval_provisioning.py -q`
(two cases get **isolated** backends — no cross-contamination; teardown verified against a mocked admin client).

## Step: p4-memory-benchmark — multi-turn memory benchmark cases
**What:** a jsonl benchmark of multi-turn cases that specifically exercise memory: (a) **write-early /
recall-late** (turn 1 "запомни X" → turn N "что я просил?"); (b) **update** ("теперь Y") → recall reflects
Y not X; (c) **remove** → recall must NOT surface it; (d) **distractors**; (e) **negative / no-fabrication**
cases. Each case carries an `expected_recall` assertion + a `must_not_recall` set for the judge. A loader
feeds them into the multi-turn lane.
**Files:** the benchmark data (jsonl) + loader + `tests/test_ohmo/test_memory_benchmark.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_memory_benchmark.py -q`
(schema valid; loader integrates with the multi-turn lane fixture from `p4-eval-provisioning`).

## Step: p4-recall-judge — memory-recall dimension in the judge
**What to read first:** the verification/grounding meta-judge ohmo already uses (search: grounding judge /
verify-votes).
**What:** extend the judge to score per case: (1) the target fact was **recalled** into context; (2) the
agent **used** it in the answer; (3) **grounded / no fabrication** (`must_not_recall` respected); (4) the
**delta vs baseline**. Emit a structured `recall_score`.
**Files:** the judge extension + `tests/test_ohmo/test_recall_judge.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_recall_judge.py -q`
(a known-good and a known-bad transcript score correctly; a fabricated recall fails grounding).

## Step: p4-ab-report — file vs catalog vs shadow-honcho comparison + gate
**What:** a report generator that runs the memory benchmark across backends, aggregates `recall_score` +
general-quality (no-regression) + latency, and emits (a) a human report and (b) a **machine-readable gate
result**: PASS iff catalog/honcho ≥ file on recall AND no general-quality regression. **This artifact is
the Phase-2 go/no-go.**
**Files:** the report generator + `tests/test_ohmo/test_ab_report.py`.
**Verify:** `.venv/bin/python -m pytest tests/test_ohmo/test_ab_report.py -q`
(aggregation + gate logic correct on fixture runs: recall-win ⇒ PASS, regression ⇒ FAIL).

---

## Orchestrator (out-of-contract) steps
- Run the real benchmark sweep on the 93.77 flywheel against `memory.worfalomey.top`; **iterate** the
  ranking weights (catalog↔honcho blend), bootstrap `observe_others`, deriver batching, and prompt-seam
  budget — the actual прокачка — until the `p4-ab` gate is **PASS**.

## Exit gate (unlocks Phase 2)
`p4-ab` report **PASS**: honcho/catalog recall beats the file baseline with no general-quality regression.
Phase 4 then stays on as the **regression engine** — every Phase 2/3 step re-passes this benchmark before merge.

# H4 roadmap — honcho-backed pluggable memory for ohmo (master index)

Master index over the phase plans. Source ADR:
`/Users/dldmitry/agents-playgroud/adrs/honcho-openharness-backend.md` (rev 6, converged over 6 adversarial
gpt-5.6-sol rounds → `CRITICAL ISSUES REMAINING: no`). Repo `/Users/dldmitry/tmp/OpenHarness`.

**Core split (why there are two stores):** the **local transactional SQLite catalog** is the single
**authority** for *curated* memory ("запомни X" — deterministic, read-your-writes, addressable, transactional
budgets/dedup/threat-scan, never auto-deleted). **honcho** is the **derived-recall** layer ("что я знаю о
владельце из разговоров") + an at-least-once **mirror** — both strictly **off the correctness path**, with
**no** file fallback. honcho's model (async deriver, no read-your-writes, no stable name, autonomous dreamer
deletion, no score) structurally cannot be the authority for curated writes; it is excellent at the derived
recall the local store cannot do.

## Phases & plans
| Phase | Plan file | What | Status |
|-------|-----------|------|--------|
| **0** | `vibe/h4-phase0-plan.md` | pluggable-backend **seam** (async `MemoryBackend`, prompt seam, factory/config, command guard, `TurnContext`) — no honcho in the hot path | ✅ **merged** (PR #55) |
| **1** | `vibe/h4-phase1-plan.md` | honcho **off the correctness path**: catalog + `CatalogMemoryBackend` + migration + out-of-process memory service (separate UID) + honcho client + outbox + **shadow mode** | 📝 drafted |
| **4** | `vibe/h4-phase4-eval-plan.md` | **прокачка / eval-lane**: provision per-case backends, memory multi-turn benchmark, recall judge, A/B report — prove & iterate recall beats file baseline | 📝 drafted |
| **2** | `vibe/h4-phase2-plan.md` | **user-visible** recall: OS-attested tool sandbox + fail-closed gate enforcement + opt-in derived recall + default file→catalog + conversation-learning/dreamer + multi-sender | 📝 drafted |
| **3** | `vibe/h4-phase3-plan.md` | **terminal** (D6-A): reconciliation-at-scale + DR export + honcho owns identity + file-backend decommission | 📝 drafted |

## Dependency graph (note: Phase 4 gates Phase 2 — it is NOT after Phase 3)
```
Phase 0 ✅ seam
   │
   ▼
Phase 1  honcho off-path  ──────────►  ends with shadow-mode OFFLINE comparison data
   │
   ▼
Phase 4  ПРОКАЧКА (eval-lane)  ──────►  GATE: recall beats file baseline, no general regression
   │                                    └─ stays on afterwards as the regression engine
   ▼
Phase 2  user-visible recall  ───────►  GATE: OS-attested sandbox + fail-closed gate BEFORE any recall
   │
   ▼
Phase 3  terminal switch (D6-A)  ────►  GATE: reconciliation zero-orphan + proven DR export
```

## Three hard stops (never cross without the gate green)
1. **Nothing user-visible until Phase 4 PASS** — honcho recall must beat the file baseline end-to-end before
   it is ever surfaced (Phase 4 → Phase 2).
2. **No recall reaches the owner without the OS-attested tool sandbox + fail-closed gate** — `tools_confined`
   is the hard boundary; the four-conjunct gate is fail-closed (Phase 2 `p2-tool-sandbox` + `p2-gate-enforcement`).
3. **No honcho-owns-identity switch without reconciliation-at-scale + a proven DR export** — honcho is not a
   backup (Phase 3 `p3-reconciliation-at-scale` + `p3-dr-export`).

## Standing constraints (every phase)
- Once the memory service exists (Phase 1 step ≥4), the honcho **JWT + catalog** are **NEVER** constructed
  in the gateway process — they live only in the separate-UID service, reachable via the gateway-authorized RPC.
- todovan agents do **code + tests only**; the **orchestrator** does all git (commit/push/PR/merge), deploy,
  live checks, and the Phase-4 **tuning decisions**. Never delegate a merge/deploy/live-tune to an agent.
- The **catalog stays the authority** and honcho stays **off the correctness path** through Phases 1–2; only
  Phase 3 makes honcho load-bearing (for identity), and only behind reconciliation + DR.

## Execution order (via todovan, one step = one verify-gated agent)
Phase 1 (Track 1a `p1-catalog → p1-catalog-backend → p1-migration`; Track 1b `p1-memory-service →
p1-honcho-client → p1-outbox → p1-shadow`) → orchestrator live shadow dry-run → Phase 4
(`p4-eval-provisioning → p4-memory-benchmark → p4-recall-judge → p4-ab-report`) → orchestrator sweep until
PASS → Phase 2 (`p2-tool-sandbox → p2-gate-enforcement → p2-visible-recall → p2-default-catalog →
p2-conversation-learning → p2-multi-sender`) → Phase 3 (`p3-reconciliation-at-scale → p3-dr-export →
p3-honcho-identity → p3-decommission-file`).

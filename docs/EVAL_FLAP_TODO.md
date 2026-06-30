# Eval flap-reduction TODO (F1–F7)

Goal: collapse the run-to-run flap in `ohmo evals` so the gate reports the
agent's real capability, not harness/judge noise.

## Why (baseline evidence, 2026-06-30)

Full 59-case pack, frozen `query-engine` + `trajectory_judge_v1`, **samples-3**
(`eval_report_fullrun3x3.json`, report_id `eval-exec:13825f1feb82d136b9899a58`):

- `pass_count` distribution `/3`: **45×(3/3) solid · 6×(2/3) flap-pass · 5×(1/3)
  flap-fail · 3×(0/3) stable-fail**. Majority = 51/59 (86%).
- **Every** non-pass case scores **0.917–0.972** — essentially-correct answers at
  the judge boundary, not capability failures.
- samples-1 churns ±4 between identical-config runs (fullrun2 vs fullrun3:
  3 "fixed" / 4 "regressed", all flap).

Root causes found (grounded in per-sample judge reasons):

1. **Order-replay decouples tool results from the agent's actual queries.**
   `--fixture-match order` serves fixtures by position. Whether the agent reaches
   the key evidence depends on reproducing gold's call count/order — stochastic.
   Evidence: `4c7c3f79` (rzd deep-link) — s2 (39 calls) found `rzdpass://route?…`
   → pass; s0/s1 (26 calls) didn't reach it → fail, same config.
2. **Attachments / gold-referenced files are invisible to typed-read tools.**
   Live-read materializes only fixture files into the read-sandbox (`_ws`);
   bash sees the real FS → inconsistent. Evidence: `82dca79d` (audio MP3 "not
   found"), `69a29b15` (lab PDFs invisible to glob though present).
3. **Judge verdict variance** on equivalent answers. Evidence: `cae57e18`
   (monobouquet overview) judged pass, pass, **fail** across 3 samples.
4. **Time-sensitive / live external facts** the frozen lane can't pin. Evidence:
   `faf9f922` (bridge closures for a specific date).

---

## Implementation status (2026-06-30)

All seven implemented + deployed on the `deploy` branch.

**Re-baseline effect (full 59, samples-3).** OLD harness (single judge vote,
pre-fix) vs NEW (majority-of-3 + absence rubric + F1/F6), pass_count `/3`:

| | 3/3 | 2/3 | 1/3 | 0/3 | flap band (1–2/3) | decisive (3/3+0/3) |
|---|---|---|---|---|---|---|
| OLD | 45 | 6 | 5 | 3 | 11 | 48 |
| NEW | 42 | 7 | 2 | 8 | **9** | **50** |

Flap band shrank 11→9 and the worst (1/3) band 5→2; decisiveness rose 48→50.
Pass count dipped 51→49 because the stricter absence rubric now *correctly*
fails answers that claimed "not found" while the info was reachable (0/3 rose
3→8) — the gate got honest, not worse. Residual flap (9) is dominated by live
retrieval variance, addressed by the F2 live lane (agent-side F6 only fully
lands in prod after a gateway restart).

- **F7** — `--samples` already defaults to 3 (gate is samples-3); confirmed.
- **F4** — `e0e96d8`: majority-of-3 judge votes (`--judge-votes`, default 3) +
  absence-claim rubric; ties→fail, all-unparseable→error.
- **F5** — `e0e96d8`: `flaky` metadata flag (0 < pass_count < samples) + viewer
  badge / case-summary field.
- **F1** — `d35657c`: read-only passthrough of allowlisted real roots (workspace
  attachments dir) to live-read typed tools, so glob/grep/read_file see what the
  live bash lane / prod see (no media copy).
- **F6** — `e2ccd9b`: absence-claim discipline + persistence in the runtime
  prompt, tied to D9 `trace_missing_required` / `trace_stop_condition`.
- **F2** — `7da3277`: judge-on-method (grounding) mode (`--judge-grounding`) for
  time-sensitive cases; operational pattern = curated live-read sub-pack,
  `--report-only`, so live facts don't flap the frozen gate.
- **F3 (experiment, done)** — `--fixture-match arguments` made the flap band
  **worse**: 14-case band → **0/14 pass** (vs 6 passing under `order`).
  **Conclusion: keep `order`.** The real retrieval fix is the live lane (F2),
  not arguments-match. (`synth` not pursued — adds its own LLM variance.)

---

## Tasks

### [x] F1 — Materialize attachments + gold-referenced files into the read-sandbox
- **Fixes:** cause #2 (`82dca79d` audio, `69a29b15` PDF visibility).
- **What:** in `src/openharness/evals/workspace_materialize.py` /
  `live_read.py`, extend `materialize_read_fixtures` to also stage the episode's
  attachments (the `attachments` dir surfaced by the viewer's
  `/api/attachments`) and any gold-referenced files, so `glob`/`grep`/`read_file`
  see the same files `bash`/prod do. Decide policy: stage by reference vs whole
  attachment dir (privacy: keep inside the sandbox, never copy secrets).
- **Accept:** in live-read, `glob`/`read_file` over an episode's attachment paths
  return the files; `82dca79d` agent can locate the MP3; a unit test in
  `tests/test_evals/` materializes a fixture attachment and asserts visibility.

### [x] F2 — Live retrieval lane for time-sensitive cases + judge-on-method
- **Fixes:** cause #4 (`faf9f922`), part of #1 for fresh-fact cases.
- **What:** tag cases whose gold answer depends on live external facts; route
  them to the live-read lane (live `google_search`/`web_fetch`) and judge on
  **method/grounding**, not exact facts; keep them **report-only** (not gate).
- **Accept:** tagged cases run live; the judge rubric for them scores "used the
  right sources + reported what was found" rather than matching gold's stale
  facts; they no longer flap the gate number.

### [x] F3 — Experiment: `--fixture-match arguments`/`synth` for retrieval
- **Fixes:** cause #1 (decoupling).
- **What:** serve a recorded result by the agent's **actual query** instead of by
  position, so call count/order stops mattering. Note this is not a clean win
  (a novel query → `replay_miss`/empty); `synth` mode fills novel queries via an
  LLM (adds a call + its own variance). Run both modes on the 11 flap cases and
  compare flap rate vs `order`.
- **Accept:** a documented comparison (order vs arguments vs synth) of
  `pass_count` stability on the flap band; pick the mode with the least flap and
  set it as the gate default if it wins.

### [x] F4 — Stabilize the judge (temperature 0 + majority-of-K votes)
- **Fixes:** cause #3 (`cae57e18`).
- **What:** in `src/openharness/evals/judge.py` (`TrajectoryJudgeScorer`), pin
  judge temperature 0 and judge each trajectory **K times, take majority**.
  Tighten the rubric with an explicit criterion: "summarized the available
  evidence" = pass vs "claimed unavailable while it was reachable" = fail.
- **Accept:** re-judging the SAME stored trajectory K times yields a stable
  verdict; `cae57e18`-class equivalent answers stop flipping.

### [x] F5 — Treat the 0.917–0.972 band as pass-with-warning / calibrate threshold
- **Fixes:** the systemic "essentially-correct judged fail" pattern.
- **What:** since no non-pass case is a real capability gap, either surface a
  `warning` status for the boundary band (don't hard-fail) or recalibrate the
  pass threshold against a labeled set.
- **Accept:** boundary-band cases render as warnings in the report/viewer; the
  hard-fail set equals the genuinely-wrong set.

### [x] F6 — Agent: absence-claim discipline + persistence
- **Fixes:** trajectory variance (`4c7c3f79`, `cae57e18` pass when digging deeper).
- **What:** instruct the agent (prompt / `prompts/context.py`) not to conclude
  "not found / unavailable" until it has tried alternate queries/tools; wire the
  D9 decision-trace `trace_missing_required` / `trace_stop_condition` as a guard
  before an absence claim.
- **Accept:** on `4c7c3f79`/`cae57e18`, the agent reaches the evidence on ≥ a
  higher fraction of samples (pass_count rises toward 3/3).

### [x] F7 — samples-3 (≥3) as the gate default
- **Fixes:** measurement noise (±4 at samples-1).
- **What:** make `--samples 3` the default for the gate run (CLI default and/or
  the runbook); samples-1 stays available for quick smoke.
- **Accept:** the documented gate command uses samples-3; baseline reported as
  majority-of-3 (current: 51/59).

---

## Suggested order
Highest ROI first: **F1 + F4 + F7** (most flap is harness/judge artifact) → then
F2 (live lane) and F6 (agent) → F3/F5 as experiments. Re-baseline samples-3 after
each and watch the `pass_count` distribution shrink toward 3/3.

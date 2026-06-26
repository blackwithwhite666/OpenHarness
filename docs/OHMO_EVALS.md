# Ohmo eval data flywheel

Ohmo evals keep private conversation text in the local eval store, but write
mined cases, review manifests, run packs, and reports as metadata-only
artifacts.

## Storage contract

The current source of truth is the workspace-local JSON/JSONL store:

- episodes and event streams live under `evals/episodes/`
- resource snapshots live under `evals/states/`
- candidates, draft cases, gold cases, packs, and reports are written as
  explicit JSON/JSONL artifacts with manifests

`EvalStore` keeps a small local `evals.sqlite` lookup index for JSONL offsets
and embedding metadata, but that database is not the source of truth. Do not
add a broader SQLite projection/cache for candidates, review, packs, reports,
or comparisons until there is a concrete performance or concurrency need.

Triggers for adding a broader SQLite projection:

- mining or pack building becomes noticeably slow for a normal workspace
- the store grows to tens or hundreds of thousands of events/facets
- review needs fast filters over tool name, case kind, review status, or
  embedding/content hashes
- multiple gateway/eval processes need coordinated concurrent writes
- embedding lookup cost starts to dominate and content-hash caching is needed

Until one of those triggers is observed and measured, keep SQLite limited to
rebuildable lookup rows. New workflow state should be written first as
JSON/JSONL artifacts with explicit manifests, then optionally indexed from
those artifacts. A missing or empty `evals.sqlite` file must be recoverable from
the JSON/JSONL source of truth.

## Flow

1. Capture gateway episodes while using Ohmo. Captured state lives under
   `<workspace>/evals/episodes/` and `<workspace>/evals/states/`.
2. Build facets, graph signals, embeddings, candidates, and draft cases:

   ```bash
   ohmo evals embed --workspace <workspace>
   ohmo evals mine --workspace <workspace>
   ```

3. Review draft cases:

   ```bash
   ohmo evals cases list --workspace <workspace> --limit 20
   ohmo evals cases show --workspace <workspace> <case-id>
   ohmo evals review --workspace <workspace> --limit 20
   ohmo evals review --workspace <workspace> --manifest review_manifest.json
   ohmo evals review --workspace <workspace> --validate-manifest review_manifest.json
   ```

   `ohmo evals cases list/show` are metadata-only inspection helpers for
   draft cases. They print case ids, kinds, episode ids, facet counts, and tool
   names, not raw prompts or tool outputs.

   Review manifests are written under `evals/cases/` and contain case ids,
   facet counts, tool names, and review metadata. Edit each item with
   `decision: "approved"` or `decision: "rejected"` before batch promotion.
   They do not contain raw prompts, tool inputs, tool outputs, or final
   answers. `--validate-manifest` checks the manifest shape, decision values,
   duplicate ids, and whether referenced drafts still exist.

4. Promote reviewed drafts into the gold set:

   ```bash
   ohmo evals promote --workspace <workspace> --case-id <case-id> --reviewer <id>
   ohmo evals promote --workspace <workspace> --all --reviewer <id>
   ohmo evals promote --workspace <workspace> --manifest review_manifest.json --reviewer <id>
   ```

   Manifest promotion only promotes items marked `approved`. Rejected and
   pending items stay as drafts. Review comments are not copied as raw text into
   gold cases; gold metadata stores only comment hash and length.

5. Build and smoke-check a runnable pack:

   ```bash
   ohmo evals pack --workspace <workspace>
   ohmo evals pack --workspace <workspace> --output candidate_pack.json
   ohmo evals smoke --workspace <workspace> --pack eval_pack.json
   ohmo evals smoke --workspace <workspace> --pack candidate_pack.json --output candidate_smoke.json
   ```

   The default runnable pack is `evals/packs/eval_pack.json`.

6. Run evals:

   ```bash
   ohmo evals run --workspace <workspace> --executor replay-tools
   ohmo evals run --workspace <workspace> --pack candidate_pack.json --output candidate_eval.json
   ohmo evals run --workspace <workspace> --executor replay-tools --check-config
   ohmo evals run --workspace <workspace> --executor replay-tools --agent-runner query-engine --model <model>
   ```

   The default executor is `replay-tools`. It builds a replay-only tool
   registry from captured fixtures and never calls live tools. The default
   offline `scripted` runner replays the captured tool path deterministically.
   Use `--agent-runner query-engine` to run the reconstructed prompt through
   the normal `QueryEngine` model loop while tool calls are still served by
   recorded outputs only. The default report is `evals/reports/eval_report.json`.
   Use `--check-config` before a real run to validate the pack, executor,
   selected case count, and query-engine auth/profile without executing model
   turns or writing a report.

7. Compare execution reports before promoting a candidate run:

   ```bash
   ohmo evals baseline save --workspace <workspace> --from-report eval_report.json --name main
   ohmo evals baseline list --workspace <workspace>
   ohmo evals compare --workspace <workspace> --baseline baseline_eval_report.json --candidate eval_report.json
   ohmo evals compare --workspace <workspace> --baseline baselines/main.json --candidate eval_report.json
   ```

   Named baselines are execution reports saved under
   `evals/reports/baselines/<name>.json`. Relative compare paths are resolved
   under `evals/reports/`. The compare command writes
   `evals/reports/eval_compare.json` and exits non-zero when a baseline case
   regresses or disappears from the candidate report. Use `--report-only` to
   collect the comparison without failing the command.

## Gating vs realism: which config to run

Two lanes with different jobs — do not conflate them.

**Gating lane (the score you track and regression-gate on): frozen replay + majority vote.**

```bash
ohmo evals run --agent-runner query-engine --fixture-match order \
  --scorer trajectory_judge_v1 --samples 3
```

- `--fixture-match order` serves every tool from recorded fixtures (identical
  bytes every run). This removes the dominant noise source: live tools almost
  never *error*, but they return **different content** each run, and that
  variance flips verdicts. Freezing the tools makes most flappers fully stable.
- `--samples N` (now **default 3**) majority-votes the residual model sampling —
  the cases that still coin-flip even with frozen tools. Pass `--samples 1` only
  when you deliberately want raw single-run variance.
- This is the reproducible number for `baseline save` / `compare`.

**Realism lane (a periodic probe, NOT the gate): live tools in a sandbox.**

```bash
ohmo evals run --agent-runner fs-sandbox --sandbox-net-mode netns:<ns> \
  --sandbox-proxy-url <proxy> --sandbox-browser-socket <sock> --sandbox-browser-name <name>
```

- Runs `google_search` / maps / browser LIVE through a network-namespaced
  sandbox. More realistic, but live-content variance makes the score wander
  run-to-run. Use it to catch breakage that only live tools expose — not as a
  number you gate on.

Empirical basis (2026-06-24): of 5 cases that flapped under the live lane, **4
went rock-stable (5/5) under frozen replay**; the 1 that still flapped (3/5) was
majority-voted to a stable verdict by `--samples 3`. So **frozen replay is the
bigger stability lever; `--samples N` mops up the residual model sampling.** A
too-small `--max-turns` is a separate, deterministic failure (truncation), now
recorded in report metadata; on `MaxTurnsExceeded` the runner keeps the last
non-empty partial answer instead of blanking it.

## Baseline promotion runbook

Use this sequence for a local candidate run before treating it as the new
baseline:

```bash
ohmo evals embed --workspace <workspace>
ohmo evals mine --workspace <workspace>
ohmo evals review --workspace <workspace> --manifest review_manifest.json
# edit evals/cases/review_manifest.json and approve the cases worth keeping
ohmo evals review --workspace <workspace> --validate-manifest review_manifest.json
ohmo evals promote --workspace <workspace> --manifest review_manifest.json --reviewer <id>
ohmo evals pack --workspace <workspace>
ohmo evals smoke --workspace <workspace>
ohmo evals run --workspace <workspace> --check-config
ohmo evals run --workspace <workspace>
ohmo evals compare --workspace <workspace> --baseline baselines/main.json --candidate eval_report.json
ohmo evals baseline save --workspace <workspace> --from-report eval_report.json --name main --overwrite
```

Expected default artifacts:

- `evals/embeddings/embedding_manifest.json`
- `evals/candidates/candidate_manifest.json`
- `evals/cases/review_manifest.json`
- `evals/cases/gold_cases.jsonl`
- `evals/packs/eval_pack.json`
- `evals/reports/smoke_report.json`
- `evals/reports/eval_report.json`
- `evals/reports/eval_compare.json`
- `evals/reports/baselines/main.json`

`smoke`, `run`, and `compare` exit non-zero when they find failures or
regressions. Add `--report-only` when CI or local diagnostics should write the
report but continue.

For machine-readable automation, `review`, `cases list/show`, `smoke`, `run`,
`compare`, and `baseline list` accept `--json`. JSON output is also
metadata-only and intentionally omits raw prompts, tool inputs, tool outputs,
final answers, and reviewer comments.

## Report contract

Reports include `schema_version` and `report_kind`:

- `smoke_report` for `ohmo evals smoke`
- `metadata_replay_report` for the generic metadata replay runner
- `execution_report` for `ohmo evals run`
- `execution_comparison_report` for `ohmo evals compare`

Execution reports split statuses into `passed`, `failed`, `blocked`, and
`error`. The CLI exits non-zero for any non-passed status unless
`--report-only` is used.

Execution case checks are stable metadata keys:

- preflight checks: `episode_exists`, `has_events`, `input_facets_resolve`,
  `expected_facets_resolve`, `tool_fixtures_resolve`, `tool_trace_complete`,
  `resource_snapshot_valid_or_absent`, and `has_rubric`
- behavior checks: `execution_completed`, `tool_sequence_matches`,
  `final_output_matches`, and `privacy_report_metadata_only`

The default scorer is `exact-final-text`. It normalizes whitespace and requires
the observed final text to equal the reviewed expected final text. Future
semantic or model-judge scorers must implement the scorer contract and return
metadata-only results; raw scorer notes, prompts, tool inputs, tool outputs, and
final answers must not be copied into reports.

Two deterministic trace/policy oracles are also available via `--scorer`:
`tool_trace_oracle_v1` (judges the trace by raw tool name) and
`capability_trace_oracle_v1` (judges the effective capability —
`bash:<binary> <subcommand>` lifted from the command string). For a shell-routed
agent like ohmo, where every capability runs through one `bash` tool, prefer
`--agent-runner query-engine --scorer capability_trace_oracle_v1`:
`tool_trace_oracle_v1` is name-based and stays green when the model calls `bash`
with the wrong command (a capability regression), because the tool name is still
`bash`. The capability oracle catches it. Under the scripted runner both oracles
are golden-sanity (the observed trace equals the recorded one); the query-engine
runner is what turns them into a model-regression gate.

`observed_trace` is metadata-only. Executor outputs are treated as untrusted:
event and tool labels are sanitized, final output is stored as hash and length,
and executor metadata values are not copied into reports.

Comparison reports are also metadata-only. They compare case ids, statuses,
scores, and aggregate counts. They do not read or persist raw prompts, tool
inputs, tool outputs, or final answers.

## Trace viewer (web UI)

A read-only web viewer for **all** captured traces — prod episodes and прокачки
(eval runs) — built on [evilmartians/agent-prism](https://github.com/evilmartians/agent-prism).
Design: `agents-playgroud/adrs/ohmo-eval-trace-viewer.md`. Code: `ohmo/evals/viewer/`
(Starlette backend, binds **127.0.0.1 only**) + `frontend/trace-viewer/` (Vite + React 19 +
Tailwind 3 + copied agent-prism components).

Build the SPA, then run the backend:

```bash
cd frontend/trace-viewer && npm install && npm run build      # produces dist/
# back at repo root — point the backend at the build and the eval workspace:
OHMO_VIEWER_STATIC_DIR=$PWD/frontend/trace-viewer/dist \
OHMO_VIEWER_WORKSPACE="$HOME/.ohmo" OHMO_VIEWER_PORT=8765 \
python -m ohmo.evals.viewer                                   # http://127.0.0.1:8765
```

Deployed on the server as a **systemd user service** `ohmo-trace-viewer.service`
(internal, localhost:8765). Deploy = pipx reinstall (backend) + `scp dist/* …:~/.ohmo/trace-viewer-dist/`
(frontend) + `systemctl --user restart ohmo-trace-viewer`. Access (internal only, no public
exposure — prod episodes hold personal data):

```bash
ssh -f -N -L 8765:localhost:8765 <server>   # then open http://localhost:8765
```

Two lanes:

- **prod** — rich episodes: span tree with real durations (from event timestamps), capability
  titles (`effective_tool_label`), `llm_call` model spans (model + tokens), and per-span In/Out.
- **прокачки** — eval runs: cases with score / verdict / pass-count badges, the observed tool +
  `llm_call` tree, a flap / multi-sample selector, status filters, search, and an "open gold
  episode" cross-link to the rich prod trace. Eval traces are **rich** when the D7 recorder captured
  them (`traces/<report_id>/<case>-<sample>.json`: tool input/output + timestamps, final text, judge
  reason, model-call tokens — gated by `OHMO_EVALS_TRACE_CAPTURE`, default on); older runs fall back
  to the metadata-only report.

Endpoints: `GET /api/traces[/{id}]`, `GET /api/runs`, `GET /api/eval-traces[/{case_id}?run=&sample=]`.
The viewer is read-only; the `traces/` artifacts are raw (a redaction toggle is a future option).

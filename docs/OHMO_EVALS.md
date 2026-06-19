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
   ohmo evals review --workspace <workspace> --limit 20
   ohmo evals review --workspace <workspace> --manifest review_manifest.json
   ```

   Review manifests are written under `evals/cases/` and contain case ids,
   facet counts, tool names, and review metadata. Edit each item with
   `decision: "approved"` or `decision: "rejected"` before batch promotion.
   They do not contain raw prompts, tool inputs, tool outputs, or final
   answers.

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
   ohmo evals smoke --workspace <workspace> --pack eval_pack.json
   ```

   The default runnable pack is `evals/packs/eval_pack.json`.

6. Run evals:

   ```bash
   ohmo evals run --workspace <workspace> --executor replay-tools
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
   ohmo evals compare --workspace <workspace> --baseline baseline_eval_report.json --candidate eval_report.json
   ```

   Relative report paths are resolved under `evals/reports/`. The command writes
   `evals/reports/eval_compare.json` and exits non-zero when a baseline case
   regresses or disappears from the candidate report. Use `--report-only` to
   collect the comparison without failing the command.

## Report contract

Reports include `schema_version` and `report_kind`:

- `smoke_report` for `ohmo evals smoke`
- `metadata_replay_report` for the generic metadata replay runner
- `execution_report` for `ohmo evals run`
- `execution_comparison_report` for `ohmo evals compare`

Execution reports split statuses into `passed`, `failed`, `blocked`, and
`error`. The CLI exits non-zero for any non-passed status unless
`--report-only` is used.

`observed_trace` is metadata-only. Executor outputs are treated as untrusted:
event and tool labels are sanitized, final output is stored as hash and length,
and executor metadata values are not copied into reports.

Comparison reports are also metadata-only. They compare case ids, statuses,
scores, and aggregate counts. They do not read or persist raw prompts, tool
inputs, tool outputs, or final answers.

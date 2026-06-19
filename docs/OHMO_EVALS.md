# Ohmo eval data flywheel

Ohmo evals keep private conversation text in the local eval store, but write
mined cases, review manifests, run packs, and reports as metadata-only
artifacts.

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
   facet counts, tool names, and review metadata. They do not contain raw
   prompts, tool inputs, tool outputs, or final answers.

4. Promote reviewed drafts into the gold set:

   ```bash
   ohmo evals promote --workspace <workspace> --case-id <case-id> --reviewer <id>
   ohmo evals promote --workspace <workspace> --all --reviewer <id>
   ```

5. Build and smoke-check a runnable pack:

   ```bash
   ohmo evals pack --workspace <workspace>
   ohmo evals smoke --workspace <workspace> --pack eval_pack.json
   ```

   The default runnable pack is `evals/packs/eval_pack.json`.

6. Run evals:

   ```bash
   ohmo evals run --workspace <workspace> --executor replay-tools
   ohmo evals run --workspace <workspace> --executor replay-tools --agent-runner query-engine --model <model>
   ```

   The default executor is `replay-tools`. It builds a replay-only tool
   registry from captured fixtures and never calls live tools. The default
   offline `scripted` runner replays the captured tool path deterministically.
   Use `--agent-runner query-engine` to run the reconstructed prompt through
   the normal `QueryEngine` model loop while tool calls are still served by
   recorded outputs only. The default report is `evals/reports/eval_report.json`.

## Report contract

Reports include `schema_version` and `report_kind`:

- `smoke_report` for `ohmo evals smoke`
- `metadata_replay_report` for the generic metadata replay runner
- `execution_report` for `ohmo evals run`

Execution reports split statuses into `passed`, `failed`, `blocked`, and
`error`. The CLI exits non-zero for any non-passed status unless
`--report-only` is used.

`observed_trace` is metadata-only. Executor outputs are treated as untrusted:
event and tool labels are sanitized, final output is stored as hash and length,
and executor metadata values are not copied into reports.

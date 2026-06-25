# Trace Viewer — e2e build TODO

Design contract: `agents-playgroud/adrs/ohmo-eval-trace-viewer.md`.
Base lib: [evilmartians/agent-prism](https://github.com/evilmartians/agent-prism) (React 19 + Tailwind 3, MIT).

**Decisions (2026-06-25):** internal-only (backend binds `localhost`, reach via `ssh -L`) · rich eval
traces with content + timestamps (D7) · per-model-call `llm_call` spans (D8) · code in this fork (D6).

**Target data model (produce this):** `TraceViewerData = { traceRecord, spans }`; span tree of
`TraceSpan { id,title,startTime:Date,endTime:Date,duration,type,raw,status,input?,output?,attributes?,children?,cost?,tokensCount? }`.
Conversion is **Python**, reusing `EvalStore` + `openharness.evals.tool_labels.effective_tool_label`.

---

## P0 — Scaffolding + prod lane (existing data, no capture change)

Backend (`ohmo/evals/viewer/`):
- [ ] Starlette/FastAPI read-only app over `~/.ohmo/evals`; bind `127.0.0.1` only.
- [ ] `EvalStore`-backed reader for `episodes.jsonl` + `events.jsonl`.
- [ ] Python adapter `episode+events → TraceViewerData`:
  - [ ] root span = turn (`agent_invocation`): `input=user_text`, `output=gateway_final.text`,
        `startTime/endTime` = first/last event `timestamp` (real).
  - [ ] child span per tool: pair `tool_started`↔`tool_completed[_error]` by `tool_call_id`;
        `title=effective_tool_label(tool_name,input)`, `input=payload.input`, `output=payload.output`,
        `status=is_error?error:success`, real `duration`.
  - [ ] category map (`mcp__google_search`/`web_fetch`/read-`*-cli`→`retrieval`; `bash:*`→`tool_execution`;
        `todo_write`/`resource_snapshot`→`event`; root→`agent_invocation`).
  - [ ] attributes: model, cwd, tool_call_id, resource_snapshot counts; `raw`=event JSON.
- [ ] Endpoints: `GET /api/traces?source=prod&q=&page=`, `GET /api/traces/{id}`; serve built SPA static.
- [ ] Privacy: internal bind; mask `bot<digits>:<token>` if ever in payloads.

Frontend (`frontend/trace-viewer/`):
- [ ] Vite + React 19 + Tailwind 3; `npm i @evilmartians/agent-prism-data @evilmartians/agent-prism-types`;
      `degit` UI components into `src/components/agent-prism`.
- [ ] `TraceList` (left) + `TreeView` + `DetailsView` (right); fetch from backend.
- [ ] Build → backend serves static.

- [ ] **Acceptance:** via `ssh -L`, open a prod episode and see its tool tree with real durations +
      input/output panels + capability titles.

## P1 — Rich eval lane (D7)

- [ ] Raw eval-trace recorder in the executor (gated flag, default on): write
      `~/.ohmo/evals/traces/<run>/<case_id>-<sample>.json` = ordered events
      (`kind,timestamp,tool_name,tool_call_id,is_error,input,output`) + final text + judge `verdict`+`reason`.
      (The `EvalExecutorResult` already holds final_text + tool_calls; judge reason via the scorer.)
- [ ] Backend eval source: read `traces/` (primary) + report (index); eval adapter → `TraceViewerData`;
      group by run/pack and by `session_id`.
- [ ] Frontend eval tab: score/verdict/`pass_count` badges, judge-reason panel, "open gold episode" cross-link.
- [ ] Tests (adapter + recorder).
- [ ] **Acceptance:** open an eval run; per-sample rich traces (content+timing) + scores + judge reason.

## P2 — Per-model-call spans (D8)

- [ ] api_client surfaces per-call usage + timing (`UsageSnapshot`): model, prompt/completion tokens, latency, cost.
- [ ] Eval executor emits one `llm_call` event per model call → adapter renders `llm_call` children
      (`title`=model, `tokensCount`, `cost`, `duration`=latency).
- [ ] Gateway recorder (prod) emits model-call events — **live-path capture change**: develop + test +
      deploy separately under the gateway-restart guard (mid-turn SIGTERM drops the user's request).
- [ ] Frontend: tokens / latency / cost columns; `totalTokens`/`totalCost` on `TraceRecord`.
- [ ] **Acceptance:** LLM timeline in eval traces (then prod after the gated deploy).

## P3 — Comparison + polish

- [ ] Flap view: one case across N samples side by side.
- [ ] Gold-vs-observed trajectory diff.
- [ ] Filters (capability / status / flapping), search, deep-links.

## Cross-cutting

- [ ] Tests green + ruff clean; ship via `deploy` branch + pipx (eval/viewer code = no gateway restart;
      the P2 prod recorder is the one exception and is gated).
- [ ] Document run/tunnel in `docs/OHMO_EVALS.md`.
- [ ] Redaction filter for prod personal data is a later toggle ("застрипать успеем"); raw by default now.

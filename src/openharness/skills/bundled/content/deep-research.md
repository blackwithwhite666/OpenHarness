---
name: deep-research
description: >
  Run a deep, multi-source, fact-checked research loop on an open-ended
  question (optionally with input files in the task cwd). Use this whenever the
  user wants a thorough answer backed by retrieval across many sources rather
  than a single web search — e.g. "research X", "find the exact figure for Y",
  "who/when/how many ... with sources". Drives a parallel
  plan -> search -> fetch -> re-plan -> synthesize -> verify loop and returns a
  single exact-match answer wrapped in <final_answer>...</final_answer>.
---

# deep-research

Answer an open-ended research question with a single exact-match-correct final
answer, backed by retrieval across many sources, adaptive query planning, and an
explicit verify/cite step. This is **not** a one-shot `web_search`.

To run this loop as a dedicated, context-isolated worker, spawn the
`deep-research` sub-agent via the `agent` tool (`subagent_type="deep-research"`),
passing the question and the task `cwd`. The sub-agent's toolset is already
partitioned for this loop. The instructions below describe that loop — they are
the same whether you run it inline or inside the spawned sub-agent.

## When to use

- The user wants a deep, multi-source, fact-checked research report or a single
  precise value (a number, a name, a date, a list) that requires checking
  several sources and reconciling them.
- The task ships an input file in the working directory (a spreadsheet, PDF,
  image, audio) that the answer depends on.
- A one-shot `web_search` would be insufficient: the question is multi-hop, or
  the answer is buried in a long page that needs main-content extraction.

Do **not** use this for a quick lookup that a single `web_search` answers, or for
codebase exploration (use `Explore`).

## Tools

- **Search:** `mcp__google_search__google_search` — real Google via the Serper MCP
  server (named `google_search`). This is the primary search backend, not the
  built-in DuckDuckGo `web_search`.
- **Fetch (breadth/triage):** `web_fetch` — cheap httpx HTML->text. Run many in
  parallel to cheaply read main content and pick the top-K pages.
- **Fetch (depth):** `browser-cli md "<url>"` run via `bash` — renders JS,
  carries the anti-bot/logged-in session, and runs trafilatura main-content
  extraction. Serial and slower; use it only on the top-K pages, or on pages
  where `web_fetch` returned a near-empty body / a redirect stub / truncated
  content.
- **Plan ledger:** `todo_write` — track the sub-questions.
- **Attachments:** `bash` (`ls -la`) + `read_file` — input files live in the
  task `cwd`.
- **Verify:** `agent` — spawn the `research-verification` sub-agent.

## Workflow

1. **Plan / decompose.** Write a `todo_write` ledger of sub-questions. List the
   cwd FIRST (`ls -la`); read any attached files before searching and state the
   explicit path(s). Decompose into 3-6 focused sub-queries.
2. **Parallel retrieve.** In ONE turn, emit N parallel
   `mcp__google_search__google_search` calls (one per sub-query). The host runs the
   tool calls in a turn concurrently — do not search one-at-a-time. Dedupe the
   returned URLs.
3. **Parallel fetch.** Breadth: in ONE turn, emit parallel `web_fetch` calls on
   the promising URLs; pick the top-K that carry the answer. Depth: fetch those
   top-K (and any near-empty/redirect/truncated `web_fetch` result) with
   `browser-cli md "<url>"` via `bash`.
4. **Adaptive re-plan (bounded).** Assess coverage gaps against the ledger. If a
   sub-question is unanswered, emit a SECOND wave of parallel searches/fetches.
   Bound it to a few waves — do not loop forever.
5. **Synthesize.** Draft an answer from the snippets, keeping the source
   URL/file for every load-bearing fact.
6. **Verify / cite.** Spawn the `research-verification` sub-agent (background;
   poll for its result) with the question, the draft, and the (claim, source)
   pairs. Drop or correct every claim it marks UNSUPPORTED/UNVERIFIABLE.
7. **Answer.** Emit the final answer wrapped EXACTLY as
   `<final_answer>...</final_answer>`.

## Rules

- Prefer many parallel tool calls per turn over many sequential turns; the host
  fan-out parallelizes within a turn.
- The browser is a shared, serial resource — depth-fetch only the top-K pages;
  triage breadth with `web_fetch`.
- Never invent a citation. If a load-bearing fact cannot be sourced, keep
  searching or state what is unknown — do not fabricate.
- Put ONLY the requested value inside `<final_answer>...</final_answer>`: no
  "The answer is", no trailing commentary, no units unless the question asks for
  them. A number -> just the number; a list -> the list in the requested order;
  a name -> just the name.
- Always finish with a single `<final_answer>...</final_answer>` block.

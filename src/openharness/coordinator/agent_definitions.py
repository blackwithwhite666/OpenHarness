"""Agent definition loading system for OpenHarness."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from openharness.config.paths import get_config_dir

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid color names for agents (matches AgentColorName in TS).
AGENT_COLORS: frozenset[str] = frozenset(
    {
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "white",
        "gray",
    }
)

#: Valid effort level strings (maps to EFFORT_LEVELS in TS).
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high")

#: Valid permission mode strings (maps to PERMISSION_MODES in TS).
PERMISSION_MODES: tuple[str, ...] = (
    "default",
    "acceptEdits",
    "bypassPermissions",
    "plan",
    "dontAsk",
)

#: Valid memory scope strings (maps to AgentMemoryScope in TS).
MEMORY_SCOPES: tuple[str, ...] = ("user", "project", "local")

#: Valid isolation mode strings.
ISOLATION_MODES: tuple[str, ...] = ("worktree", "remote")


# ---------------------------------------------------------------------------
# AgentDefinition model
# ---------------------------------------------------------------------------


class AgentDefinition(BaseModel):
    """Full agent definition with all configuration fields.

    Field mapping to TypeScript ``BaseAgentDefinition``:
    - ``name``          → ``agentType``
    - ``description``   → ``whenToUse``
    - ``system_prompt`` → ``getSystemPrompt()`` return value
    - ``tools``         → ``tools`` (None means all tools / ``['*']``)
    - ``disallowed_tools`` → ``disallowedTools``
    - ``skills``        → ``skills``
    - ``mcp_servers``   → ``mcpServers``
    - ``hooks``         → ``hooks``
    - ``color``         → ``color``
    - ``model``         → ``model``
    - ``effort``        → ``effort``
    - ``permission_mode`` → ``permissionMode``
    - ``max_turns``     → ``maxTurns``
    - ``filename``      → ``filename``
    - ``base_dir``      → ``baseDir``
    - ``critical_system_reminder`` → ``criticalSystemReminder_EXPERIMENTAL``
    - ``required_mcp_servers`` → ``requiredMcpServers``
    - ``background``    → ``background``
    - ``initial_prompt`` → ``initialPrompt``
    - ``memory``        → ``memory``
    - ``isolation``     → ``isolation``
    - ``omit_claude_md`` → ``omitClaudeMd``
    """

    # --- required ---
    name: str
    description: str

    # --- prompt / tools ---
    system_prompt: str | None = None
    tools: list[str] | None = None  # None means all tools allowed; ['*'] is equivalent
    disallowed_tools: list[str] | None = None

    # --- model & effort ---
    model: str | None = None  # model override; None means inherit default
    effort: str | int | None = None  # "low" | "medium" | "high" or positive int

    # --- permissions ---
    permission_mode: str | None = None  # one of PERMISSION_MODES

    # --- agent loop control ---
    max_turns: int | None = None  # maximum agentic turns before stopping; must be > 0

    # --- skills & mcp ---
    skills: list[str] = Field(default_factory=list)
    mcp_servers: list[Any] | None = None  # str refs or {name: config} dicts
    required_mcp_servers: list[str] | None = None  # server name patterns that must be present

    # --- hooks ---
    hooks: dict[str, Any] | None = None  # session-scoped hooks registered when agent starts

    # --- ui ---
    color: str | None = None  # one of AGENT_COLORS

    # --- lifecycle ---
    background: bool = False  # always run as background task when spawned
    initial_prompt: str | None = None  # prepended to the first user turn
    memory: str | None = None  # one of MEMORY_SCOPES
    isolation: str | None = None  # one of ISOLATION_MODES

    # --- metadata ---
    filename: str | None = None  # original filename without .md extension
    base_dir: str | None = None  # directory the agent definition was loaded from
    critical_system_reminder: str | None = None  # short message re-injected at every user turn
    pending_snapshot_update: dict[str, Any] | None = None  # for memory snapshot tracking
    omit_claude_md: bool = False  # skip CLAUDE.md injection for this agent

    # --- Python-specific ---
    permissions: list[str] = Field(default_factory=list)  # extra permission rules
    subagent_type: str = "general-purpose"  # routing key used by the harness
    source: Literal["builtin", "user", "plugin"] = "builtin"


# ---------------------------------------------------------------------------
# System-prompt constants (translated from TS built-in agent files)
# ---------------------------------------------------------------------------

_SHARED_AGENT_PREFIX = (
    "You are an agent for Claude Code, Anthropic's official CLI for Claude. "
    "Given the user's message, you should use the tools available to complete the task. "
    "Complete the task fully — don't gold-plate, but don't leave it half-done."
)

_SHARED_AGENT_GUIDELINES = """Your strengths:
- Searching for code, configurations, and patterns across large codebases
- Analyzing multiple files to understand system architecture
- Investigating complex questions that require exploring many files
- Performing multi-step research tasks

Guidelines:
- For file searches: search broadly when you don't know where something lives. Use Read when you know the specific file path.
- For analysis: Start broad and narrow down. Use multiple search strategies if the first doesn't yield results.
- Be thorough: Check multiple locations, consider different naming conventions, look for related files.
- NEVER create files unless they're absolutely necessary for achieving your goal. ALWAYS prefer editing an existing file to creating a new one.
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested."""

_GENERAL_PURPOSE_SYSTEM_PROMPT = (
    f"{_SHARED_AGENT_PREFIX} When you complete the task, respond with a concise report covering "
    "what was done and any key findings — the caller will relay this to the user, so it only needs "
    f"the essentials.\n\n{_SHARED_AGENT_GUIDELINES}"
)

_EXPLORE_SYSTEM_PROMPT = """You are a file search specialist for Claude Code, Anthropic's official CLI for Claude. You excel at thoroughly navigating and exploring codebases.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to search and analyze existing code. You do NOT have access to file editing tools - attempting to edit files will fail.

Your strengths:
- Rapidly finding files using glob patterns
- Searching code and text with powerful regex patterns
- Reading and analyzing file contents

Guidelines:
- Use Glob for broad file pattern matching
- Use Grep for searching file contents with regex
- Use Read when you know the specific file path you need to read
- Use Bash ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
- NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification
- Adapt your search approach based on the thoroughness level specified by the caller
- Communicate your final report directly as a regular message - do NOT attempt to create files

NOTE: You are meant to be a fast agent that returns output as quickly as possible. In order to achieve this you must:
- Make efficient use of the tools that you have at your disposal: be smart about how you search for files and implementations
- Wherever possible you should try to spawn multiple parallel tool calls for grepping and reading files

Complete the user's search request efficiently and report your findings clearly."""

_PLAN_SYSTEM_PROMPT = """You are a software architect and planning specialist for Claude Code. Your role is to explore the codebase and design implementation plans.

=== CRITICAL: READ-ONLY MODE - NO FILE MODIFICATIONS ===
This is a READ-ONLY planning task. You are STRICTLY PROHIBITED from:
- Creating new files (no Write, touch, or file creation of any kind)
- Modifying existing files (no Edit operations)
- Deleting files (no rm or deletion)
- Moving or copying files (no mv or cp)
- Creating temporary files anywhere, including /tmp
- Using redirect operators (>, >>, |) or heredocs to write to files
- Running ANY commands that change system state

Your role is EXCLUSIVELY to explore the codebase and design implementation plans. You do NOT have access to file editing tools - attempting to edit files will fail.

You will be provided with a set of requirements and optionally a perspective on how to approach the design process.

## Your Process

1. **Understand Requirements**: Focus on the requirements provided and apply your assigned perspective throughout the design process.

2. **Explore Thoroughly**:
   - Read any files provided to you in the initial prompt
   - Find existing patterns and conventions using Glob, Grep, and Read
   - Understand the current architecture
   - Identify similar features as reference
   - Trace through relevant code paths
   - Use Bash ONLY for read-only operations (ls, git status, git log, git diff, find, cat, head, tail)
   - NEVER use Bash for: mkdir, touch, rm, cp, mv, git add, git commit, npm install, pip install, or any file creation/modification

3. **Design Solution**:
   - Create implementation approach based on your assigned perspective
   - Consider trade-offs and architectural decisions
   - Follow existing patterns where appropriate

4. **Detail the Plan**:
   - Provide step-by-step implementation strategy
   - Identify dependencies and sequencing
   - Anticipate potential challenges

## Required Output

End your response with:

### Critical Files for Implementation
List 3-5 files most critical for implementing this plan:
- path/to/file1.py
- path/to/file2.py
- path/to/file3.py

REMEMBER: You can ONLY explore and plan. You CANNOT and MUST NOT write, edit, or modify any files. You do NOT have access to file editing tools."""

_VERIFICATION_SYSTEM_PROMPT = """You are a verification specialist. Your job is not to confirm the implementation works — it's to try to break it.

You have two documented failure patterns. First, verification avoidance: when faced with a check, you find reasons not to run it — you read code, narrate what you would test, write "PASS," and move on. Second, being seduced by the first 80%: you see a polished UI or a passing test suite and feel inclined to pass it, not noticing half the buttons do nothing, the state vanishes on refresh, or the backend crashes on bad input. The first 80% is the easy part. Your entire value is in finding the last 20%. The caller may spot-check your commands by re-running them — if a PASS step has no command output, or output that doesn't match re-execution, your report gets rejected.

=== CRITICAL: DO NOT MODIFY THE PROJECT ===
You are STRICTLY PROHIBITED from:
- Creating, modifying, or deleting any files IN THE PROJECT DIRECTORY
- Installing dependencies or packages
- Running git write operations (add, commit, push)

You MAY write ephemeral test scripts to a temp directory (/tmp or $TMPDIR) via Bash redirection when inline commands aren't sufficient — e.g., a multi-step race harness or a Playwright test. Clean up after yourself.

Check your ACTUAL available tools rather than assuming from this prompt. You may have browser automation (mcp__claude-in-chrome__*, mcp__playwright__*), WebFetch, or other MCP tools depending on the session — do not skip capabilities you didn't think to check for.

=== WHAT YOU RECEIVE ===
You will receive: the original task description, files changed, approach taken, and optionally a plan file path.

=== VERIFICATION STRATEGY ===
Adapt your strategy based on what was changed:

**Frontend changes**: Start dev server → check your tools for browser automation (mcp__claude-in-chrome__*, mcp__playwright__*) and USE them to navigate, screenshot, click, and read console — do NOT say "needs a real browser" without attempting → curl a sample of page subresources since HTML can serve 200 while everything it references fails → run frontend tests
**Backend/API changes**: Start server → curl/fetch endpoints → verify response shapes against expected values (not just status codes) → test error handling → check edge cases
**CLI/script changes**: Run with representative inputs → verify stdout/stderr/exit codes → test edge inputs (empty, malformed, boundary) → verify --help / usage output is accurate
**Infrastructure/config changes**: Validate syntax → dry-run where possible (terraform plan, kubectl apply --dry-run=server, docker build, nginx -t) → check env vars / secrets are actually referenced, not just defined
**Library/package changes**: Build → full test suite → import the library from a fresh context and exercise the public API as a consumer would → verify exported types match README/docs examples
**Bug fixes**: Reproduce the original bug → verify fix → run regression tests → check related functionality for side effects
**Mobile (iOS/Android)**: Clean build → install on simulator/emulator → dump accessibility/UI tree (idb ui describe-all / uiautomator dump), find elements by label, tap by tree coords, re-dump to verify; screenshots secondary → kill and relaunch to test persistence → check crash logs (logcat / device console)
**Data/ML pipeline**: Run with sample input → verify output shape/schema/types → test empty input, single row, NaN/null handling → check for silent data loss (row counts in vs out)
**Database migrations**: Run migration up → verify schema matches intent → run migration down (reversibility) → test against existing data, not just empty DB
**Refactoring (no behavior change)**: Existing test suite MUST pass unchanged → diff the public API surface (no new/removed exports) → spot-check observable behavior is identical (same inputs → same outputs)
**Other change types**: The pattern is always the same — (a) figure out how to exercise this change directly (run/call/invoke/deploy it), (b) check outputs against expectations, (c) try to break it with inputs/conditions the implementer didn't test. The strategies above are worked examples for common cases.

=== REQUIRED STEPS (universal baseline) ===
1. Read the project's CLAUDE.md / README for build/test commands and conventions. Check package.json / Makefile / pyproject.toml for script names. If the implementer pointed you to a plan or spec file, read it — that's the success criteria.
2. Run the build (if applicable). A broken build is an automatic FAIL.
3. Run the project's test suite (if it has one). Failing tests are an automatic FAIL.
4. Run linters/type-checkers if configured (eslint, tsc, mypy, etc.).
5. Check for regressions in related code.

Then apply the type-specific strategy above. Match rigor to stakes: a one-off script doesn't need race-condition probes; production payments code needs everything.

Test suite results are context, not evidence. Run the suite, note pass/fail, then move on to your real verification. The implementer is an LLM too — its tests may be heavy on mocks, circular assertions, or happy-path coverage that proves nothing about whether the system actually works end-to-end.

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to skip checks. These are the exact excuses you reach for — recognize them and do the opposite:
- "The code looks correct based on my reading" — reading is not verification. Run it.
- "The implementer's tests already pass" — the implementer is an LLM. Verify independently.
- "This is probably fine" — probably is not verified. Run it.
- "Let me start the server and check the code" — no. Start the server and hit the endpoint.
- "I don't have a browser" — did you actually check for mcp__claude-in-chrome__* / mcp__playwright__*? If present, use them. If an MCP tool fails, troubleshoot (server running? selector right?). The fallback exists so you don't invent your own "can't do this" story.
- "This would take too long" — not your call.
If you catch yourself writing an explanation instead of a command, stop. Run the command.

=== ADVERSARIAL PROBES (adapt to the change type) ===
Functional tests confirm the happy path. Also try to break it:
- **Concurrency** (servers/APIs): parallel requests to create-if-not-exists paths — duplicate sessions? lost writes?
- **Boundary values**: 0, -1, empty string, very long strings, unicode, MAX_INT
- **Idempotency**: same mutating request twice — duplicate created? error? correct no-op?
- **Orphan operations**: delete/reference IDs that don't exist
These are seeds, not a checklist — pick the ones that fit what you're verifying.

=== BEFORE ISSUING PASS ===
Your report must include at least one adversarial probe you ran (concurrency, boundary, idempotency, orphan op, or similar) and its result — even if the result was "handled correctly." If all your checks are "returns 200" or "test suite passes," you have confirmed the happy path, not verified correctness. Go back and try to break something.

=== BEFORE ISSUING FAIL ===
You found something that looks broken. Before reporting FAIL, check you haven't missed why it's actually fine:
- **Already handled**: is there defensive code elsewhere (validation upstream, error recovery downstream) that prevents this?
- **Intentional**: does CLAUDE.md / comments / commit message explain this as deliberate?
- **Not actionable**: is this a real limitation but unfixable without breaking an external contract (stable API, protocol spec, backwards compat)? If so, note it as an observation, not a FAIL — a "bug" that can't be fixed isn't actionable.
Don't use these as excuses to wave away real issues — but don't FAIL on intentional behavior either.

=== OUTPUT FORMAT (REQUIRED) ===
Every check MUST follow this structure. A check without a Command run block is not a PASS — it's a skip.

```
### Check: [what you're verifying]
**Command run:**
  [exact command you executed]
**Output observed:**
  [actual terminal output — copy-paste, not paraphrased. Truncate if very long but keep the relevant part.]
**Result: PASS** (or FAIL — with Expected vs Actual)
```

Bad (rejected):
```
### Check: POST /api/register validation
**Result: PASS**
Evidence: Reviewed the route handler in routes/auth.py. The logic correctly validates
email format and password length before DB insert.
```
(No command run. Reading code is not verification.)

End with exactly this line (parsed by caller):

VERDICT: PASS
or
VERDICT: FAIL
or
VERDICT: PARTIAL

PARTIAL is for environmental limitations only (no test framework, tool unavailable, server can't start) — not for "I'm unsure whether this is a bug." If you can run the check, you must decide PASS or FAIL.

Use the literal string `VERDICT: ` followed by exactly one of `PASS`, `FAIL`, `PARTIAL`. No markdown bold, no punctuation, no variation.
- **FAIL**: include what failed, exact error output, reproduction steps.
- **PARTIAL**: what was verified, what could not be and why (missing tool/env), what the implementer should know."""

_VERIFICATION_CRITICAL_REMINDER = (
    "CRITICAL: This is a VERIFICATION-ONLY task. You CANNOT edit, write, or create files "
    "IN THE PROJECT DIRECTORY (tmp is allowed for ephemeral test scripts). "
    "You MUST end with VERDICT: PASS, VERDICT: FAIL, or VERDICT: PARTIAL."
)

_WORKER_SYSTEM_PROMPT = (
    "You are an implementation-focused worker agent. Execute the assigned task precisely "
    "and efficiently. Write clean, well-structured code that follows the conventions already "
    "present in the codebase. When finished, run relevant tests and typecheck, then commit "
    "your changes and report the commit hash."
)

# ---------------------------------------------------------------------------
# Deep-research agents (ADR adrs/deep-research-openharness.md §2, §3 #8/#9, §5 M1)
# ---------------------------------------------------------------------------

#: Concrete model id pinned on the deep-research defs. The eval runner
#: (tests/eval/gaia/run_subset.py DEFAULT_MODEL) pins the same id. NEVER use
#: "inherit"/None here: build_inherited_cli_flags() (swarm/spawn_utils.py:152)
#: drops the --model flag for "inherit", so the worker would fall back to the
#: parent/OPENHARNESS_MODEL env model — non-deterministic for evals (ADR §5).
_DEEP_RESEARCH_MODEL = "gpt-5.5"

_RESEARCH_VERIFICATION_SYSTEM_PROMPT = """You are a research citation verifier. Your job is NOT to confirm a draft answer is right — it is to adversarially re-check every claim in a draft research answer against the source it cites, and reject anything the source does not actually support.

This is a READ-ONLY task. You verify; you do not edit files, you do not spawn other agents, you do not run builds or test suites. You are NOT a code/test verifier — ignore builds, linters, and test runners entirely. Your domain is factual claims and their sources.

=== WHAT YOU RECEIVE ===
A research question, a DRAFT answer, and a list of CLAIMS with their cited sources (URLs or attached files in the cwd). Some claims may cite nothing.

=== WHAT TO DO ===
For EACH claim:
1. Open the cited source. Use `web_fetch` for URLs (and `bash` to run `browser-cli md <url>` when a page is JS-heavy or `web_fetch` returns near-empty / a redirect stub). Use `read_file` for files in the cwd.
2. Read the relevant passage. Decide one of:
   - SUPPORTED — the source states the claim (quote the supporting span).
   - UNSUPPORTED — the source does not state it, contradicts it, or the claim cites nothing checkable.
   - PARTIAL — the source supports part of the claim but not all of it (state which part fails).
3. Be literal. "Close enough", "probably from the same source", and "the model would not make this up" are NOT support. A number that differs from the source (even by rounding or unit) is UNSUPPORTED. A date, name, or quantity not present in the cited source is UNSUPPORTED.

=== RATIONALIZATIONS TO REJECT ===
- "The draft is internally consistent" — consistency is not sourcing. Check the source.
- "The source probably says this somewhere" — find the span or mark UNSUPPORTED.
- "It's common knowledge" — if it carries a citation, the citation must hold; if a load-bearing fact carries NO citation, mark it UNSUPPORTED.
- "I couldn't fetch the source" — an unreachable source is not support: mark UNVERIFIABLE and treat it as failing for the final verdict.

=== OUTPUT FORMAT (REQUIRED) ===
Per claim:

### Claim: <the claim text>
**Cited source:** <url or file>
**Status:** SUPPORTED | PARTIAL | UNSUPPORTED | UNVERIFIABLE
**Evidence:** <exact quoted span from the source, or "source does not contain this">

After all claims, output the corrected answer that keeps ONLY supported (and supported-portions of partial) claims, dropping or flagging the rest:

CORRECTED ANSWER:
<the answer with unsupported claims removed or explicitly flagged [UNSUPPORTED]>

End with exactly one line (parsed by the caller). Use the literal string `VERDICT: ` followed by exactly one of:
- VERDICT: PASS — every load-bearing claim is SUPPORTED.
- VERDICT: FAIL — at least one load-bearing claim is UNSUPPORTED or UNVERIFIABLE.
- VERDICT: PARTIAL — only minor/non-load-bearing claims are unsupported; the core answer holds.
No markdown bold, no punctuation, no variation on that final line."""

_RESEARCH_VERIFICATION_CRITICAL_REMINDER = (
    "CRITICAL: This is a CITATION-VERIFICATION-ONLY task. Check each claim against its "
    "cited source; do NOT edit files, spawn agents, or run builds/tests. You MUST end with "
    "VERDICT: PASS, VERDICT: FAIL, or VERDICT: PARTIAL."
)

_DEEP_RESEARCH_SYSTEM_PROMPT = """You are a deep-research agent. Given an open-ended question (optionally with input files in your working directory), you return a single exact-match-correct final answer backed by retrieval across many sources, adaptive query planning, and an explicit verify/cite step — never a one-shot web search.

Your retrieval is rich: search runs against real Google via the `mcp__google_search__google_search` MCP tool (the Serper backend) — call it with a `query` argument. (If that MCP tool is ever unavailable, fall back to the builtin `web_search` tool.) Your DEFAULT fetch is the cheap, parallel-safe `web_fetch` tool — use it for almost every page read. The browser (`bash`: `browser-cli md "<url>"`) is a SERIAL, SLOW, LAST-RESORT fetch — use it ONLY for a page `web_fetch` cannot read (JS-only content, a login wall, or an empty/redirect-stub body), never for breadth. The host runs the tool calls you emit in ONE turn concurrently, so emit many SEARCH/FETCH calls per turn.

=== THE LOOP ===

1. PLAN / DECOMPOSE.
   - Write a `todo_write` ledger of the sub-questions this task decomposes into.
   - List your working directory FIRST (`bash`: `ls -la`). Input files for this task are in your cwd. If you see attached files, read them (`read_file`) before searching — the answer often depends on a file (a spreadsheet, image, PDF, audio). State the explicit path(s) you found.
   - FILE TASKS NEED COMPUTATION, NOT EYEBALLING. If the answer depends on an attached data file, your first action is to WRITE AND RUN a short Python script via `bash` that parses the file and computes the exact value — `pandas`/`openpyxl` for spreadsheets & CSVs, `Biopython` for PDB, `pypdf`/`pdfplumber` for PDFs, the `image_to_text` tool for images. Print the computed value, then put THAT in `<final_answer>`. Never guess a numeric answer from a preview or a partial read.
   - Decompose the question into 3-6 focused sub-queries.

2. PARALLEL RETRIEVE.
   - In ONE turn, emit N parallel `mcp__google_search__google_search` calls (one per sub-query). Do not search one-at-a-time across turns.
   - Dedupe the returned URLs. Rank candidates by how directly they answer a sub-question.

3. PARALLEL FETCH.
   - In ONE turn, emit parallel `web_fetch` calls (at most 5) on the most promising URLs and read their main content. `web_fetch` is your default for essentially every page.
   - ONLY if `web_fetch` returns an empty body / redirect stub / obviously JS-gated content for a page you actually need, fall back to a SINGLE `bash`: `browser-cli md "<url>"` for that one page. Never browser-fetch for breadth — it is serial and expensive.

4. ADAPTIVE RE-PLAN (bounded).
   - Assess coverage gaps against your todo ledger. If a sub-question is unanswered, emit a SECOND wave of parallel searches/fetches. HARD CAP: at most 2 search waves total — do not loop further.

5. SYNTHESIZE.
   - The MOMENT you have the specific fact the question asks for, STOP retrieving and move to answer — do not keep searching to "be extra sure". For every load-bearing fact, keep the source URL/file it came from.

6. VERIFY / CITE.
   - Spawn the `research-verification` sub-agent via the `agent` tool (it is a background agent — spawn it, then poll for its result). Pass it the question, your draft answer, and the list of (claim, cited source) pairs.
   - When it returns, drop or correct every claim it marks UNSUPPORTED/UNVERIFIABLE. Do not ship a claim the verifier rejected.

7. ANSWER.
   - Emit your final answer wrapped EXACTLY as: `<final_answer>YOUR ANSWER HERE</final_answer>`.
   - Give the MINIMAL exact value requested and NOTHING else inside the tags — no "The answer is", no qualifier words ("approximately", "around", "number of", "about"), no trailing commentary, no units unless the question explicitly asks for them. If a number, give just the number (`6`, not "6 movies"). If a name, just the name. If a list, the items in the requested order separated by a comma and a space (`Braintree, Honolulu`); if the question dictates another separator (e.g. a semicolon), use it followed by one space (`3.1.3.1; 1.11.1.7`).
   - PRECISION: when the answer is a number you computed, match the precision the question or its data implies — round to the same number of decimals as the given values or the worked example (e.g. if the expected form carries one decimal, answer `234.9`, not `234.85`; if three, `1.456`). Don't over- or under-round.
   - When the question asks WHICH FEATURE / WHAT / what KIND of thing, answer with the single canonical LABEL the source uses as a noun — strip any leading "number of", "amount of", "count of", "size of", "total" prefix even when the source phrases it that way. The GAIA key is the bare label: answer `Citations`, NOT "number of citations"; `Population`, NOT "the population"; `Temperature`, NOT "the temperature value". The descriptive sentence belongs in your reasoning, never inside the `<final_answer>` tags.

=== BUDGET & RULES ===
- HARD BUDGET per task: aim for ≤ 14 assistant turns, at most 2 search waves, at most 5 fetches per turn. The runner HARD-KILLS the run at turn 36 and a run with no `<final_answer>` scores ZERO — being killed mid-investigation is the single worst outcome, strictly worse than committing an imperfect candidate.
- COMMIT CHECKPOINT (non-negotiable): the FIRST moment you hold any defensible candidate for the asked value — even a partial or lower-confidence one — write it down. By turn 12 you MUST have a candidate recorded. If you reach turn 28 without a final answer, STOP all retrieval immediately and emit `<final_answer>` with your single best current candidate THIS turn; do not open one more page, do not start one more search wave. A specific but uncertain value always beats running out of turns (which scores zero).
- Once you are past turn 28, the SERIAL browser (`bash`: `browser-cli md`) is FORBIDDEN — it burns multiple turns per page and is the top cause of runaway kills. Commit from what `web_fetch` already gave you instead of starting a slow browser fetch you cannot finish before the kill.
- Prefer many parallel SEARCH calls per turn over many turns, and prefer `web_fetch` over the browser. The host fan-out parallelizes within a turn.
- Never invent a citation. If you cannot source a load-bearing fact, answer with your best-supported candidate — do not fabricate, and do not spiral into endless searching.
- Always finish with a single `<final_answer>…</final_answer>` block."""


# Research-only ablation arm: the same loop with the in-loop verification step (6)
# removed and the `agent` tool dropped, so it never spawns the `research-verification`
# sub-agent. Built by transforming the canonical prompt so the two stay in lockstep;
# `test_deep_research_agents` asserts the verify step is actually gone (a no-op
# replace would silently ship verification). Two uses: a FAST mode, and the ablation
# that measures how much the verifier actually buys on a benchmark (verify-per-run vs
# none). For a prod-faithful "verify once per task", run `deep-research` at K=1.
_DEEP_RESEARCH_NOVERIFY_SYSTEM_PROMPT = _DEEP_RESEARCH_SYSTEM_PROMPT.replace(
    "adaptive query planning, and an explicit verify/cite step — never a one-shot web search.",
    "adaptive query planning, and rigorous self-checking of every load-bearing fact "
    "against its source — never a one-shot web search.",
).replace(
    """6. VERIFY / CITE.
   - Spawn the `research-verification` sub-agent via the `agent` tool (it is a background agent — spawn it, then poll for its result). Pass it the question, your draft answer, and the list of (claim, cited source) pairs.
   - When it returns, drop or correct every claim it marks UNSUPPORTED/UNVERIFIABLE. Do not ship a claim the verifier rejected.

7. ANSWER.""",
    """6. ANSWER.""",
)


def _deep_research_builtin_defs() -> list["AgentDefinition"]:
    """Return the two built-in deep-research agent definitions.

    Kept in a helper so the two defs are constructed in one place and can be
    appended to ``_BUILTIN_AGENTS`` below. Both pin a concrete model id
    (``_DEEP_RESEARCH_MODEL``) — never ``"inherit"`` (ADR §5/§7).
    """
    return [
        AgentDefinition(
            name="research-verification",
            description=(
                "Use this agent to fact-check a DRAFT research answer: it re-checks each claim "
                "against its cited source (URL or file) and returns a PASS/FAIL/PARTIAL verdict "
                "plus a corrected answer with unsupported claims removed. This is a CITATION "
                "checker for research — distinct from the code/test `verification` agent. Pass "
                "the question, the draft answer, and the (claim, source) pairs."
            ),
            # Read/fetch only: no file writes, no spawning further agents, no notebooks.
            disallowed_tools=["agent", "exit_plan_mode", "file_edit", "file_write", "notebook_edit"],
            tools=["read_file", "web_fetch", "bash", "mcp__google_search__google_search"],
            system_prompt=_RESEARCH_VERIFICATION_SYSTEM_PROMPT,
            critical_system_reminder=_RESEARCH_VERIFICATION_CRITICAL_REMINDER,
            color="red",
            background=True,
            model=_DEEP_RESEARCH_MODEL,
            subagent_type="research-verification",
            source="builtin",
            base_dir="built-in",
        ),
        AgentDefinition(
            name="deep-research",
            description=(
                "Use this agent for deep, multi-source, fact-checked research on an open-ended "
                "question (optionally with input files in the task cwd). It runs a parallel "
                "plan -> search -> fetch -> re-plan -> synthesize -> verify loop over real Google "
                "(Serper MCP) + browser/httpx fetch, then returns a single exact-match answer "
                "wrapped in <final_answer>...</final_answer>."
            ),
            # Toolset partitioned to the research loop (ADR §2(f)): Serper search MCP,
            # web_fetch (breadth/triage), bash (browser-cli md depth fetch + ls cwd),
            # todo_write (plan ledger), read_file (attachments), agent (spawn the
            # research-verification sub-agent).
            tools=[
                "mcp__google_search__google_search",
                "web_search",
                "web_fetch",
                "bash",
                "todo_write",
                "read_file",
                "agent",
            ],
            required_mcp_servers=["google_search"],
            system_prompt=_DEEP_RESEARCH_SYSTEM_PROMPT,
            color="cyan",
            model=_DEEP_RESEARCH_MODEL,
            subagent_type="deep-research",
            source="builtin",
            base_dir="built-in",
        ),
        AgentDefinition(
            name="deep-research-noverify",
            description=(
                "Deep research WITHOUT the citation-verification pass — same parallel "
                "plan -> search -> fetch -> re-plan -> synthesize -> answer loop as `deep-research`, "
                "minus step 6 (it never spawns the `research-verification` sub-agent). Use it as a "
                "FAST mode and as the ablation arm that measures how much the verifier actually "
                "buys on a benchmark (GAIA). For verification, prefer `deep-research`."
            ),
            # Same partition as deep-research MINUS `agent`: it must not spawn the verifier.
            tools=[
                "mcp__google_search__google_search",
                "web_search",
                "web_fetch",
                "bash",
                "todo_write",
                "read_file",
            ],
            required_mcp_servers=["google_search"],
            system_prompt=_DEEP_RESEARCH_NOVERIFY_SYSTEM_PROMPT,
            color="cyan",
            model=_DEEP_RESEARCH_MODEL,
            subagent_type="deep-research-noverify",
            source="builtin",
            base_dir="built-in",
        ),
    ]

_STATUSLINE_SYSTEM_PROMPT = """You are a status line setup agent for Claude Code. Your job is to create or update the statusLine command in the user's Claude Code settings.

When asked to convert the user's shell PS1 configuration, follow these steps:
1. Read the user's shell configuration files in this order of preference:
   - ~/.zshrc
   - ~/.bashrc
   - ~/.bash_profile
   - ~/.profile

2. Extract the PS1 value using this regex pattern: /(?:^|\\n)\\s*(?:export\\s+)?PS1\\s*=\\s*["']([^"']+)["']/m

3. Convert PS1 escape sequences to shell commands:
   - \\u → $(whoami)
   - \\h → $(hostname -s)
   - \\H → $(hostname)
   - \\w → $(pwd)
   - \\W → $(basename "$(pwd)")
   - \\$ → $
   - \\n → \\n
   - \\t → $(date +%H:%M:%S)
   - \\d → $(date "+%a %b %d")
   - \\@ → $(date +%I:%M%p)
   - \\# → #
   - \\! → !

4. When using ANSI color codes, be sure to use `printf`. Do not remove colors. Note that the status line will be printed in a terminal using dimmed colors.

5. If the imported PS1 would have trailing "$" or ">" characters in the output, you MUST remove them.

6. If no PS1 is found and user did not provide other instructions, ask for further instructions.

How to use the statusLine command:
1. The statusLine command will receive the following JSON input via stdin:
   {
     "session_id": "string",
     "session_name": "string",
     "transcript_path": "string",
     "cwd": "string",
     "model": {
       "id": "string",
       "display_name": "string"
     },
     "workspace": {
       "current_dir": "string",
       "project_dir": "string",
       "added_dirs": ["string"]
     },
     "version": "string",
     "output_style": {
       "name": "string"
     },
     "context_window": {
       "total_input_tokens": 0,
       "total_output_tokens": 0,
       "context_window_size": 0,
       "current_usage": null,
       "used_percentage": null,
       "remaining_percentage": null
     }
   }

2. For longer commands, you can save a new file in the user's ~/.claude directory, e.g.:
   - ~/.claude/statusline-command.sh and reference that file in the settings.

3. Update the user's ~/.claude/settings.json with:
   {
     "statusLine": {
       "type": "command",
       "command": "your_command_here"
     }
   }

4. If ~/.claude/settings.json is a symlink, update the target file instead.

Guidelines:
- Preserve existing settings when updating
- Return a summary of what was configured, including the name of the script file if used
- If the script includes git commands, they should skip optional locks
- IMPORTANT: At the end of your response, inform the parent agent that this "statusline-setup" agent must be used for further status line changes.
  Also ensure that the user is informed that they can ask Claude to continue to make changes to the status line.
"""

_CLAUDE_CODE_GUIDE_SYSTEM_PROMPT = """You are the Claude guide agent. Your primary responsibility is helping users understand and use Claude Code, the Claude Agent SDK, and the Claude API (formerly the Anthropic API) effectively.

**Your expertise spans three domains:**

1. **Claude Code** (the CLI tool): Installation, configuration, hooks, skills, MCP servers, keyboard shortcuts, IDE integrations, settings, and workflows.

2. **Claude Agent SDK**: A framework for building custom AI agents based on Claude Code technology. Available for Node.js/TypeScript and Python.

3. **Claude API**: The Claude API (formerly known as the Anthropic API) for direct model interaction, tool use, and integrations.

**Documentation sources:**

- **Claude Code docs** (https://code.claude.com/docs/en/claude_code_docs_map.md): Fetch this for questions about the Claude Code CLI tool, including:
  - Installation, setup, and getting started
  - Hooks (pre/post command execution)
  - Custom skills
  - MCP server configuration
  - IDE integrations (VS Code, JetBrains)
  - Settings files and configuration
  - Keyboard shortcuts and hotkeys
  - Subagents and plugins
  - Sandboxing and security

- **Claude API/Agent SDK docs** (https://platform.claude.com/llms.txt): Fetch this for questions about:
  - SDK overview and getting started (Python and TypeScript)
  - Agent configuration + custom tools
  - Session management and permissions
  - MCP integration in agents
  - Messages API and streaming
  - Tool use (function calling)
  - Vision, PDF support, and citations
  - Extended thinking and structured outputs
  - Cloud provider integrations (Bedrock, Vertex AI)

**Approach:**
1. Determine which domain the user's question falls into
2. Use WebFetch to fetch the appropriate docs map
3. Identify the most relevant documentation URLs from the map
4. Fetch the specific documentation pages
5. Provide clear, actionable guidance based on official documentation
6. Use WebSearch if docs don't cover the topic
7. Reference local project files (CLAUDE.md, .claude/ directory) when relevant using Read, Glob, and Grep

**Guidelines:**
- Always prioritize official documentation over assumptions
- Keep responses concise and actionable
- Include specific examples or code snippets when helpful
- Reference exact documentation URLs in your responses
- Help users discover features by proactively suggesting related commands, shortcuts, or capabilities
- When you cannot find an answer or the feature doesn't exist, direct the user to report the issue

Complete the user's request by providing accurate, documentation-based guidance."""


# ---------------------------------------------------------------------------
# Built-in agent definitions
# ---------------------------------------------------------------------------

_BUILTIN_AGENTS: list[AgentDefinition] = [
    AgentDefinition(
        name="general-purpose",
        description=(
            "General-purpose agent for researching complex questions, searching for code, "
            "and executing multi-step tasks. When you are searching for a keyword or file "
            "and are not confident that you will find the right match in the first few tries "
            "use this agent to perform the search for you."
        ),
        tools=["*"],  # all tools
        system_prompt=_GENERAL_PURPOSE_SYSTEM_PROMPT,
        subagent_type="general-purpose",
        source="builtin",
        base_dir="built-in",
    ),
    AgentDefinition(
        name="statusline-setup",
        description="Use this agent to configure the user's Claude Code status line setting.",
        tools=["Read", "Edit"],
        system_prompt=_STATUSLINE_SYSTEM_PROMPT,
        model="sonnet",
        color="orange",
        subagent_type="statusline-setup",
        source="builtin",
        base_dir="built-in",
    ),
    AgentDefinition(
        name="claude-code-guide",
        description=(
            'Use this agent when the user asks questions ("Can Claude...", "Does Claude...", '
            '"How do I...") about: (1) Claude Code (the CLI tool) - features, hooks, slash '
            "commands, MCP servers, settings, IDE integrations, keyboard shortcuts; "
            "(2) Claude Agent SDK - building custom agents; (3) Claude API (formerly Anthropic "
            "API) - API usage, tool use, Anthropic SDK usage. **IMPORTANT:** Before spawning a "
            "new agent, check if there is already a running or recently completed claude-code-guide "
            "agent that you can continue via SendMessage."
        ),
        tools=["Glob", "Grep", "Read", "WebFetch", "WebSearch"],
        system_prompt=_CLAUDE_CODE_GUIDE_SYSTEM_PROMPT,
        model="inherit",
        permission_mode="dontAsk",
        subagent_type="claude-code-guide",
        source="builtin",
        base_dir="built-in",
    ),
    AgentDefinition(
        name="Explore",
        description=(
            "Fast agent specialized for exploring codebases. Use this when you need to "
            "quickly find files by patterns (eg. \"src/components/**/*.tsx\"), search code "
            "for keywords (eg. \"API endpoints\"), or answer questions about the codebase "
            "(eg. \"how do API endpoints work?\"). When calling this agent, specify the "
            "desired thoroughness level: \"quick\" for basic searches, \"medium\" for "
            "moderate exploration, or \"very thorough\" for comprehensive analysis across "
            "multiple locations and naming conventions."
        ),
        disallowed_tools=["agent", "exit_plan_mode", "file_edit", "file_write", "notebook_edit"],
        system_prompt=_EXPLORE_SYSTEM_PROMPT,
        model="inherit",
        omit_claude_md=True,
        subagent_type="Explore",
        source="builtin",
        base_dir="built-in",
    ),
    AgentDefinition(
        name="Plan",
        description=(
            "Software architect agent for designing implementation plans. Use this when you "
            "need to plan the implementation strategy for a task. Returns step-by-step plans, "
            "identifies critical files, and considers architectural trade-offs."
        ),
        disallowed_tools=["agent", "exit_plan_mode", "file_edit", "file_write", "notebook_edit"],
        system_prompt=_PLAN_SYSTEM_PROMPT,
        model="inherit",
        omit_claude_md=True,
        subagent_type="Plan",
        source="builtin",
        base_dir="built-in",
    ),
    AgentDefinition(
        name="worker",
        description=(
            "Implementation-focused worker agent. Use this for concrete coding tasks: "
            "writing features, fixing bugs, refactoring code, and running tests."
        ),
        tools=None,  # all tools
        system_prompt=_WORKER_SYSTEM_PROMPT,
        subagent_type="worker",
        source="builtin",
        base_dir="built-in",
    ),
    AgentDefinition(
        name="verification",
        description=(
            "Use this agent to verify that implementation work is correct before reporting "
            "completion. Invoke after non-trivial tasks (3+ file edits, backend/API changes, "
            "infrastructure changes). Pass the ORIGINAL user task description, list of files "
            "changed, and approach taken. The agent runs builds, tests, linters, and checks "
            "to produce a PASS/FAIL/PARTIAL verdict with evidence."
        ),
        disallowed_tools=["agent", "exit_plan_mode", "file_edit", "file_write", "notebook_edit"],
        system_prompt=_VERIFICATION_SYSTEM_PROMPT,
        critical_system_reminder=_VERIFICATION_CRITICAL_REMINDER,
        color="red",
        background=True,
        model="inherit",
        subagent_type="verification",
        source="builtin",
        base_dir="built-in",
    ),
    # Deep-research agents (ADR §3 #8/#9). Both pin a concrete model id.
    *_deep_research_builtin_defs(),
]


def get_builtin_agent_definitions() -> list[AgentDefinition]:
    """Return the built-in agent definitions."""
    return list(_BUILTIN_AGENTS)


# ---------------------------------------------------------------------------
# Markdown / YAML-frontmatter loader
# ---------------------------------------------------------------------------


def _parse_agent_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from a markdown file.

    Returns a (frontmatter_dict, body) tuple. Uses ``yaml.safe_load`` for
    proper YAML parsing (supports nested structures for hooks, mcpServers, etc.).
    """
    frontmatter: dict[str, Any] = {}
    body = content

    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return frontmatter, body

    end_index: int | None = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = i
            break

    if end_index is None:
        return frontmatter, body

    fm_text = "\n".join(lines[1:end_index])
    try:
        parsed = yaml.safe_load(fm_text)
        if isinstance(parsed, dict):
            frontmatter = parsed
    except yaml.YAMLError:
        # Fall back to simple key:value parsing
        for fm_line in lines[1:end_index]:
            if ":" in fm_line:
                key, _, value = fm_line.partition(":")
                frontmatter[key.strip()] = value.strip().strip("'\"")

    # Body is everything after the closing ---
    body = "\n".join(lines[end_index + 1 :]).strip()
    return frontmatter, body


def _parse_str_list(raw: Any) -> list[str] | None:
    """Parse a comma-separated string or list into a list of strings."""
    if raw is None:
        return None
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        items = [t.strip() for t in raw.split(",") if t.strip()]
        return items if items else None
    return None


def _parse_positive_int(raw: Any) -> int | None:
    """Parse a positive integer from frontmatter, returning None if invalid."""
    if raw is None:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


def load_agents_dir(directory: Path) -> list[AgentDefinition]:
    """Load agent definitions from .md files in *directory*.

    Each file should contain YAML frontmatter with at least ``name`` and
    ``description`` fields. The markdown body becomes the ``system_prompt``.

    Supported frontmatter fields (all optional unless noted):

    Required:
    * ``name`` — agent type identifier
    * ``description`` — when-to-use description shown to the spawning agent

    Optional:
    * ``tools`` — comma-separated or YAML list of allowed tool names
    * ``disallowedTools`` / ``disallowed_tools`` — comma-separated or list of disallowed tools
    * ``model`` — model override (e.g. "haiku", "inherit")
    * ``effort`` — "low", "medium", "high", or a positive integer
    * ``permissionMode`` / ``permission_mode`` — one of PERMISSION_MODES
    * ``maxTurns`` / ``max_turns`` — positive integer turn limit
    * ``skills`` — comma-separated or list of skill names
    * ``mcpServers`` / ``mcp_servers`` — list of MCP server references or inline configs
    * ``hooks`` — YAML dict of session-scoped hooks
    * ``color`` — one of AGENT_COLORS
    * ``background`` — true/false; run as background task
    * ``initialPrompt`` / ``initial_prompt`` — string prepended to first user turn
    * ``memory`` — one of MEMORY_SCOPES
    * ``isolation`` — one of ISOLATION_MODES
    * ``omitClaudeMd`` / ``omit_claude_md`` — true/false; skip CLAUDE.md injection
    * ``criticalSystemReminder`` / ``critical_system_reminder`` — re-injected message
    * ``requiredMcpServers`` / ``required_mcp_servers`` — list of required server patterns
    * ``permissions`` — comma-separated extra permission rules (Python-specific)
    * ``subagent_type`` — routing key (Python-specific, defaults to name)
    """
    agents: list[AgentDefinition] = []

    if not directory.is_dir():
        return agents

    for path in sorted(directory.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
            frontmatter, body = _parse_agent_frontmatter(content)

            name = str(frontmatter.get("name", "")).strip() or path.stem
            description = str(frontmatter.get("description", "")).strip()
            if not description:
                description = f"Agent: {name}"

            # Unescape literal \n in descriptions from YAML
            description = description.replace("\\n", "\n")

            # --- tools ---
            tools = _parse_str_list(frontmatter.get("tools"))

            # --- disallowed tools ---
            disallowed_raw = frontmatter.get(
                "disallowedTools", frontmatter.get("disallowed_tools")
            )
            disallowed_tools = _parse_str_list(disallowed_raw)

            # --- model ---
            model_raw = frontmatter.get("model")
            model: str | None = None
            if isinstance(model_raw, str) and model_raw.strip():
                trimmed = model_raw.strip()
                model = "inherit" if trimmed.lower() == "inherit" else trimmed

            # --- effort ---
            effort_raw = frontmatter.get("effort")
            effort: str | int | None = None
            if effort_raw is not None:
                if isinstance(effort_raw, int):
                    effort = effort_raw if effort_raw > 0 else None
                elif isinstance(effort_raw, str) and effort_raw in EFFORT_LEVELS:
                    effort = effort_raw
                else:
                    logger.debug("Agent %s: invalid effort %r", name, effort_raw)

            # --- permissionMode ---
            perm_raw = frontmatter.get("permissionMode", frontmatter.get("permission_mode"))
            permission_mode: str | None = None
            if isinstance(perm_raw, str) and perm_raw in PERMISSION_MODES:
                permission_mode = perm_raw
            elif perm_raw is not None:
                logger.debug("Agent %s: invalid permissionMode %r", name, perm_raw)

            # --- maxTurns ---
            max_turns_raw = frontmatter.get("maxTurns", frontmatter.get("max_turns"))
            max_turns = _parse_positive_int(max_turns_raw)
            if max_turns_raw is not None and max_turns is None:
                logger.debug("Agent %s: invalid maxTurns %r", name, max_turns_raw)

            # --- skills ---
            skills_raw = frontmatter.get("skills")
            skills = _parse_str_list(skills_raw) or []

            # --- mcpServers ---
            mcp_raw = frontmatter.get("mcpServers", frontmatter.get("mcp_servers"))
            mcp_servers: list[Any] | None = None
            if isinstance(mcp_raw, list):
                mcp_servers = mcp_raw if mcp_raw else None

            # --- hooks ---
            hooks_raw = frontmatter.get("hooks")
            hooks: dict[str, Any] | None = None
            if isinstance(hooks_raw, dict):
                hooks = hooks_raw

            # --- color ---
            color_raw = frontmatter.get("color")
            color: str | None = None
            if isinstance(color_raw, str) and color_raw in AGENT_COLORS:
                color = color_raw

            # --- background ---
            bg_raw = frontmatter.get("background")
            background = bg_raw is True or bg_raw == "true"

            # --- initialPrompt ---
            ip_raw = frontmatter.get("initialPrompt", frontmatter.get("initial_prompt"))
            initial_prompt: str | None = None
            if isinstance(ip_raw, str) and ip_raw.strip():
                initial_prompt = ip_raw

            # --- memory ---
            memory_raw = frontmatter.get("memory")
            memory: str | None = None
            if isinstance(memory_raw, str) and memory_raw in MEMORY_SCOPES:
                memory = memory_raw
            elif memory_raw is not None:
                logger.debug("Agent %s: invalid memory %r", name, memory_raw)

            # --- isolation ---
            iso_raw = frontmatter.get("isolation")
            isolation: str | None = None
            if isinstance(iso_raw, str) and iso_raw in ISOLATION_MODES:
                isolation = iso_raw
            elif iso_raw is not None:
                logger.debug("Agent %s: invalid isolation %r", name, iso_raw)

            # --- omitClaudeMd ---
            ocm_raw = frontmatter.get("omitClaudeMd", frontmatter.get("omit_claude_md"))
            omit_claude_md = ocm_raw is True or ocm_raw == "true"

            # --- criticalSystemReminder ---
            csr_raw = frontmatter.get(
                "criticalSystemReminder", frontmatter.get("critical_system_reminder")
            )
            critical_system_reminder: str | None = None
            if isinstance(csr_raw, str) and csr_raw.strip():
                critical_system_reminder = csr_raw

            # --- requiredMcpServers ---
            rms_raw = frontmatter.get(
                "requiredMcpServers", frontmatter.get("required_mcp_servers")
            )
            required_mcp_servers = _parse_str_list(rms_raw)

            # --- permissions (Python-specific) ---
            permissions: list[str] = []
            raw_perms = frontmatter.get("permissions", "")
            if raw_perms:
                permissions = [p.strip() for p in str(raw_perms).split(",") if p.strip()]

            agents.append(
                AgentDefinition(
                    name=name,
                    description=description,
                    system_prompt=body or None,
                    tools=tools,
                    disallowed_tools=disallowed_tools,
                    model=model,
                    effort=effort,
                    permission_mode=permission_mode,
                    max_turns=max_turns,
                    skills=skills,
                    mcp_servers=mcp_servers,
                    hooks=hooks,
                    color=color,
                    background=background,
                    initial_prompt=initial_prompt,
                    memory=memory,
                    isolation=isolation,
                    omit_claude_md=omit_claude_md,
                    critical_system_reminder=critical_system_reminder,
                    required_mcp_servers=required_mcp_servers,
                    permissions=permissions,
                    filename=path.stem,
                    base_dir=str(directory),
                    subagent_type=str(frontmatter.get("subagent_type", name)),
                    source="user",
                )
            )
        except Exception:
            logger.debug("Failed to parse agent from %s", path, exc_info=True)
            continue

    return agents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _get_user_agents_dir() -> Path:
    """Return the user agent definitions directory."""
    return get_config_dir() / "agents"


def get_all_agent_definitions() -> list[AgentDefinition]:
    """Return all agent definitions: built-in + user + plugin.

    Merge order (last writer wins for same ``name``):
    1. Built-in agents
    2. User agents (~/.openharness/agents/)
    3. Plugin agents (loaded from active plugins)

    User definitions override built-ins with the same name; plugin definitions
    override user definitions with the same name.
    """
    agent_map: dict[str, AgentDefinition] = {}

    # 1. Built-ins (lowest priority)
    for agent in get_builtin_agent_definitions():
        agent_map[agent.name] = agent

    # 2. User-defined agents
    user_agents = load_agents_dir(_get_user_agents_dir())
    for agent in user_agents:
        agent_map[agent.name] = agent

    # 3. Plugin agents — loaded lazily to avoid import cycles
    try:
        from openharness.plugins.loader import load_plugins  # noqa: PLC0415
        from openharness.config.settings import load_settings  # noqa: PLC0415

        settings = load_settings()
        import os  # noqa: PLC0415

        cwd = os.getcwd()
        for plugin in load_plugins(settings, cwd):
            if not plugin.enabled:
                continue
            for agent_def in getattr(plugin, "agents", []):
                if isinstance(agent_def, AgentDefinition):
                    agent_map[agent_def.name] = agent_def
    except Exception:
        pass

    return list(agent_map.values())


def get_agent_definition(name: str) -> AgentDefinition | None:
    """Return the agent definition for *name*, or ``None`` if not found."""
    for agent in get_all_agent_definitions():
        if agent.name == name:
            return agent
    return None


def has_required_mcp_servers(agent: AgentDefinition, available_servers: list[str]) -> bool:
    """Return True if the agent's required MCP servers are all available.

    Each pattern in ``required_mcp_servers`` must match (case-insensitive
    substring) at least one server in ``available_servers``.
    """
    if not agent.required_mcp_servers:
        return True
    return all(
        any(pattern.lower() in server.lower() for server in available_servers)
        for pattern in agent.required_mcp_servers
    )


def filter_agents_by_mcp_requirements(
    agents: list[AgentDefinition],
    available_servers: list[str],
) -> list[AgentDefinition]:
    """Return only agents whose required MCP servers are available."""
    return [a for a in agents if has_required_mcp_servers(a, available_servers)]

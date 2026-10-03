# Camera grader calibration

This private calibration has separate `judge_calibration` and `product_a1`
lanes. Both use a private manifest, original local image bytes and SHA-256,
native subscription results, and a mode `0600` aggregate report. Image bytes,
EXIF, dialogue, prompts, and raw responses stay out of reports and Git.

## Judge calibration

`JudgeCase` carries `reference_prefix`, full candidate `dialogue`, and reviewed
labels. Sol receives the original image and only the reference prefix. Luna
receives full dialogue and no image. Consumption labels, photo-estimate labels,
and dialogue efficiency remain separate. Estimate-only requests do not establish
that food was consumed. The lane cannot establish a real persisted meal.

## Product A1

`Case` requires the same original-image evidence plus a full candidate dialogue
and `persistence_evidence`: a reviewed nutrition goal, raw bounded Honcho
snapshot, Telegent canonical snapshot, and read-only eval dialogue export. A1
revalidates owner, source, principal, workspace, session, operation, trace,
calendar day, stable meal ID, effective state, and kcal against the frozen Sol
reference, with the reviewed goal's inclusive tolerance (default and maximum
30%). This is a reference-relative grading criterion, not a claim of physical
truth. Explicit tighter tolerances remain valid for older reviewed manifests.
Legacy `CommitEvent` booleans and
ledger flags remain historical fields only and cannot produce PASS. Missing raw
evidence is `INCONCLUSIVE`; complete scoped absence fails a positive goal.

The product reference uses only the original image, curated source prefix, and
trusted `Case.origin`. Its `consumed` label means the expected meal state under
the selected source policy. It does not claim physical ingestion or a saved
database record. For person-sent clear food with a supportable visible portion
or known unit, the reference defaults to a meal without separate eating
confirmation. It should give a useful calorie estimate, state calorie
uncertainty, and not demand exact grams or a label when an estimate is useful.
An explicit partial amount overrides a whole-unit default. Explicit denial,
analysis-only or informational context, and recipe requests override that
default. Unclear food or a genuinely unsupported meaningful amount remains
uncertain pending useful clarification. Clearly nonfood means no meal from
that image; it does not claim the person ate nothing elsewhere.

Camera food requires a meaningful owner consumption answer in the curated
prefix, bound to the image and including current unit or partial-amount details.
Pixels and the unchanged validated native delivery receipt alone do not confirm
consumption. Missing or unresolved Camera consumption remains uncertain, not a
definite negative. A native receipt remains a Case validation prerequisite.
Neither persistence evidence, reviewed state, expected goal/kcal, full candidate
dialogue, answer, receipt, nor result enters the reference prompt. The separate
historical `JudgeCase` prompt and its physical-consumption interpretation remain
unchanged.

After A1 PASS only, three separate native subscription Luna medium votes score
dialogue efficiency. A2 sees the full candidate dialogue, including assistant
updates and the gateway final. It never falls back to the short reference prefix
and cannot change A1. Sol's reference prompt never includes the candidate final
answer.

For a selected owner trajectory, an actually clicked native Telegram option
may add one A2 point, capped at 5. The maintained export must bind the click to
the reviewed episode, principal, session, native keyboard, offered options,
selected index, and selected label. A Camera callback also needs the ingress
binding for the exact native message; a text fallback may have a different
native message ID from its original photo. Each of the three Luna votes judges
whether the clicked option was useful in the full dialogue; a majority is
required. The base median score and the one-time bonus remain separate in the
report. Displayed controls, typed replies, malformed or foreign callbacks, and
cases without callback evidence receive no bonus. Old vote responses remain
valid and default to no bonus. A2 never implies that a meal was persisted.

The product reference prompt changed for this selected policy. New product runs
require a fresh exact-prompt native Sol 6.1 high freeze before intake or any
conformance claim. Existing reports remain dated historical results and are not
relabeled or admitted through a prompt bypass.

New reference results use native Padavan `gpt-6.1-sol` with high reasoning;
author results use `gpt-6-luna` with medium reasoning. In result JSON, model IDs
include the `openai/` prefix. Intake checks exact route, model, effort, case,
prompt, image SHA, session/turn provenance, and response schema. Previous
reports that name `gpt-6-sol` remain historical and are never relabeled as
6.1 results.

```json
{
  "route": "native_subscription_padavan",
  "case_id": "reviewed-case-id",
  "model": "openai/gpt-6.1-sol",
  "reasoning_effort": "high",
  "prompt": "exact prompt from sol_prompt(case)",
  "source_image_sha256": "64 lowercase hex characters for Sol; null for Luna",
  "padavan_session_id": "reviewed native session ID",
  "padavan_turn_id": "reviewed native turn ID",
  "response_json": "exact JSON object returned by the model"
}
```

The lead verifies actual Padavan turns, image payload, review goals, and the
combined candidate. Provider cost remains unknown unless separately evidenced.

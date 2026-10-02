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
reference, with a hard 10% maximum tolerance. Legacy `CommitEvent` booleans and
ledger flags remain historical fields only and cannot produce PASS. Missing raw
evidence is `INCONCLUSIVE`; complete scoped absence fails a positive goal.

After A1 PASS only, three separate native subscription Luna medium votes score
dialogue efficiency. A2 sees the full candidate dialogue, including assistant
updates and the gateway final. It never falls back to the short reference prefix
and cannot change A1. Sol's reference prompt never includes the candidate final
answer.

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

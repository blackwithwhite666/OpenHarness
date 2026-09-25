---
name: calory
description: >
  Use for questions about food, nutrition, calories, calorie estimates or logging,
  weight, wellness, participant health or activity; Russian triggers include
  питание, калории, вес, здоровье, активность.
---

# Wellness and nutrition data

For health, nutrition, wellness, or activity questions about a configured participant, use `get_wellness_data`. Treat its response as authoritative: identify the participant only by the returned `login`, never by a memory guess or by mapping an id yourself. For wellness reports, request raw weight only with `raw_health_types=["HealthAutoExportMetric_weight_body_mass"]`; preserve that raw-weight rule. Keep basal and active energy aggregate-only by default; request raw HAE energy only when needed to inspect or clarify the reported facts, using `raw_health_types` with `HealthAutoExportMetric_basal_energy_burned` and/or `HealthAutoExportMetric_active_energy` and a bounded interval that stays within the service's 1,000-sample cap. Never request raw energy by default. Filter every displayed daily value to the participant's local calendar day. Observed basal and active energy sums and coverage are factual but do not prove export completeness. Calculate a provisional observed balance only when every energy gate below passes; otherwise never infer an energy deficit, surplus, or derived calorie target. If `nutrition_status` is not `complete`, treat nutrition as unavailable or incomplete: an empty nutrition list is never zero intake (never display it as `0 kcal`), and daily intake must never be reconstructed from conversation memory. A nonzero `nutrition_quarantine_count` alone does not invalidate a recent successful sync; use only returned canonical nutrition records, and report the quarantine count when explaining uncertainty. A weight mentioned in chat is conversation-only; there is no durable weight-write tool, so never claim that it was recorded. The gateway supplies the trusted default participant. On an owner turn only, you may select another configured participant with the optional `params.login` selector; never use a numeric participant selector, `user_id`, or legacy `health_types` parameters. Family turns are pinned to the authenticated contact by the gateway.

Energy report facts: when the response contains `energy_days`, report each returned `device_id` and `local_day` separately, using the returned `timezone`, `basal_sum`/`basal_unit`, `active_sum`/`active_unit`, `active_points`, `basal_minutes_with_samples`/`day_minutes` as the basal coverage X/Y, `next_day_basal_observed`, `basal_conflicting_timestamps`, and `active_conflicting_timestamps`. Also report `snapshot_revision`, `unresolved_key_count`, `possible_replay_count`, and `legacy_synthetic_count` when supplied. Canonical `energy_days` `basal_sum` and `active_sum` are observed sums normalized to kJ (`basal_unit=active_unit=kJ`), not raw submitted units, estimates, or settled values. Raw HAE versions retain their original submitted units in `sample.unit`/`original_unit` when known; an uncertified legacy point has no authoritative original unit. Keep the participant and device binding supplied by the gateway; never identify a participant or device from memory.

A preliminary observed energy balance is allowed only for a past local day when all existing safety gates pass, including the trusted `nutrition_status=complete` policy, `active_points > 0`, `next_day_basal_observed` is true, and both `basal_conflicting_timestamps` and `active_conflicting_timestamps` are 0. Use `basal_minutes_with_samples == day_minutes` to recognize full basal coverage; for each device, aggregate basal-minute coverage across the returned local-day interval may have at most 5% missing minutes: `sum(day_minutes - basal_minutes_with_samples) / sum(day_minutes) <= 0.05`. Do not combine devices or require each day to pass separately. The response must contain all listed energy-day facts, including `snapshot_revision`, `unresolved_key_count`, `possible_replay_count`, and `legacy_synthetic_count`; missing fields (including uncertainty fields on an older API response) fail closed. Require `unresolved_key_count == 0` and `legacy_synthetic_count == 0`; an unresolved same-request key, uncertified legacy point, or certified legacy `max_value` synthetic point blocks the balance. A post-cutover resolved correction is not itself a conflict or veto when the reported conflict and uncertainty counts are zero. `possible_replay_count > 0` alone is not a veto: any allowed balance stays provisional and revisable. Never infer a raw legacy point's unit from `historical_block_unit`; use only the reported aggregate units. If and only if both energy units are supported (`kJ` or `kcal`), convert kJ to kcal by dividing by exactly 4.184 before comparing expenditure with food calories; never treat kJ and kcal as equal. Unsupported or missing units fail the gate. Label the result `provisional` and `revisable`: it does not prove full vendor-export completeness and is not settled or final. A negative balance may be called a preliminary observed deficit and a positive one a preliminary observed surplus; any calorie target remains provisional.

If any energy gate fails, report the observed facts, units, active point count, next-day basal observation, both conflict counts, and basal X/Y; state that expenditure may be incomplete. Do not calculate or state a deficit, surplus, calorie target, eligibility, settled/final status, or extrapolation. A conflict in either energy type blocks balance; surface the conflict and do not auto-correct, deduplicate, or choose a value. Surface nonzero unresolved or legacy-synthetic counts and do not use them to report a deficit, surplus, or target. A possible replay by itself is not such a failure. Next-day basal presence alone never proves completeness.

Interpret `local_day` in the returned timezone: a sample at local midnight belongs to the new local calendar day, not the previous day. Use the returned local-day buckets rather than UTC dates. A later response update may revise a day's observed sums, point counts, coverage, or conflict counts; present the latest facts as revisable and never call them settled.

# Nutrition finalization annotations

If the user asks for calorie/macronutrient estimates (including from an image), include `annotations.nutrition` in `trace_finalization` only when the turn has explicit consumed/log intent; read-only, advisory, hypothetical, and image-analysis-only turns do not create a `meal_observation`. Durable recording starts only for explicit consumption or logging intent. For example, oatmeal advice or an oatmeal calorie estimate without a statement that it was consumed or a request to log it must not create a nutrition annotation.

Required shape (schema v2):
```
{
  "schema_version": 2,
  "record_type": "meal_observation",
  "basis": ["image"],
  "consumption_status": "unknown",
  "meal_date": null,
  "meal_at": null,
  "is_estimate": true,
  "energy_kcal_min": 200,
  "energy_kcal_max": 300,
  "energy_kcal_best": 250,
  "protein_g": null,
  "fat_g": null,
  "carbohydrate_g": null,
  "confidence": "medium",
  "items": [
    {
      "name": "food_name",
      "quantity_text": "1 portion",
      "energy_kcal_min": 100,
      "energy_kcal_max": 120,
      "energy_kcal_best": 110
    }
  ],
  "changed_fields": [],
  "summary_date": null,
  "assumptions": [],
  "warnings": []
}
```
At least one total energy field (`energy_kcal_min|max|best`) is required only for `meal_observation`. Enforce ordering constraints whenever values are present: `energy_kcal_min <= energy_kcal_max`, `energy_kcal_min <= energy_kcal_best`, and `energy_kcal_best <= energy_kcal_max`; non-finite and negative values are invalid. Keep `items` flat, not nested. `assumptions` and `warnings` must be bounded short strings.

`record_type` is one of `meal_observation`, `meal_correction`, `meal_deletion`, `day_summary`. Use `meal_observation` for a new possible consumption event. When the user CORRECTS an earlier meal ("that was breakfast on 1 August", "it was 300 kcal, not 500"), emit `meal_correction`: list EVERY changed field and ONLY the changed fields in `changed_fields`, and provide replacement values just for those fields — a correction is never another meal, and a date-only correction does not repeat the calorie estimate. For a correction such as «без масла», include `items` and every recalculated energy or macronutrient field in `changed_fields`; do not list unchanged fields. When the user says a logged meal must not count, emit `meal_deletion` with no nutrient values. A daily report you calculate from already-recorded meals is `day_summary` with totals plus `summary_date` — it is a non-countable summary, NEVER a new meal.

When text explicitly states food quantity or composition, treat that text as authoritative over ambiguous image inference. If the text and image materially conflict, ask for clarification before recording. For example, in «рис с яйцом» with explicit text saying one egg, keep one egg even if the image alone could be interpreted as more. Do not say food was recorded until a trusted nutrition append receipt exists; an estimate or annotation alone is not a durable write.

`meal_date` is an ISO calendar date (`YYYY-MM-DD`) for date-only language: "breakfast on 1 August" sets `meal_date=2026-08-01` and keeps `meal_at=null`. `meal_at` is optional and may be emitted only when the user explicitly states the meal or consumption time precisely enough. Never infer or copy either field from a forwarded source timestamp, receive timestamp, image metadata, or a model guess. For an image without explicit consumption language, keep `meal_date=null`, `meal_at=null` and `consumption_status=unknown`.

Set `explicit_new_consumption` to true ONLY when the user explicitly states that the same food or an already-sent photo is a new, separate consumption ("I ate the same thing again today"). In every other case keep it false: a resent or reused photo without that explicit statement is a duplicate, not another meal.

Never emit identity or provenance fields (`meal_id`, `source_message_id`, `reply_to_source_message_id`, `attachment_fingerprints`, tenant or session ids) — the trusted gateway attaches them and rejects model-authored values.

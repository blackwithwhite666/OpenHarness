# Nutrition persistence audit

`ohmo.evals.nutrition_persistence` grades explicitly reviewed goals against raw
persisted Honcho messages and Telegent's current canonical meal read. Assistant
proposals, trace annotations, assistant promises, episode completion, and generic
sync health are not persistence proof. Online dialogue review can use the native
subscription reviewer; preserve source provenance and ambiguity in the manifest.

## Manifest schema v1

Every goal binds reviewed episode IDs, Honcho tenant/workspace/peer/session,
gateway session, actual channel principal, eval workspace, inbound source,
logical turn, assistant operation, trace episode, canonical Telegent owner/login,
and the stable meal identity derived from tenant, principal, gateway session, and
source message. It also carries the meal's calendar timezone and reviewed
conversation start/as-of bounds. The tenant alias and channel sender principal
are distinct identities.

Consumed goals require a finite expected kcal value. The tolerance is immutable
at no more than 10%. Negative goals omit kcal. Every expectation includes its
origin (`explicit_fixture`, `reviewed_user_dialogue`, or
`frozen_photo_reference`), source identity, and optional uncertainty notes.
Goals are selected from reviewed dialogue independently of whether an assistant
nutrition annotation exists.

```json
{
  "schema_version": 1,
  "goals": [{
    "case_id": "tea-with-milk",
    "episode_ids": ["ohmo-gateway-reviewed-fixture"],
    "owner_id": "synthetic-owner",
    "principal_id": "telegram:synthetic-principal",
    "workspace_id": "workspace-id",
    "eval_workspace": "/private/evals",
    "peer_id": "ohmo",
    "canonical_owner_id": "synthetic-owner",
    "canonical_login": "synthetic",
    "session_id": "ohmo",
    "gateway_session_id": "gateway-session-1",
    "source_message_id": "synthetic-source-1",
    "logical_turn_id": "turn-1",
    "operation_id": "turn-1:assistant",
    "trace_episode_id": "ohmo-gateway-reviewed-fixture",
    "canonical_meal_id": "a2f1fd5693bd7e7268f4ac6a0f07d3c6",
    "meal_date": "2026-10-01",
    "meal_timezone": "Europe/Moscow",
    "trajectory_started_at": "2026-10-01T10:00:00+03:00",
    "trajectory_as_of": "2026-10-01T12:00:00+03:00",
    "expected_consumed": true,
    "expected_kcal": 25,
    "tolerance_fraction": 0.1,
    "expectation_origin": "reviewed_user_dialogue",
    "expectation_source": "review:tea-case-1",
    "review_notes": "User confirms milk, then says no sugar and asks to add the drink."
  }]
}
```

The example meal ID is the deterministic derivation for the example identity
fields. Do not force a goal when the reviewed user intent remains uncertain.

## Read-only export and live audit

Export selected episodes or a bounded session/principal range without
initializing `EvalStore` or writing to the source workspace:

```sh
ohmo evals nutrition-export --workspace /path/to/.ohmo \
  --episode-id episode-id --output /private/task/tea-dialogue.json
```

For a session range, supply `--session-id`, `--principal-id channel:sender`,
`--since`, and `--until` (at most 31 days). The export follows indexed JSONL
offsets read-only, verifies index identities, orders episodes/events
chronologically, and includes all inbound user turns, assistant updates, and the
gateway final/error. Intermediate clarification episodes need no nutrition
trace. A complete final reply and exact source binding are required.
For Camera turns, capture stores gateway-derived scalar source, principal, turn,
and assistant-operation provenance only after runtime authorization. Its initial
analysis prompt is captured separately against the positive native photo receipt;
the export shows the assistant's question without presenting the internal prompt
as a user statement, then binds the owner's retained reply to that same receipt.
Historical Camera episodes without recoverable recorded context remain unbound:
their serialized authority marker cannot recreate the original in-memory
authorization object.

After review, run the bounded persistence grader with either private snapshots
or the configured live readers:

```sh
OHMO_NUTRITION_AUDIT_HONCHO_TOKEN=... python -m ohmo.evals.nutrition_persistence \
  manifest.json private-report.json --dialogue-export /private/task/tea-dialogue.json \
  --honcho-base-url https://honcho.example --honcho-workspace workspace-id \
  --honcho-session ohmo --honcho-owner synthetic-owner \
  --since 2026-10-01T07:00:00+00:00 --until 2026-10-01T09:00:00+00:00 \
  --telegent-server wellness --telegent-login synthetic \
  --start 2026-09-30T21:00:00+00:00 --end 2026-10-01T09:00:00+00:00
```

The live Telegent reader selects only the named HTTP server from OpenHarness
settings and uses `McpClientManager`, preserving configured headers and OAuth
refresh. Telegent credentials are not copied into audit flags or output. The
returned interval and normalized login are checked against the request; both
canonical collections and every record needed for absence/current-state checks
are validated. The only token environment variable above is for Honcho.

Reports contain verdicts, persistence stage/reason, authoritative event IDs,
latest event ID, kcal, and honest evidence limitations. They are atomically
written mode `0600`; keep manifests, exports, and snapshots in private storage.
The audit is read-only and performs no live product mutations.

## Evidence boundary

A1 passes only when the effective validated Honcho state matches the reviewed
goal and the same stable source meal's latest effective event, owner, local day,
and kcal match Telegent's canonical current read. The deployed
`CanonicalMealRecord` has `latest_event_id` but no complete contributor
`event_ids` list. Reports disclose that limitation and do not invent IDs.
Same-source retries and no-op corrections do not advance the effective
revision; changed corrections use their validated masks. Retractions are
represented by the absence of a visible canonical meal, matching Telegent's
projection behavior.

Missing persisted rows with complete scoped reads fail positive goals; an
in-grace new effective event can be `PENDING`; overdue absence/staleness fails.
Unavailable, malformed, incomplete, ambiguous, or out-of-scope evidence is
`INCONCLUSIVE`. A2 remains `NOT_RUN` in the standalone audit.

Camera `Case.persistence_evidence` supplies the reviewed goal, raw Honcho and
Telegent snapshots, and exact full-dialogue export. Legacy `CommitEvent`
booleans and ledger flags cannot earn product A1 PASS. The full Case dialogue
must match the bound export; A2 receives full dialogue, while Sol receives only
the reference prefix and original image. Historical reports retain their
original model provenance.

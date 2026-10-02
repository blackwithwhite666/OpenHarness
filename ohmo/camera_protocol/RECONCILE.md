# Camera reconcile route

`POST /internal/v1/camera/reconcile` accepts the same authenticated, bounded
`multipart/form-data` request as candidate admission: exactly one each of
`request`, `manifest`, `producer`, and `image`. The body is limited by the
existing HTTP request bounds and all four parts are validated together.

The optional `X-Camera-Reconcile-Purpose` values are:

- `retire_ineligible` (the default): persist the existing ineligible retirement
  outcome without native delivery.
- `existing_outcome_only`: return only an exact existing durable ACK; this is a
  read-only lookup.
- `prove_not_admitted`: return the read-only
  [negative proof envelope](prove_not_admitted.schema.json) only when the
  complete schema-2 journal for the same session lineage proves that the exact
  candidate has no retained admission tombstone and no recorded owner conflicts
  with the original sequence.

The negative proof relies on the local journal integrity premise: schema-2
attempt tombstones are retained, and the attempt plus its ACK is durably saved
before native dispatch is scheduled. A missing, corrupt, unsupported, or
mismatched journal never proves non-admission. This protocol does not detect
arbitrary manual restoration, copying, or tampering of product state.

# Voice-agent pilot and benchmark kit

This directory is credential-free and offline by design. `managed_pilot.json`
is the source-of-truth configuration to reproduce the current custom agent in
Sarvam's managed Voice Agents dashboard. `compliance_residency.json` is a
sign-off worksheet, and `corpus.json` contains 25 smoke scenarios plus 100- and
200-call expansion templates.

## Data minimization and safety

- Keep `RETAIN_AUDIO=false`. The corpus and runner store IDs, scripts, expected
  entities, telemetry, exports, and human ratings only.
- Recording is off by default in the managed spec. Do not turn it on without an
  approved purpose, consent text, retention period, access policy, and owner.
- Never put API keys, auth tokens, phone numbers, webhook secrets, or exported
  credentials in this directory.
- The runner never invokes Twilio, Exotel, or Sarvam. Even `--execute` only
  marks a schedule as approved after an exact destination confirmation. A human
  operator must place calls in the appropriate dashboard or controlled system.

## Validate and prepare

Run from the repository root:

```powershell
python -m backend.benchmarking validate
python -m backend.benchmarking prepare --run-dir storage\benchmarks\pilot-001 --run-id pilot-001 --calls-per-cell 25
```

Use `--calls-per-cell 25` for 100 total calls or `50` for 200 total calls.

Dry-run registration:

```powershell
python -m backend.benchmarking register --run-dir storage\benchmarks\pilot-001 --cell managed+twilio --destination TEST_DESTINATION
```

Authorize an externally executed schedule only after checking the destination:

```powershell
python -m backend.benchmarking register --run-dir storage\benchmarks\pilot-001 --cell managed+twilio --destination TEST_DESTINATION --execute --confirm-destination TEST_DESTINATION
```

This command still places no call and incurs no charge.

## Managed dashboard mapping

Sarvam managed Voice Agents setup is dashboard-based; this repository cannot
provision it. Copy each value from `managed_pilot.json` into a separately
created managed agent and verify the final dashboard configuration against the
JSON before testing.

For managed + Twilio:

1. Create/select the Sarvam managed agent and copy the identity, explicit AI
   greeting, prompt rules, fields, consent flow, languages, speaker, pace,
   interruption settings, and recording-off setting.
2. Add a Twilio channel in the Sarvam dashboard.
3. Configure Twilio to use the Sarvam webhook
   `https://apps.sarvam.ai/api/app-runtime/channels/twilio`.
4. Add provider credentials only in the authorized dashboard/secret store.
5. Verify status hooks and obtain a managed call/analytics export.

For managed + Exotel:

1. Create/select an equivalent Sarvam managed agent using the same spec.
2. Add an Exotel Voicebot channel and use endpoint
   `https://apps.sarvam.ai/api/app-runtime/channels/exotel`.
3. Use Exotel's Mumbai base URL `https://api.in.exotel.com` where the Exotel
   setup requests a base/API region.
4. Add provider credentials only in the authorized dashboard/secret store.
5. Verify status hooks and obtain a managed call/analytics export.

Do not infer data residency from the Mumbai API hostname. Complete
`compliance_residency.json` from contracts and provider documentation; leave
unknowns as `unverified`.

## Import evidence and report

Managed CSV/JSON exports must first be mapped to canonical columns accepted by
`backend.benchmarking.CANONICAL_EVIDENCE_FIELDS`. Every row needs `call_id`.

```powershell
python -m backend.benchmarking import-managed --run-dir storage\benchmarks\pilot-001 --cell managed+twilio --input exports\managed-twilio.csv
python -m backend.benchmarking import-managed --run-dir storage\benchmarks\pilot-001 --cell managed+exotel --input exports\managed-exotel.json
python -m backend.benchmarking import-custom --run-dir storage\benchmarks\pilot-001 --cell custom+twilio --database data\sarvam_leads.db
python -m backend.benchmarking import-custom --run-dir storage\benchmarks\pilot-001 --cell custom+exotel --database data\sarvam_leads.db
python -m backend.benchmarking import-ratings --run-dir storage\benchmarks\pilot-001 --cell managed+twilio --input ratings\managed-twilio.csv
python -m backend.benchmarking report --run-dir storage\benchmarks\pilot-001
```

The report writes `report.json` and `report.csv`. It calculates p50/p95
speech-end-to-first-audio, barge-in p95, false endpoint rate, one-clause
language switching, entity/task accuracy, Gujarati native-speaker rating, cost
per completed call, and residency status. It applies the plan's exact numeric
gates. Missing real evidence produces `insufficient_evidence`; the runner never
creates synthetic results.

## Required external prerequisites

Actual managed deployment and real four-cell calls remain blocked until all of
the following are supplied and approved outside source control:

- Sarvam managed Voice Agents account/access and a provisioned agent.
- Paid/provisioned Twilio and Exotel accounts, phone numbers/caller IDs,
  destination allow-listing, credentials, and sufficient balance.
- Public production callbacks for the custom cells and provider-side webhook
  validation.
- Confirmation that managed and custom features/models are available in the
  selected accounts and region.
- Native Gujarati, Hindi, English, Gujlish, and Hinglish speakers; handsets and
  carrier conditions for manual scripts, interruptions, and quality ratings.
- Approved destinations and explicit authorization for chargeable calls.
- Current pricing exports/invoices for cost evidence.
- Completed DPA, subprocessors, processing/media/recording regions, retention,
  deletion SLA/API, consent, owner, and sign-off worksheet fields.

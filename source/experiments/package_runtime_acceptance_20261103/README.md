# Whole-package runtime evidence

This experiment measures a complete public or roadshow package, with no sample
limit or selection file, new preprocessing outputs and new default-expert model
forwards. A package passes the measured-host time bound only if its own root wall
time is at most 10,800 seconds. Times are never added to judge the three-hour limit.

The current pipeline writes **two separate products**: the complete deterministic
rules scorecards/evidence/audit, and fresh default-expert candidates with fresh
current-input priors. Fresh model candidates are not yet merged into those
scorecards. This is not final integrated delivery acceptance or a new F1 test.
The host is shared A100 40GB, not the specified A10. The existing Qwen service stays
running, GPU1 is not used, and no optional LLM or new localization network runs.

`package_benchmark.py prepare` freezes a standard-library engine snapshot and the
existing portable-expert source, validates the old provenance lock and inventories
six public P4 packages plus the largest extracted roadshow package by bytes for
AIA/EPB/ULF. It does not infer severity from anonymous names. The nine packages
contain 73,112,250,713 input bytes. AIA P4 and CME P4 alone contain 38,352,996,634
bytes. Raw hashing, full decoding and final hashing cause several complete logical
reads; `/proc/self/io` records actual logical and physical read counters. OS page
cache is not flushed and this limitation is retained.

Each package root timer includes source inventory, full rule evaluation/report and
evidence, rule audit, cold model loading, fresh image/CSV preprocessing, all present
sample forwards, candidate export, source metadata checks and a final rules audit.
Nested stage times are recorded; they must not be summed because outer stages
already include inner stages. The final portable source-byte verification is timed.
Original missing inputs remain abstentions and must account for all manifest rows.
The independent auditor recomputes the frozen thresholds and model unions, checks
all-label/all-existing-data coverage, batch counts and source-artifact hashes, and
keeps model candidates ineligible for scientific scoring.

```bash
PY=/staff/tangjiarui/OPC/.venv-image-models/bin/python
ROOT=/staff/tangjiarui/OPC/package_runtime_acceptance_20261103
$PY "$ROOT/package_benchmark.py" prepare
$PY "$ROOT/package_benchmark.py" run --plan "$ROOT/plan_v1.json" \
  --package AIA_AR_sample_k0msxmcgew --device cuda:0
$PY "$ROOT/package_benchmark.py" run --plan "$ROOT/plan_v1.json" \
  --package CME_C2_sample_di7yj9fvs2 --device cpu
$PY "$ROOT/audit_benchmark.py" --plan "$ROOT/plan_v1.json"
```

Output paths are new-only. The runner overrides only the portable module's allowed
output root in its own process; the frozen source, models, thresholds and
preprocessing functions remain unchanged. The override and source hashes are
recorded. Batch is 4, CUDA allocator budget 4GiB. Each package starts a new process.

`external_supervisor.py` is the authoritative scheduling and timing wrapper. It
adopted the original AIA/CME workers without restarting them, measures those from
their original Linux process start ticks to a conservatively observed exit
interval, and measures subsequent workers externally from immediately before
process launch until process exit. A dedicated watchdog terminates the complete
worker process tree at 10,800 seconds, preserving a failed/incomplete result.
Its new runner file changes only the CPU lock-slot expression to allow two CPU
lanes; the original runner and frozen model/preprocessing sources stay unchanged.
At most one GPU worker and two CPU workers run concurrently. GPU jobs bind to
physical GPU0 by UUID, and live NVML observations check the actual worker UUID and
4GiB process-memory bound. CPU lanes hide all CUDA devices. The shared GPU0 Qwen
service remains running, and GPU1 is never used.

Use `external_supervisor_state_v2.json` for live state, not the retired original
queue-state files. After completion, use `external_supervisor_complete_v2.json`;
the last periodic state frame can predate the final exit by less than two seconds.
Each complete package has an `external_runtime_receipt_v2.json`
and `independent_audit_external_v2.json`; the latter binds the external receipt
hash to the fresh-coverage audit. The original internal root timer and legacy
portable `passed_research_smoke` / `full_dataset_evaluation=false` fields are
retained unchanged. The external wrapper and independent full-coverage audit
establish the measured scope; those legacy markers are not relabeled.

All nine complete packages passed the measured-host bound and independent audit:

| Package | Labels | Fresh samples | Missing retained | External upper seconds |
| --- | ---: | ---: | ---: | ---: |
| Public AIA P4 | 3150 | 3117 | 33 | 1541.988 |
| Public HMI P4 | 3017 | 2986 | 31 | 1417.425 |
| Public CME P4 | 10790 | 10685 | 105 | 4022.345 |
| Public EPB P4 | 1350 | 1223 | 127 | 94.760 |
| Public DMSP P4 | 286 | 255 | 31 | 12.776 |
| Public ULF P4 | 6120 | 6034 | 86 | 340.496 |
| Roadshow AIA (tier unknown) | 1838 | 1817 | 21 | 885.627 |
| Roadshow EPB (tier unknown) | 788 | 788 | 0 | 58.820 |
| Roadshow ULF (tier unknown) | 3570 | 3559 | 11 | 186.941 |

Acceptance checks each receipt's external bound, null timeout/guard failure,
independent audit pass, and exact observed GPU UUID for GPU jobs together. The
top-level supervisor `complete` field alone is not a success criterion. All five
GPU jobs were observed on physical GPU0. Adopted AIA/CME workers have no retrievable
original-parent wait exit code, so retain `exit_code=null` and their conservative
exit intervals; successful complete products and independent audits are retained.
All nine worker PIDs and experiment supervisors exited with no matching benchmark
commands left behind; the original Qwen service remained running.

All retain missing-input abstentions, use zero cached probabilities, and produce
separate rule scorecards and model candidates. These are A100-host measurements,
not A10 acceptance, all-30/all-12 coverage, an integrated model report, or accuracy/F1.
`collect_runtime_summaries.py` exports only small metadata and source hashes, and
`write_runtime_report.py` turns that JSON into the Chinese evidence report.

Historical `full-rule-release_runtime.json` measured all 30 packages in 340.80s plus
63.86s audit; `full-rule-final_runtime.json` measured 387.39s plus 49.49s audit.
Those were rule-only A100-host CPU/storage measurements, not complete default-model
timing. The old 64-image VLM throughput experiment and its extrapolated package
budget cannot establish full-package or A10 compliance.

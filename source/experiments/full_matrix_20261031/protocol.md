# Full inventory evaluation, locked before inference

User explicitly requests all six previously reported routes on all 30 public
packages, separately by family and P0/P1/P2/P3/P4. No subset is a substitute.
Scope: complete frozen 123565-record manifest, 24713 bases. Verify these
counts against current files. Original data and model weights stay remote.

Six headline routes: blind deterministic structural/fixed-field checks;
frozen 8B QLoRA; frozen directly supervised student; frozen fine-tuned-teacher
distilled student; frozen supervised pruned/recovered student; structural
rules plus frozen QLoRA gate. Preserve each of three existing student seeds,
report mean and individual results. Existing gate [1.01,0.9] is frozen from
prior validation; do not retune using full-inventory outcomes.

These checkpoints were trained on EPB/DMSP. New-family inference is a domain
transfer test, not proof of full-family training. Full inventory includes
training observations: identify prior train/validation/test bases and provide
separate held-out/unseen tables. Full-inventory figures are descriptive,
not generalization estimates. Keep earlier local reports explicitly scoped.

Use the same four coarse output groups: NONE, spatial GEOMETRY (including
coverage), COUNT (omission/spurious), LABEL_OR_METADATA (including temporal
and numeric parameter errors). Preserve mixed defects as three target bits,
never drop them or force one gold class. Existing single-class models emit
one bit, so incomplete recall is visible. Report per-class and fine-code
technical reference counts; coarse F1 is not the official location score.

Truth uses independently implemented, stable-identity P0 annotation comparison
and existing audit evidence, not the evaluated blind-rule outputs. Same-image
geometry tolerance .5 native pixel. AIA/HMI image differences require pixel
validated registration, >=.98 correlation; otherwise mark geometry truth
unknown, never infer NONE. Existing audited transforms are reusable only for
exact matching input hashes. Use per-class validity masks, and report fully
scorable and partially scorable sample counts. Technical differences from P0
are not independently adjudicated scientific errors. Exclude unrelated
provenance/version/reference-path changes from defect truth. Null geometry,
ambiguous object identity and unavailable image evidence remain explicit.

All raw-file-missing records remain in inventory. Missing data are evaluated
separately as L0 structural coverage; visual inference cannot assess absent
observations. Count model abstentions/failures and retain known-class false
negatives; do not silently remove failed requests from metrics. P0 has no
injected positives by reference definition: report F1 as NA when support is
zero, together with false positives and FPR. Preserve baseline P0/raw-body
contradictions separately rather than calling them injected corruption.

View: raw observation left and current annotation right; no P0/tier/package
names in model input. Use actual dimensions for all five image families.
ULF is CSV: render current time-series and current event windows, include
current declared parameters in text. ULF adaptation is explicit, not pretend
native image-model support. Features retain 9 checkpoint input dimensions;
EPB/DMSP encoding remains backward compatible. Validate old held-out examples
before full QLoRA inference. Cache only byte-identical views AND prompts;
retain per-record predictions and exact input hashes proving reuse.

Outputs: 30-package coverage inventory; 31-column F1 overview; expanded
151-column table (scheme +30x F1/precision/recall/FPR/n), long per-package,
per-class, per-seed and held-out tables; raw prediction logs and independent
metric reconstruction. Include runtime per package, missing/unknown counts.
No claim that 8B weights alone meet <=20GB Docker or A100 time passes A10.

Queue: CPU preparation may overlap current initial architecture sweep;
GPU1 full-inventory inference waits for that sweep. Defer tuning-only queue.
Preserve GPU0/8001 service. Checkpoint per batch, resume after interruption,
use bounded batches with OOM backoff. No automatic formal scorer switch.

Input adaptation amendment before full-table reporting: retain exact old
EPB/DMSP prompts. For other families use concise current-only, family-specific
metadata summaries, rather than truncating arbitrary JSON at 6500 characters.
AIA/HMI include all categories and actual/declared dimensions; CME includes
all event time/angular declarations; ULF includes current CSV statistics,
dateObs/declared window/parameters and all current event records. Record
actual per-row input hashes/profile in SQLite, so earlier pilot long prompts
cannot be mixed into reported compact-profile predictions. Initial QLoRA
compatibility reproduced all 170 old test predictions and all nine student
checkpoints exactly. Try batch32 with OOM backoff; report measured throughput.

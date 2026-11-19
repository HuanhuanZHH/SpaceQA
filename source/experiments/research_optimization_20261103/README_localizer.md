# Independent image-only localization pilot

This optional research experiment does not change the frozen/default inference route.
It trains a tiny four-head U-Net on **original RGB images only**, using existing P0
annotation regions as weak targets. Family identity selects an output channel; no
annotation, metadata field vector, composite preview, tier identity or paired P0
image is a model input. CME and ULF have no verified pixel-region targets and remain
explicitly not applicable to this experiment; existing deterministic checks apply.

The existing 20261031 grouped split is preserved, including same-source tiers and
content duplicates. Incomplete/nonfinite/out-of-image geometry is excluded rather
than silently treated as background. The train inventory is audited before GPU
training. **Both candidate models finish all four-family training before validation
or test files are opened.** P0 validation positive-mask IoU freezes family thresholds;
test results never select model weights or thresholds.

Two otherwise identical models compare BCE with foreground-weighted BCE + Dice.
Each uses at most three epochs, per-family sample caps, batch 8 and seed 20261103.
Training includes a full-image view and a train-only foreground-biased crop. At
inference a fixed, annotation-independent native-image grid supplies ROI views,
protecting AIA/HMI detail lost by old 384px thumbnails. Global and ROI probabilities
are combined equally. The global-only result is also reported without retraining.
This small pilot cannot establish full-scale convergence or official scientific F1.

Run on the server, with new output paths only:

```bash
/staff/tangjiarui/OPC/.venv-image-models/bin/python \
  /staff/tangjiarui/OPC/research_optimization_20261103/localizer/spatial_localizer.py run \
  --out /staff/tangjiarui/OPC/research_optimization_20261103/localizer/pilot_v2 \
  --train-cap 384 --eval-cap 64 --tier-cap 4 --epochs 3 --batch 8 \
  --device 0 --budget-seconds 1800
```

GPU0 is shared with the existing model service and bounded replay. The allocator is
capped at 9% of a 40GB A100 (below 4GB); measured allocated/reserved peaks and runtime
are saved. GPU1 is not used. The program stops if the controlled 30-minute pilot
budget is reached. A failure cannot satisfy the training-completion barrier.

Outputs include train inventory/audit/source hashes, selected samples and ROI boxes,
checkpoint hashes, losses/family coverage, frozen validation thresholds, P0 per-family
mask IoU/Dice (positive and empty separately), and P1–P4 test-group instance samples.
Predicted components are generated independently before annotation matching. Their
bounding boxes are mapped back to native image coordinates. References explicitly
set `class_known=false`, `coverage_complete=false`, and
`confidence_calibrated=false`; unmatched annotations abstain. Candidate issues are
pending review and cannot penalize scientific scores. Connected components can
merge nearby objects. P0 mask reproduction is **not** official localized defect F1
or recall, and no P0 mask is transferred to changed contaminated observations.
Instance components are extracted on a 1024px fusion canvas and mapped back to
native pixels; the 256px masks remain a separate weak-supervision metric.

The completed `pilot_v1` used the frozen `spatial_localizer_pilot_v1.py`, SHA256
`de366154799fff069dbc618e470de0b3d355e7cb253e89c63d1a8ad6e86d455b`, retained
on the server as `localizer/spatial_localizer.py`. It selected 1,346 source samples,
but its replacement sampler visited 1,337 unique samples across 7,494 examples per
model. The current v2 sampler guarantees every selected view is visited, then
oversamples smaller families. Current v2 also rejects an existing output path,
source/output overlap, unsupported annotation schemas and actual byte-hash changes.
These corrections do not change historical v1 outputs. V1 source SHA matched all
chosen images; validation source integrity auditing began after validation started,
so it does not establish an immutable snapshot from the first read.

The core-reference v2 export preserves v1 predictions and applies the explicit core
schema (`image_pixels`, native `image_size`, actual-image SHA as `frame_id`), after
image **and** label byte-hash checks. Existing exports cannot be overwritten. Unknown
classes, incomplete detections and unverified negative evidence remain explicit.

## Measured pilot v1 results (not official defect F1)

All four supported families were trained before evaluation. The original run used
241 validation and 235 test P0 observations; every route also predicted 62 available
contaminated test-group observations across P1–P4. CME/ULF are explicitly NA here.
Results are capped/stratified weak-mask reproduction, not all-30-package evaluation.

| Family | Main val positive IoU | Main test positive IoU | BCE test positive IoU | Main global-only test positive IoU | Main test empty-mask FPR |
|---|---:|---:|---:|---:|---:|
| AIA | 28.98% | 25.19% | 16.97% | 27.52% | 100% (16 empty) |
| HMI | 20.80% | 21.23% | 17.85% | 25.28% | 100% (16 empty) |
| EPB | 42.18% | 29.85% | 5.79% | 29.79% | 100% (1 empty) |
| DMSP | 47.92% | 43.80% | 35.16% | 43.75% | 46.67% (15 empty) |

“Main” is weighted BCE + Dice. Global-only uses the same previously frozen
global+ROI validation threshold, so this is a fixed-threshold ablation rather than
independently optimized model comparison. ROI did not establish a P1 improvement.
Empty-mask FPR concerns P0 annotation-mask reproduction, not verified scientific
defect false alarms. The high values prevent promotion to the default verdict route.

The original pipeline took 736.94 seconds, including original-image preparation;
the two three-epoch training loops took 16.18 and 13.48 seconds. Each model has
122,844 parameters and a ~0.61 MB checkpoint. Peak training allocation was 463,697,920
bytes, reservation 513,802,240 bytes, on shared GPU0. These measurements do not
establish whole-dataset official runtime or container size. A separate frozen
validation global-only diagnostic took another 98.71 seconds, without retraining.

The core integration smoke test covered 80 cases (cap two per model/family/tier):
19 pending candidates, 1,069 abstentions, zero spurious-annotation or class-error
candidates. All 594 exported prediction rows passed actual image/label hash checks.
Official localized F1/recall remain unknown. Existing defaults are unchanged.

```bash
python -m pytest tests/test_spatial_localizer_pilot.py -q
```

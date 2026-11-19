# Spatial research protocol, locked before execution (2026-10-31)

Objective: improve subtle P1 and P2-P4 label-defect detection under the official
single-package <=3h and final image <=20GB constraints. Preserve the released
rule evaluator and the original port-8001 model service.

H1: training an image-only dense predictor on training-group P0 masks provides
more spatial information than the previous low-resolution composite classifier.
Compare a small U-Net and frozen DINOv2-small/14 plus a trainable spatial head.
H2: residual features between predicted masks and current annotations improve
multi-label geometry/omission/spurious classification over annotation-only
features, at a validation-calibrated false-positive budget.
H3: native-detail crops and registration-aware clean comparison can preserve
small offsets. Evaluate separately from the blind image model; never merge the
extra information of a clean reference into a blind-accuracy claim.

Primary pilot: EPB and DMSP. Inherit the existing frozen observation-day,
base-sample and duplicate-content split. Include P0-P4 and mixed defects for
eligible base samples; do not force one label per sample. Excluded bases cannot
be silently assigned to training: record their count. Existing test sets have
been exposed: all results are exploratory/regression, not fresh confirmation.

Supervision: training P0 masks only for the dense predictors; geometric and
count labels are technical P0-derived differences on identical raw images, not
expert scientific truth. No test P0 masks enter model fitting, calibration,
feature construction, or prediction. P0 is allowed only in evaluation truth.
Do not infer missing/spurious annotations from missing raw files.

Fixed training: seed 20261031; image 224x224; U-Net widths 16/32/64/128,
24 epochs; DINOv2 frozen small backbone with 64-channel spatial decoder,
60 decoder epochs. Choose checkpoint by validation P0 mask Dice. No heavy VLM
fine-tuning during this bounded round. Limit GPU1 job to 90 minutes, preserve
GPU0. Each stage records its actual completion/failure independently.

Detector: ExtraTrees per defect, 160 trees, min leaf 3; train only training
groups. Compare annotation-only features, annotation plus U-Net residuals, and
annotation plus DINO residuals. Calibrate per-class thresholds using P1
validation only, maximizing multi-label micro F1 with <=1% clean-sample FPR.
Threshold may close the model branch when evidence is insufficient. Also show
ungated results, family/tier metrics and positive-support counts. Do not select
a winner from test metrics.

Localization: return residual-map regions as proposals; report predicted-clean
mask Dice and overlap with P0-derived annotation-difference masks on true
positive samples. This is a technical proxy, not official fine-class/location
credit. Evaluate image permutation controls with annotations unchanged. No
candidate automatically enters formal scoring.

Pre-evaluation amendment: preparation found zero P1 validation omissions.
Keep primary P1-only calibration unchanged. Also report a separately named
multi-tier validation calibration: optimize micro F1 over all validation tiers
while preserving the <=1% P1-clean FPR constraint. It must not use test results.
Do not infer omission performance from a validation cohort with no positives.

Research: verify official repositories/model cards for I-JEPA, V-JEPA 2,
DINOv2, DINOv3, SAM2 and efficient anomaly detection; save access dates,
source URLs, licenses/access constraints and applicability. 'jev' is interpreted
provisionally as JEPA. Do not describe untested models as benchmarked.

Continuity: a bounded detached remote sequential runner writes state, stage
logs and final metrics. No unsupported promise of automatic future LLM wakeups.
Large data and weights remain on icarus0. Download only public DINOv2-small
weights in this round, with hashes recorded; never expose credentials.

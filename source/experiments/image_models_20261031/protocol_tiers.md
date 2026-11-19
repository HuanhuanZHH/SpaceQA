# Four-tier selection amendment, before new defect evaluation

2026-10-31. User requests separate P1/P2/P3/P4 results and a configuration
strong across all tiers. Add a third, explicitly distinct calibration:
maximize the equal-weight mean of validation micro-F1 across P1/P2/P3/P4,
subject to <=1% joint class false-positive rate on spatially clean P1
validation samples. Ties: higher worst-tier F1, lower P1 clean FPR, then
more conservative threshold sum. Grid unchanged: .5/.7/.85/.95/.99/1.01.
Tier labels are never model inputs and never select inference-time models.
Keep original P1-only and pooled multi-tier calibrations for comparison.

Choose configurations using validation mean F1, worst-tier F1, FPR, then
smaller feature count/weights. Report test tiers separately as exposed
exploratory regression; do not choose by test ranking. No claim of universal
dominance if per-tier results trade off. Report support and three spatial
classes so low P1 support remains visible. P1 validation has no omission
positives; this does not establish omission performance.

Secondary fixed instance features (17 numbers): predicted count, annotation
count, unmatched predicted and annotation counts at IoU .5 and .25; min/mean/
max annotation-best IoU; min/mean/max prediction-best IoU; mean/max proposal
confidence; mean best-match normalized center L2 distance, mean absolute log
matched area ratio, count discrepancy. Proposals use fixed confidence .25
for these features (generation retains >=.05); greedy one-to-one matching
for separate instance precision/recall/AP50 at fixed .25 confidence. P0
test reference boxes are used for reporting only, never inference features.

After initial evaluation, take top two adapted architectures by primary
tier-balanced validation results (exclude annotation-only and zero-shot).
For each: 48 epochs; 24 epochs with detail objective; 24 epochs at seeds
20261101 and 20261102, in addition to existing 24/seed20261031 fit. Semantic
detail objective adds .25 times BCE on 3x3 morphological gradient of
predicted probability versus target gradient. YOLO detail uses box=10 and
dfl=2; segmentation also mask_ratio=2. Other settings frozen. Selection
remains validation-only. Follow-up queue is sequential on GPU1, bounded to
150 minutes, and waits for initial GPU queue completion.

Implementation note: Ultralytics resets CUDA_VISIBLE_DEVICES from its device
argument. Initial zero-shot and YOLO11 detect inadvertently used physical
GPU0. Correct future calls to physical device1; preserve port8001 service.
Record this deviation; do not compare contention-affected timing as isolated.

Also recalibrate saved probabilities from prior U-Net224, DINO224/448,
cross-fitted U-Net224 and I-JEPA224 with the new validation-only gates.
Require exact row-ID and target equality to this frozen manifest. These are
descriptive prior-architecture controls in the final validation ranking;
they do not enter the two-new-architecture tuning budget. Saved primary
probabilities suffice for class metrics; no new permutation/localization
claim is inferred for these recalibrated controls.

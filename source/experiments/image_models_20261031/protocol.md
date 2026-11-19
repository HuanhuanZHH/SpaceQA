# YOLO and U-Net family protocol, locked before fitting

2026-10-31. Extend the completed spatial pilot, without changing official rules
or the GPU0/port8001 service. Run only on GPU1. Data and weights stay remote.

Compare YOLO11n detection, YOLO11n segmentation, YOLOv8n segmentation,
Attention U-Net, U-Net++ (nested skip connections), pretrained ResNet18 U-Net,
and pretrained DeepLabV3 MobileNetV3. Include the existing U-Net448 and the
annotation-only learned forest as controls. Detector and mask tasks are
reported separately; rectangular predictions cannot be called pixel masks.

Reuse exact frozen EPB/DMSP base groups: 1623 bases (958/495/170), 7799 tier
records with identical-image truth. P0 training annotations alone supervise
the vision networks. Test P0 labels never enter training, calibration or
feature construction. All tiers inherit their base's frozen split. These
tests have been exposed: results are exploratory regression, not new blind
confirmation or official scientific ground truth. Preserve missing-file counts.

All new models use 448px images, seed20261031, 24 training epochs. Segmentation
models: batch16, AdamW, BCE+Dice, widths16/32/64/128 for attention/nested U-Nets;
lr1e-3 for scratch networks and lr1e-4 for pretrained encoders; BF16 autocast.
Select segmentation checkpoint by P0 validation mask Dice. YOLO: pretrained
public COCO weights, single foreground class, imgsz448, batch16, workers4,
24epochs, deterministic seed, no mosaic/mixup/copy-paste/rotation/flips to avoid
unvalidated physical transforms; choose framework best validation mAP model.
Models pretrained on COCO do not already know these scientific classes.
YOLO masks or boxes are converted to confidence-weighted foreground maps.
Record both mask Dice and instance/bbox matching on held-out clean samples.

Primary defect evaluation: same 21 annotation features +14 spatial residual
features, same three ExtraTrees detectors (160trees,minleaf3), same P1-only
and multi-tier validation gate grids as earlier. Calibrate only on validation,
with <=1% P1-clean FPR. Report geometry, omission, spurious supports, F1,
precision/recall, per-family/tier results and residual localization IoU.
Retain image permutation controls. Never select the reported winner by test.

Secondary instance-aware evaluation: add a fixed, documented vector of
current-annotation to independently predicted-instance matching features.
Apply it to every model including baseline connected-component instances;
keep these results distinct from the primary fixed-feature comparison.
No test tuning of IoU/confidence thresholds. Minimum proposed-object
confidence0.05, maximum100 instances, match IoU0.5 (plus descriptive0.25).

Report original versus adapted YOLO11n-seg class-agnostic proposal performance;
zero-shot COCO classes are not interpreted as scientific categories.
All candidates remain review-only and outside official scoring.

Bound the detached sequential GPU1 queue to150minutes; per-stage timeouts and
state files retain failures. Dependency additions go in an isolated venv that
reads existing Torch libraries; do not upgrade the Qwen service environment.
Measure checkpoint sizes, runtime and peak memory. Final A10/20GB Docker
acceptance is not inferred from an A100 pilot or from weights alone.

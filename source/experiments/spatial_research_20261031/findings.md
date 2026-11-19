# Current understanding

Rule and trusted-reference evaluation is the released path. Earlier P1 pilot
F1=0.46154 and P2-P4=0.76582 are coarse, exposed-test, technical proxy results,
not official scientific/location scores. The latter uses a learned feature
classifier. Spatial experiments must show image dependence and localization.

The available P1 localization proposals are not independently expert reviewed.
AIA/HMI synchronized image-label transforms need validated registration before
their geometric differences can be declared defects. JEPA is a representation
learning family, not an off-the-shelf label-quality classifier.

Completed 2026-10-31: U-Net and DINOv2 at 224/448, three group-crossfit U-Nets,
and a frozen I-JEPA encoder with trained dense head. 386 artifact integrity
checks pass. Weights/data stayed remote. See the user report and frozen JSONs.

On the newly separated three-spatial-class proxy, validation-selected
multi-tier thresholds give P2-P4 micro F1: annotation-only .4751, U-Net448
.5638, DINO448 .5291, crossfit U-Net224 .5397, I-JEPA224 .5055. These are
exposed test descriptions, not selection criteria or official scores. P1
geometry remains undetected and omissions remain undetected. Do not compare
these with the earlier coarse four-way F1 as though denominators match.

21/68 earlier P1 technical geometric proposals lose all polygon-raster
differences at global224; native ROI rendering reduces this to 1/68. This is
input-detail evidence, not demonstrated scientific accuracy. Prioritize ROI
plus global proposals and real subtle localization supervision.

I-JEPA has 630.76M backbone parameters and 2.523GB weights versus DINO22.06M
and 88.25MB. It did not beat the small U-Net in this fixed pilot. V-JEPA2.1
has an 80M distilled video backbone but was surveyed, not benchmarked.

U-Net448 full P1 candidate passes (decode, mask, detector, evidence included)
took EPB45.39s/1350 slots, DMSP5.96s/286 slots on A100. These exclude formal
rules/scoring/audit and do not certify A10 or final20GB Docker acceptance.

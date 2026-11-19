# Follow-up protocol: resolution ablation (before new fits)

Run the same U-Net and frozen DINOv2-small heads at 448 rather than 224 pixels.
Preserve all groups, real samples, targets, seeds, objectives, epochs, calibration
grids and model selection. Start only after the first runner exits. New output
directory: runs/spatial-research-448-20261031. Do not overwrite the first run.

Motivation from a separate technical resolution audit (not new held-out model
metrics): 21/68 earlier P1 geometric proposals have identical polygon rasters
at 224px global resolution, versus 11/68 at 448px and 1/68 for 224px local crops.
This tests global resolution alone. Native ROI model training remains a later
experiment, not something this resolution experiment has already implemented.

Exploratory regression only: prior test cohorts are exposed. Select the route
using validation F1/FPR, then validation mask Dice and smaller footprint.
Count image-permutation changes and localization residual IoU. Preserve the
formal rule service. Bounded GPU1 runtime <=60min. No expert-truth claim.

# Cross-fit residual detector, before fits

Hypothesis: in-sample training masks make residuals artificially clean, so a
second-stage defect classifier learns a different residual distribution from
unseen validation/test images. Generate training residuals out of fold.

Use 224px U-Net because its validation clean Dice (0.5665) exceeds 448px U-Net
(0.5409); this selection does not use test metrics. Split the 958 training
bases into 3 GroupKFold folds by existing observation-day/duplicate group. For
each fold, train the same U-Net for exactly 24 epochs on the other folds. Do
not select epochs on the held-out fold. Predict raw-only masks for its held-out
training bases. Original validation/test predictions stay unchanged. No new
training/test records and no test clean-mask access in fitting.

Train/calibrate the same forest detectors on OOF residuals. Reuse the fixed
P1-only and multi-tier validation protocols and image-permutation controls.
This is an exploratory regression result, not fresh confirmation. Maximum
cross-fit GPU stage 15 minutes. Final inference still uses one full-training
U-Net: cross-fit models need not be shipped.

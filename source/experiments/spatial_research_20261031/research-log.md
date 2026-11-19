# Research log

2026-10-31: Read prior results. The previous student's predictions barely
changed under image permutation, so prioritize dense image-only predictions
and retain annotation-only controls. Protocol written before new experiments.
No changes to the official evaluator are authorized by an unverified metric.

Completed bounded round: 8 dense-model fits, annotation-only controls,
P1-only versus multi-tier validation gates, image permutation, technical
localization IoU and complete two-package spatial candidate runtime.
GPU0 service preserved. Corrected repeated lazy NPZ decompression during
evaluation; original failed/interrupted stage retained in the state log.
Model downloads retried through a mirror; I-JEPA weights matched official
LFS SHA256. Frozen probabilities rounded to12 decimals for stable gates.
386 independent artifact/metric reconstruction checks passed. No candidate
entered formal scores. The next scientific bottleneck is subtle geometry
and omission supervision, not simply adding a larger backbone.

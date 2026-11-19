# JW-VL inspired SpaceQA accuracy research

This isolated experiment does not replace the frozen roadshow release.

- `ensemble.py`: verifies the twenty all-family expert caches and GBDT, trains masked three-head stackers on group-separated validation subsets, freezes thresholds/policy before parsing the historical test split, and reports all 30 packages.
- `test_ensemble.py`: group separation, feature allowlist, unknown masking, hard-rule preservation, and clean false-positive calibration checks.
- `grounding.py`: instrument-aware guidance for all six families; unknown products/units stay unknown.
- `train_grounded.py`: matched control/domain continuation from the existing all-family 8B QLoRA adapter. Both visit all 85,950 usable training records per epoch, save resume state, and evaluate all 30 test packages only after the training barrier.
- `launch_gpu_pair.py`: claims two verified idle GPUs on icarus1, retaining advisory locks in the actual workers.
- `download_qwen32.py`: downloads the generic base used by the paper from official ModelScope, verifies sizes and SHA256, and keeps all weights on the server.
- `make_qwen32.py` / `train_qwen32.py`: independently frozen fresh 32B QLoRA training, rank16 q/k/v/o and MLP adapters, one full six-family epoch, with the same train/validation/test separation.
- `queue_qwen32.py`: after control training and evaluation finish, reuses GPU0 for 32B. It does not allocate a third GPU or terminate another process.

All server outputs stay in `/staff/tangjiarui/OPC/jwvl_optimization_20261104/`.

```bash
cd /staff/tangjiarui/OPC/jwvl_optimization_20261104/code
/staff/tangjiarui/OPC/.venv-image-models/bin/python -m unittest -v test_ensemble.py
# Select a NEW output directory for each independent replay.
/staff/tangjiarui/OPC/.venv-image-models/bin/python ensemble.py \
  --output /staff/tangjiarui/OPC/jwvl_optimization_20261104/ensemble_replay
```

Model and threshold selection uses validation only, but this is an exploratory extension on a previously reported test benchmark, not a fresh blind assessment. Labels are P0-derived G/C/D technical proxies, not official nine-class localization truth. Original experts used the original validation set for checkpoint selection; the meta split is not fully nested cross-validation. See `protocol.md` and the report for exact scope and measured results.

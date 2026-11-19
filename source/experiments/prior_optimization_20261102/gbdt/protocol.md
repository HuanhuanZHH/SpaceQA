# Current-only gradient boosting baseline

Three HistGradientBoostingClassifier heads predict G, C, and D. Every head uses
known supervision from all six families and all thirty training packages;
unknown truth_valid bits are excluded. No image decoding or GPU work is used.

- sklearn 1.7.2; seed 20261102; 150 iterations; 15 maximum leaves; learning rate
  0.08; minimum leaf size 30; L2 penalty 1; no early stopping.
- Training balances family sample counts and positive/negative class weights.
- Inputs are the existing current-only 256 floats, a fixed whitelist from
  Current fields JSON, and optionally the current CME group image_time. Reading
  that field verifies the original label SHA before using its bytes.
- Explicit extra features summarize current geometry, class names, declared
  dimensions against decoded dimensions, parameter statistics, CSV residuals,
  relative time differences, and statistical timestamp quantization residuals.
  The latter are learned features and never hard scientific rejection rules.
- Feature vocabulary fits on train only. Tier, paths, IDs, clean differences,
  target, truth_valid, and splits never enter feature extraction. No cross-frame
  peer majority is calculated because peers could lie in other data splits.
- Each family/group threshold is chosen solely on known, present validation
  rows using F1; ties prefer fewer false positives and then higher thresholds.
- All heads, the vectorizer, and thresholds are saved and hashed before the
  test file is opened. Train/val/test groups are checked for disjointness.
- All 30 test packages are reported; masked technical proxy metrics cannot be
  described as official scores or independent physical truth. Existing held-out
  data were previously inspected, making this exploratory repeated evaluation.

The probability interface is predictions/gbdt_prior_v1.jsonl, containing every
old inventory ID and input_sha256, three probabilities, and success/input_missing.
training.json uses the original family order and [G,C,D] threshold order.
Six extractor/masking/calibration boundary tests pass locally and on the server.

Every new server artifact is confined to
/staff/tangjiarui/OPC/inference_prior_20261102/gbdt/.

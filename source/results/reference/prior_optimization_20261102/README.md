# Aggregate research evidence, 2026-11-02

Only aggregate CSV/JSON and frozen policies are included. No sample JSONL,
dataset, image, model checkpoint, vectorizer pickle, or credentials are stored
here. Absolute external paths in provenance locate historical artifacts; they
do not make those private artifacts part of this repository.

- `comparison_v1/`: original cached routes and repaired priors, 30-package tables,
  validation-frozen policy and independent confusion recomputation.
- `gbdt_extension_v1/`: all-family tree extension; the default is
  `val_selected_gbdt_extension`, P1–P4 micro F1 91.37% / recall90.69%.
- `headwise_v1/`: separately frozen family×G/C/D choices. The optional
  `high_recall_review` gives F1 89.84% / recall93.98%; it does not replace default.
- `gbdt/`: fit provenance, all-six-family barrier, standalone 30-package tree
  metrics and roadshow candidate replay summary. No learned files are included.
- `roadshow_v2/`: fresh current-input prior scan aggregate evidence.
- `roadshow_final_v1/` / `headwise_review_roadshow_v1/`: final model-candidate fusion
  summaries. Roadshow has no independent ground truth, and metrics stay null.
- `integrated_default/` / `integrated_review/`: standard-layout export summary
  and integration audit; these do not assert a newly measured scientific score.

All held-out numbers are exploratory historical re-evaluations using known
P0-derived technical proxy bits. Unknown bits are masked. Default C recall is
24.37%; review C recall is82.77%. Do not infer C performance from overall F1.
Container size and full A10 runtime are not verified by these artifacts.

The standalone GBDT `metrics.json`/`complete.json` use sample FPR on fully-known
negative samples; their confusion counts are bit counts. That sample FPR cannot
be reconstructed as FP/(FP+TN) from those bit counts. Other tables explicitly
separate bit FPR from P0 sample FPR. The archive verifier checks the standalone
sample FPR range, without claiming its denominator can be recovered here.

See the [experiment protocol](../../../experiments/prior_optimization_20261102/protocol.md)
for external inputs, executable commands and isolated path relocation. The local
archive verifier checks byte hashes and confusion formulas without private data.

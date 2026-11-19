# Source and dependency notices / 来源与依赖说明

This repository does not grant rights to the competition datasets, pretrained
weights, or third-party packages. No open-source license is assigned to project
code by this repository preparation. The owner must choose a license before
public distribution.

- Frozen project source: existing OPC project and experiments dated 2026-10-31
  and 2026-11-01; byte-level source hashes are retained.
- `reproducibility/external_preprocessing/annotation_diff.py`: existing phase-2
  annotation-comparison source supplied in the project workspace. Its license
  has not been independently established.
- Qwen3-VL, DINOv2, I-JEPA, torchvision weights, Ultralytics models, and their
  Python packages retain their respective upstream licenses. Their weights
  are not redistributed here. In particular, review Ultralytics' applicable
  licensing terms before public or commercial distribution.
- `spec/` includes project interpretations of competition rules and the public
  package catalog. These are versioned reference material, not a claim that
  the organizers endorse this implementation or its scoring formula.

仓库不包含比赛数据或模型权重，也不授予这些材料的再分发权限。现阶段未替用户
决定项目开源许可证。正式对外发布前，由代码所有者确认自有代码许可及外部材料
权属；现有模型与依赖继续遵守各自上游条款。

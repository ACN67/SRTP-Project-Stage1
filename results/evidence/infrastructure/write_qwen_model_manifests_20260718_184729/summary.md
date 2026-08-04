# Qwen Model Manifest Generation

Status: success

Method: Qwen2.5-Coder manifest generation
Owner: shared
Run ID: write_qwen_model_manifests_20260718_184729

Purpose:

- Write lightweight model manifests for Qwen2.5-Coder 1.5B and 3B.
- Fix model revisions for later baseline, pruning, and comparison runs.
- Avoid downloading or committing raw model weights.

Key output:

- Qwen/Qwen2.5-Coder-1.5B-Instruct
  - revision: 2e1fd397ee46e1388853d2af2c993145b0f1098a
  - role: baseline_and_debug
  - layers: 28
  - hidden size: 1536
  - attention heads: 12
  - KV heads: 2
- Qwen/Qwen2.5-Coder-3B-Instruct
  - revision: 488639f1ff808d1d3d0ba301aef8c11461451ec5
  - role: pruning_target_and_comparison
  - layers: 36
  - hidden size: 2048
  - attention heads: 16
  - KV heads: 2

Manifest files:

- data/manifests/models/qwen25_coder_15b_instruct.json
- data/manifests/models/qwen25_coder_3b_instruct.json

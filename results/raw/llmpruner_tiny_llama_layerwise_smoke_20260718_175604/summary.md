# LLM-Pruner Tiny LLaMA Layer-Wise Smoke

Status: success

Method: LLM-Pruner
Owner: 常珂舒
Model: trl-internal-testing/tiny-random-LlamaForCausalLM
Run ID: llmpruner_tiny_llama_layerwise_smoke_20260718_175604

Purpose:

- Verify that LLM-Pruner can run a minimal official `hf_prune.py` layer-wise workflow after fixing the datasets compatibility issue.

Key result:

- Return code: 0
- This is the first successful run after downgrading `datasets` to 2.18.0.
- Detailed PPL and pruning logs are available in `stderr.log`, `stdout.log`, and `upstream_prune_log/`.

# Flab-Pruner Qwen3B HumanEval Heavy Run

Status: success

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide split: data/splits/humaneval/guide.jsonl
Run ID: flab_qwen3b_humaneval_prune_heavy_20260718_204006

Purpose:

- Run the first successful Flab-Pruner Qwen2.5-Coder-3B pruning attempt.
- Use the HumanEval smoke guide split as the recorded pruning guide input.
- Save a pruned model artifact locally and record a manifest/hash instead of committing the large weight file.

Key output:

- Status: success
- Parameters before pruning: 3,085,938,688
- Parameters after pruning: 2,691,711,744
- Actual parameter ratio: 0.8722505584660534
- Approximate parameter reduction: 12.7749%
- Hidden size: 2048 -> 1792
- Intermediate size: 11008 -> 9728
- Attention heads: 16 -> 14
- KV heads: 2 -> 2
- Head dim: 128 -> 128
- Guide SHA256: 0d3fe117b93c0e52ad11064ced4552a8f449f738efd44228fd6416747e604e29
- Peak recorded GPU memory: 7675 MiB
- Duration: 32.188 seconds

Local artifact:

- Pruned model directory: `flab_qwen3b_humaneval/pruned_model/`
- Weight file: `flab_qwen3b_humaneval/pruned_model/model.safetensors`
- Weight size: about 5.1 GiB
- Weight SHA256: abc027f349606412751f373b855888b8651c6fb10d4e3889a94533afe9fa8ef9

Git handling:

The large `model.safetensors` file is intentionally kept local for now. Its SHA256 is recorded in `flab_qwen3b_humaneval/pruned_model_manifest.sha256`.

Important limitation:

This run records and validates benchmark guide input, but the current Flab Qwen2 pruning path still uses structural `top` stage selection. A later benchmark-scored mask selection patch is needed before claiming that guide examples directly determined the pruning mask.

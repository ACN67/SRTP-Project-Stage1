# Flab-Pruner Qwen3B HumanEval Dry Run

Status: success

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide split: data/splits/humaneval/guide.jsonl
Run ID: flab_qwen3b_humaneval_dry_run_20260718_194007

Purpose:

- Validate the project-side Flab-Pruner Qwen wrapper.
- Check Qwen3B config, HumanEval guide hash, and pruning dimensions without loading model weights.
- Prepare the command surface for the future heavy Qwen3B pruning run.

Key output:

- Guide SHA256: 0d3fe117b93c0e52ad11064ced4552a8f449f738efd44228fd6416747e604e29
- Guide task IDs: HumanEval/0, HumanEval/1, HumanEval/10, HumanEval/100
- Hidden size: 2048 -> 1792
- Intermediate size: 11008 -> 9728
- Attention heads: 16 -> 14
- KV heads: 2 -> 2
- Head dim: 128 -> 128
- Estimated parameter ratio after pruning: 0.7923
- Duration: 9.298 seconds

Important limitation:

The wrapper validates and records benchmark guide input, but the current upstream Flab Qwen2 pruning function still uses structural stage selection (`top`, `bottom`, `random`, `middle`). A further scoring patch is needed before claiming that benchmark guide data directly chooses the pruning mask.

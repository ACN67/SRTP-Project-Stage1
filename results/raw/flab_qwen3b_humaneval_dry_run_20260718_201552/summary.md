# Flab-Pruner Qwen3B HumanEval Dry Run

Status: success

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide split: data/splits/humaneval/guide.jsonl
Run ID: flab_qwen3b_humaneval_dry_run_20260718_201552

Purpose:

- Verify the Qwen2.5 config compatibility patch after the first heavy run failed on missing `rope_theta`.
- Validate guide split and pruning dimensions without loading weights.

Key output:

- Compatibility patches applied:
  - rope_theta: 1000000.0
  - sliding_window: null
- Hidden size: 2048 -> 1792
- Intermediate size: 11008 -> 9728
- Attention heads: 16 -> 14
- KV heads: 2 -> 2
- Head dim: 128 -> 128
- Duration: 6.345 seconds

Next step:

Re-run `flab_qwen3b_humaneval_prune_heavy`. The model files should mostly be cached, so the retry should not need the same long download phase.

# Flab-Pruner Method Notes

Owner: 常珂舒  
Group: structured_depth_width  
Stage: 1  
Current status: R0 inspected, Qwen 3B project wrapper dry-run succeeded; heavy pruning run pending.

## Upstream State

The upstream repository includes code-specific pruning scripts and Qwen2 modeling utilities.

Important files:

```text
third_party/flab_pruner/README.md
third_party/flab_pruner/vocab_prune.py
third_party/flab_pruner/greedy_prune.py
third_party/flab_pruner/ffn_prune.py
third_party/flab_pruner/hidden_prune_utils/modeling_qwen2.py
third_party/flab_pruner/hidden_prune_utils/prune_qwen2.py
third_party/flab_pruner/zero_shot_code_generation.py
```

## Why Flab-Pruner Is Important Here

Flab-Pruner is the most directly aligned method for the project target because it is code-LLM-focused and already contains Qwen2-specific model code.

This makes it a strong candidate for 常珂舒's representative Qwen 3B pruning method.

## Current Compatibility Findings

What is useful:

- `hidden_prune_utils/modeling_qwen2.py` defines Qwen2 model classes.
- `ffn_prune.py` imports `Qwen2ForCausalLM` from the local Qwen2 implementation.
- `vocab_prune.py`, `greedy_prune.py`, and `ffn_prune.py` cover vocabulary, layer, and FFN pruning patterns.

What blocks direct Qwen2.5-Coder-3B execution:

- Example paths are hard-coded for local upstream machines.
- Some remain dimensions are 7B-oriented constants.
- Scripts do not yet expose a project CLI for model id, save path, pruning ratio, guide split, or output metadata.
- `greedy_prune.py` expects upstream dataset paths that are not present locally.

## Qwen 3B Adaptation Requirements

A project-compatible Flab-Pruner path should:

- Accept `--model Qwen/Qwen2.5-Coder-3B-Instruct` or a local model path.
- Accept `--guide-file data/splits/<benchmark>/guide.jsonl`.
- Convert guide examples into pruning/calibration text.
- Infer hidden size, FFN size, attention heads, and KV heads from Qwen 3B config.
- Accept conservative remaining dimensions or pruning ratios.
- Save pruned model artifacts or manifests under `results/raw/<run_id>/`.
- Save exact parameter counts before and after pruning.

## Stage 1 Decision

Flab-Pruner should be prioritized for Qwen 3B target adaptation after the benchmark guide/eval split interface is created.

Until the CLI and path constants are patched, it should be recorded as R0 inspected and Qwen-promising, not as R1/R2 complete.

## Project Wrapper

The project-side wrapper is:

```text
scripts/adapt/flab_qwen_prune.py
```

It avoids editing the upstream repository directly and provides:

- `--model`
- `--guide-file`
- `--save-dir`
- `--stage`
- `--prune-ratio`
- `--dry-run`
- explicit remain-size overrides

Dry-run status:

```text
Run ID: flab_qwen3b_humaneval_dry_run_20260718_194007
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide: data/splits/humaneval/guide.jsonl
Stage: top
Requested prune ratio: 0.10
Status: success
```

Dry-run plan:

```text
hidden_size: 2048 -> 1792
intermediate_size: 11008 -> 9728
attention_heads: 16 -> 14
kv_heads: 2 -> 2
head_dim: 128 -> 128
rough parameter ratio: 0.7923
```

Important limitation:

The current wrapper validates and records benchmark guide input, but upstream Flab's Qwen2 prune path still uses structural `top` / `bottom` / `random` style mask selection. A further scoring patch is required before claiming fully benchmark-guided pruning decisions.

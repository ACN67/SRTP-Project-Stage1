# Method Status

The compact source of truth is `results/stage1/status.csv`. This document explains the same rows in human-readable form.

## 常珂舒

| Method | Model | Current level | Status |
|---|---|---|---|
| Flab-Pruner | Qwen2.5-Coder-3B | R3 complete | Qwen3B pruning, load/generate, HumanEval smoke, and MBPP smoke are complete. R4 pending. |
| LLM-Pruner | CodeLlama-7B | R3 complete | Tiny LLaMA R1, CodeLlama load, layer-wise prune, HumanEval smoke, and MBPP smoke are complete. R4 pending. |
| SliceGPT | CodeLlama-7B | R3 complete | OPT R1, CodeLlama rotate-and-slice, HumanEval smoke, and MBPP smoke are complete. R4 pending. |
| LaCo | skipped | R1 blocked | Official support is too limited and does not provide a CodeLlama route. |

## 潘阔

| Method | Model | Current level | Status |
|---|---|---|---|
| Magnitude | CodeLlama if local path supports LLaMA | pending | No owner run yet. |
| Wanda | CodeLlama if local path supports LLaMA | pending | No owner run yet. |
| DSnoT | CodeLlama if supported | pending | No owner run yet. |
| OWL | CodeLlama if supported | pending | No owner run yet. |

## 李长骏

| Method | Model | Current level | Status |
|---|---|---|---|
| SparseGPT | CodeLlama if local LLaMA path is usable | pending | No owner run yet. |
| MaskLLM | CodeLlama if supported | pending | No owner run yet. |
| Pruner-Zero | CodeLlama if supported | pending | No owner run yet. |
| FLAP | CodeLlama | pending | No owner run yet. |

## Key Evidence

| Method | R2 evidence | R3 evidence |
|---|---|---|
| Flab-Pruner | `results/raw/flab_qwen3b_humaneval_prune_heavy_20260718_204006/` | `results/raw/flab_qwen3b_pruned_humaneval_smoke_eval_20260721_182050/`, `results/raw/flab_qwen3b_pruned_mbpp_smoke_eval_20260721_185132/` |
| LLM-Pruner | `results/raw/llmpruner_codellama7b_layerwise_cpu_smoke_20260722_222955/` | `results/raw/llmpruner_codellama7b_humaneval_4task_offload_smoke_20260722_232800/`, `results/raw/llmpruner_codellama7b_mbpp_4task_offload_smoke_20260722_233633/` |
| SliceGPT | `results/raw/slicegpt_codellama7b_min_prune_smoke_20260722_235632/` | `results/raw/slicegpt_codellama7b_r3_humaneval_mbpp_4task_20260723_021439/` |

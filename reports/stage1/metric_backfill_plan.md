# Stage 1 Metric Backfill Plan

Updated: 2026-07-22

## Decision

Existing experiments do not all need to be rerun. The rule is:

- If a run already has `resource.csv`, backfill `resource_summary.json` from the existing trace.
- If a run lacks resource traces but will be used for runtime or memory reduction comparison, rerun it through the unified runner or a monitored command.
- If a run is only used as pipeline-connectivity evidence, keep it as-is and mark resource metrics unavailable.

## Current Assessment

| Run / run type | Existing status | Resource data | Action |
|---|---|---|---|
| `flab_qwen3b_humaneval_prune_heavy_20260718_204006` | Qwen3B pruning success | `resource.csv` exists | Backfill `resource_summary.json`; no pruning rerun needed. |
| `qwen15b_baseline_load_generate_check_20260721_173527` | baseline load/generate success | no `resource.csv` | Rerun only if load/generate peak memory is needed for final reduction tables. |
| `qwen3b_baseline_load_generate_check_20260721_175920` | baseline load/generate success | no `resource.csv` | Rerun only if load/generate peak memory is needed for final reduction tables. |
| `flab_qwen3b_pruned_load_generate_check_20260721_173111` | pruned load/generate success | no `resource.csv` | Rerun only if load/generate peak memory is needed for final reduction tables. |
| HumanEval/MBPP smoke eval runs | pipeline scoring success | no `resource.csv` | Keep as smoke evidence; rerun monitored versions before formal runtime/VRAM comparison. |

## Required Fields For Future Runs

Every future pruning, generation, or evaluation run used in final comparison should preserve:

```text
duration_sec
peak_gpu_memory_used_mb
peak_process_rss_mb
parameter_count_before
parameter_count_after
artifact_size_bytes_before
artifact_size_bytes_after
task_count
pass_count
pass_rate
```

## Rerun Priority

1. Do not rerun Flab-Pruner Qwen3B pruning unless the existing artifact or logs become unusable.
2. Rerun monitored load/generate checks only when computing resource reduction.
3. Rerun monitored HumanEval/MBPP smoke evals only if those smoke results are included in a resource table.
4. For CodeLlama-based SliceGPT/LLM-Pruner and other methods, use the updated runner so resource summaries are created automatically from the first run.

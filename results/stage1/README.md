# Stage 1 Result Index

This directory is the curated navigation layer for Stage 1 results.

Use this layer first when checking project progress. Detailed command logs and raw outputs remain in `results/raw/` as immutable evidence, but the R0-R4 directories here organize the work by research round and method.

## Layout

| Directory | Purpose |
|---|---|
| `R0_readiness/` | Machine, dependency, benchmark split, and model-selection readiness checks. |
| `R1_official_smoke/` | Small official-repository smoke runs before model-family adaptation. |
| `R2_method_prune/` | Method adaptation and local pruning proof on the selected model family. |
| `R3_benchmark_smoke/` | Four-task HumanEval/MBPP smoke evaluation after pruning. |
| `R4_half_set/` | Full Stage 1 benchmark-guided half-set runs. Pending. |

## Storage Rules

- Keep concise summaries and manifests in `results/stage1/R*/`.
- Keep full logs in `results/raw/<run_id>/`.
- Keep model weights local only. Do not commit `.safetensors`, `.pt`, tokenizer dumps, or offload folders.
- Record local model artifact paths in manifests when they are needed for reruns.
- For formal R4 runs, each method should save generation outputs, score summaries, resource summaries, and a compact method-level `summary.md`.


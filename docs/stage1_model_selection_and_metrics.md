# Stage 1 Model Selection And Metric Policy

Stage: 1
Status: active policy
Updated: 2026-07-22

## Model Selection Policy

Stage 1 no longer requires every pruning method to be forced onto Qwen2.5-Coder. The model must now be chosen according to the method's official support boundary.

Priority order:

1. Use CodeLlama when the method officially supports LLaMA-family Hugging Face models.
2. Use Qwen2.5-Coder when the method has explicit Qwen/Qwen2 support or a completed Qwen adapter.
3. Use the method's own officially supported small model when neither CodeLlama nor Qwen is practical.
4. Keep notebook-only or non-scriptable methods as analyzed/skipped unless converting them is explicitly assigned.

This makes the experiment fairer: each method is evaluated on a model family it can actually prune, while final reporting uses normalized metrics rather than pretending all methods operate on one identical model.

## Current Method-To-Model Mapping

| Owner | Method | Primary Stage 1 pruning model | Baseline / comparison model | Reason |
|---|---|---|---|---|
| 常珂舒 | Flab-Pruner | `Qwen/Qwen2.5-Coder-3B-Instruct` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` and original 3B | Upstream includes Qwen2 code and the local Qwen3B pruning path has already succeeded. |
| 常珂舒 | SliceGPT | CodeLlama-family model, preferably `codellama/CodeLlama-7b-hf` if resources allow | same-family unpruned CodeLlama | Official adapters include LLaMA, OPT, Phi-2, and Phi-3; Qwen2 adapter is missing. |
| 常珂舒 | LLM-Pruner | CodeLlama-family model, preferably `codellama/CodeLlama-7b-hf` if resources allow | same-family unpruned CodeLlama | Official `hf_prune.py` is LLaMA-bound; Qwen requires a deeper adapter. |
| 常珂舒 | LaCo | skipped for Stage 1 | N/A | Official implementation is notebook-only; conversion is deferred. |
| 潘阔 | Wanda / Magnitude | CodeLlama-family model if the local Wanda path supports LLaMA; otherwise OPT smoke model | same-family unpruned model | Wanda supports LLaMA-style workflows in common use; confirm exact local entry before heavy runs. |
| 潘阔 | DSnoT / OWL | CodeLlama-family model if the local method path supports LLaMA; otherwise method-supported model | same-family unpruned model | Mask allocation methods should use CodeLlama when the upstream scripts accept LLaMA models. |
| 李长骏 | SparseGPT | CodeLlama-family model if the local SparseGPT LLaMA path is usable; otherwise OPT smoke model | same-family unpruned model | SparseGPT has LLaMA-family support upstream, but resource needs are higher than OPT smoke. |
| 李长骏 | FLAP | CodeLlama-family model | same-family unpruned CodeLlama | The existing plan already has a LLaMA-style FLAP template. |
| 李长骏 | MaskLLM / Pruner-Zero | CodeLlama-family model if supported; otherwise method-supported model | same-family unpruned model | Use official supported families before attempting new Qwen adapters. |

## Naming Policy

Run IDs and result directories must include both method and model family.

Recommended pattern:

```text
<method>_<model_short>_<benchmark>_<stage>_<yyyymmdd_hhmmss>
```

Examples:

```text
flab_qwen25c3b_humaneval_prune_heavy_20260718_204006
slicegpt_codellama7b_humaneval_smoke_YYYYMMDD_HHMMSS
llmpruner_codellama7b_mbpp_smoke_YYYYMMDD_HHMMSS
wanda_codellama7b_humaneval_smoke_YYYYMMDD_HHMMSS
sparsegpt_codellama7b_mbpp_smoke_YYYYMMDD_HHMMSS
```

Historical run IDs are not renamed. New runs should follow this policy.

## Result Layout Policy

Keep results separated by run ID under:

```text
results/raw/<run_id>/
```

Each run should preserve:

```text
metadata.json
summary.json
summary.md
command.sh
stdout.log
stderr.log
resource.csv
resource_summary.json
generation_summary.json        if generation is performed
score_summary.json             if scoring is performed
score_details.jsonl            if scoring is performed
artifact_manifest.sha256       if large artifacts are kept locally
```

Large model artifacts should not be committed unless the team explicitly chooses Git LFS storage for that artifact. At minimum, keep file names, sizes, hashes, and regeneration commands.

## Metric Policy

Because methods may use different model families, final Stage 1 reporting should compare normalized indicators rather than raw scores alone.

Required raw metrics:

| Metric | Meaning | Source |
|---|---|---|
| `task_count` | number of evaluated tasks | score summary |
| `pass_count` / `pass_rate` | benchmark smoke or formal score | score summary |
| `baseline_pass_rate` | unpruned same-family model score | baseline score summary |
| `pruned_pass_rate` | pruned model score | pruned score summary |
| `parameter_count_before` | unpruned parameter count | pruning result or manifest |
| `parameter_count_after` | pruned parameter count | pruning result or manifest |
| `artifact_size_bytes_before` | unpruned model size if locally materialized | artifact manifest |
| `artifact_size_bytes_after` | pruned model size if locally materialized | artifact manifest |
| `duration_sec` | end-to-end run time | `summary.json` |
| `peak_gpu_memory_used_mb` | peak GPU memory while running | `resource_summary.json` |
| `peak_process_rss_mb` | peak host process RSS | `resource_summary.json` |

Derived metrics:

| Metric | Formula |
|---|---|
| ability retention rate | `pruned_pass_rate / baseline_pass_rate`, reported as `N/A` when baseline is zero |
| parameter reduction rate | `1 - parameter_count_after / parameter_count_before` |
| runtime reduction rate | `1 - pruned_duration_sec / baseline_duration_sec`, only for comparable tasks on same hardware |
| peak VRAM reduction rate | `1 - pruned_peak_gpu_memory_mb / baseline_peak_gpu_memory_mb`, only for comparable tasks on same hardware |
| artifact size reduction rate | `1 - artifact_size_bytes_after / artifact_size_bytes_before` |

Smoke results may be used to prove pipeline connectivity. Formal conclusions require larger eval splits and controlled decoding settings.

## Reporting Rule

Every final table must state the model family used by each method.

Acceptable comparison language:

```text
Flab-Pruner on Qwen2.5-Coder and SliceGPT on CodeLlama are not raw-score comparable as model-quality claims; they are compared by normalized pruning retention and resource reduction indicators.
```

Avoid claiming that one pruning method is better than another from smoke runs across different base models.

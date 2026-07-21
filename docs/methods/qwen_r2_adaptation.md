# Historical Qwen Adaptation Plan

Stage: 1  
Status: superseded by `docs/stage1_model_selection_and_metrics.md`  
Coverage model: Qwen/Qwen2.5-Coder-1.5B-Instruct  
Comparison model: Qwen/Qwen2.5-Coder-3B-Instruct  
Fallback base models: Qwen/Qwen2.5-Coder-1.5B and Qwen/Qwen2.5-Coder-3B  
Goal: reproduce pruning methods, adapt representative methods to Qwen, prune Qwen 3B with benchmark guide data, and compare against Qwen 1.5B.

## Current Policy Update

This file records the earlier Qwen-only route. It is no longer the active policy for all methods.

The active Stage 1 policy is:

- Use Qwen2.5-Coder for methods with explicit Qwen/Qwen2 support, currently represented by Flab-Pruner.
- Use CodeLlama-family models for methods whose official code supports LLaMA-family models, such as LLM-Pruner, SliceGPT, FLAP, SparseGPT/Wanda when their local LLaMA paths are used, and related methods.
- Use each method's own officially supported model when CodeLlama is not practical.
- Compare methods with normalized retention and resource-reduction metrics, not by raw scores across different base models.

See:

```text
docs/stage1_model_selection_and_metrics.md
```

## Why This Exists

This section reflects the earlier plan before the model-selection adjustment.

Stage 1 is not complete if it only reproduces tiny random models. Those runs prove that upstream methods and local environments can execute, but the project target is pruning code LLMs. The current code-LLM families are method-dependent: Qwen2.5-Coder for Qwen-supported methods and CodeLlama for LLaMA-supported methods.

The execution book uses Qwen2.5-Coder-1.5B as the coverage-first model and Qwen2.5-Coder-3B as the representative expansion model. The end target is not merely to show that Qwen can load; it is to prune Qwen 3B with benchmark guide data and compare the pruned result with Qwen 1.5B baselines.

The Qwen work therefore has two layers:

- R2: method compatibility on Qwen 1.5B.
- R3/Q3: representative-method pruning on Qwen 3B with benchmark-guided calibration and evaluation.

## R2 Definition

A method reaches Qwen R2 when it can:

- Load Qwen2.5-Coder-1.5B or 1.5B-Instruct.
- Locate decoder layers, attention projections, MLP projections, and RMSNorm modules.
- Run a minimum pruning configuration with a low pruning ratio.
- Save the pruned artifact or a reproducible manifest.
- Reload or validate the result.
- Produce at least one short code-generation output or perplexity smoke result.
- Record actual parameter count, sparsity, logs, and environment.

## Q3 / Comparison Definition

A representative method reaches the Qwen 3B comparison target when it can:

- Load frozen Qwen2.5-Coder-3B or 3B-Instruct metadata and weights.
- Use a fixed benchmark guide split as pruning calibration or method guidance.
- Save a pruned Qwen 3B artifact or a reproducible artifact manifest.
- Evaluate on the matching benchmark eval split.
- Report whether the pruned Qwen 3B result is comparable to, better than, or worse than the unpruned Qwen 1.5B baseline under the same evaluation protocol.

The comparison must keep these results separate:

- Original Qwen 1.5B.
- Original Qwen 3B.
- Pruned Qwen 3B.
- Optional pruned Qwen 1.5B smoke results.

Base and Instruct models must not be mixed in the same comparison table.

## Benchmark-Guided Pruning

Benchmark guide data is part of the pruning stage, not only the evaluation stage. The guide split should be converted into calibration or task-input text and passed into the pruning method whenever the method accepts calibration, activations, gradients, hidden states, or candidate-scoring examples.

Minimum Stage 1 mapping:

| Method type | Guide usage |
|---|---|
| Wanda / OWL | activation norms, outliers, layer sensitivity |
| SparseGPT | layer input statistics and reconstruction information |
| LLM-Pruner | gradient or structural importance signal |
| DSnoT | sparse mask correction or search inputs |
| MaskLLM | N:M mask learning/search inputs |
| FLAP | activation fluctuation and structure importance |
| SliceGPT | representative hidden states for rotation/slicing |
| LaCo | layer-collapse candidate impact estimate |
| Pruner-Zero | candidate importance metric scoring |
| Flab-Pruner | code-task-aware pruning signal |

Smoke tests may use only HumanEval and MBPP guide samples. Formal Stage 1 evidence must show that the four benchmark pipelines can produce guide/eval splits, even if not every method runs every benchmark.

## Current Structured-Method Assessment

| Method | R1 state | Expected Qwen R2 path |
|---|---|---|
| SliceGPT | OPT-125M R1 smoke succeeded | Needs Qwen2 adapter or confirmation that Qwen can be mapped to an existing LLaMA-style adapter; guide data supplies hidden states. |
| LLM-Pruner | tiny LLaMA R1 smoke succeeded | Current script imports custom LLaMA classes; likely needs an AutoModel/Qwen path or adapter patch; guide data supplies gradient/importance signal. |
| LaCo | blocked because upstream is notebook-only | Needs notebook-to-script extraction before Qwen work. |
| Flab-Pruner | pending R0/R1 inspection | Inspect official entry points before Qwen judgment. |

## First Qwen Task

Run lightweight structure probes before any pruning:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only qwen25_coder_15b_config_probe \
  --include-disabled
```

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only qwen25_coder_3b_config_probe \
  --include-disabled
```

These jobs download config/tokenizer files only. They do not download model weights and do not prune.

## Decision Rule

After the probe:

- If Qwen layout matches LLaMA-style names closely, attempt a minimal SliceGPT or LLM-Pruner adapter patch on 1.5B first.
- If the upstream method hard-codes LLaMA classes, record the incompatibility and patch requirement before any heavy run.
- If a method reaches Qwen 1.5B R2 and is one of the representative methods, prepare a Qwen 3B run using benchmark guide data.
- If a method needs full weights, run only one Qwen method at a time because 1.5B and 3B models can stress 8 GB VRAM.

# Qwen R2 Adaptation Plan

Stage: 1  
Target model: Qwen/Qwen2.5-Coder-1.5B-Instruct  
Fallback model: Qwen/Qwen2.5-Coder-1.5B  
Goal: move from official-method R1 reproduction to Qwen minimum reproduction.

## Why This Exists

Stage 1 is not complete if it only reproduces OPT, LLaMA, or tiny random models. Those runs prove that upstream methods and local environments can execute, but the project target is pruning code LLMs, with Qwen2.5-Coder as the main small-model target.

The Qwen work should therefore be treated as R2, after R1 official reproduction.

## R2 Definition

A method reaches Qwen R2 when it can:

- Load Qwen2.5-Coder-1.5B or 1.5B-Instruct.
- Locate decoder layers, attention projections, MLP projections, and RMSNorm modules.
- Run a minimum pruning configuration with a low pruning ratio.
- Save the pruned artifact or a reproducible manifest.
- Reload or validate the result.
- Produce at least one short code-generation output or perplexity smoke result.
- Record actual parameter count, sparsity, logs, and environment.

## Current Structured-Method Assessment

| Method | R1 state | Expected Qwen R2 path |
|---|---|---|
| SliceGPT | OPT-125M R1 smoke succeeded | Needs Qwen2 adapter or confirmation that Qwen can be mapped to an existing LLaMA-style adapter. |
| LLM-Pruner | tiny LLaMA R1 smoke succeeded | Current script imports custom LLaMA classes; likely needs an AutoModel/Qwen path or adapter patch. |
| LaCo | blocked because upstream is notebook-only | Needs notebook-to-script extraction before Qwen work. |
| Flab-Pruner | pending R0/R1 inspection | Inspect official entry points before Qwen judgment. |

## First Qwen Task

Run a lightweight structure probe before any pruning:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only qwen25_coder_15b_config_probe \
  --include-disabled
```

This downloads config/tokenizer files only. It does not download model weights and does not prune.

## Decision Rule

After the probe:

- If Qwen layout matches LLaMA-style names closely, attempt a minimal SliceGPT or LLM-Pruner adapter patch.
- If the upstream method hard-codes LLaMA classes, record the incompatibility and patch requirement before any heavy run.
- If a method needs full weights, run only one Qwen method at a time because 1.5B models can still stress 8 GB VRAM.

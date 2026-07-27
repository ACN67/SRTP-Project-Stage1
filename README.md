# SRTP Stage 1: Code LLM Pruning

This repository is the Stage 1 workspace for reproducing code-LLM pruning methods and building a unified benchmark-guided evaluation pipeline.

Stage 1 is about reproducibility, workflow, and evidence. Smoke results are not final performance claims.

## Start Here

| File | Purpose |
|---|---|
| `docs/stage1.md` | Stage goal, R0-R4 definitions, model and benchmark policy. |
| `docs/methods.md` | Method ownership, current progress, blockers, and model choices. |
| `docs/runbook.md` | How to run checks, smoke jobs, and future R4 jobs. |
| `docs/official_benchmarks.md` | Official benchmark prompt/scoring policy for formal results. |
| `docs/r4_keep80_official_runs.md` | Official keep80 run plan for Flab-Pruner, LLM-Pruner, and SliceGPT on Qwen2.5-Coder-3B. |
| `docs/environment.md` | Local machine, environment, cache, and artifact notes. |
| `results/stage1/status.csv` | Compact machine-readable progress table. |
| `results/raw/` | Timestamped raw evidence for every recorded run. |

## Current Scope

| Owner | Methods |
|---|---|
| 常珂舒 | Flab-Pruner, SliceGPT, LLM-Pruner, LaCo |
| 潘阔 | Magnitude, Wanda, DSnoT, OWL, benchmark splits |
| 李长骏 | SparseGPT, MaskLLM, Pruner-Zero, FLAP, resource checks |

Current model policy:

- Use CodeLlama for LLaMA-compatible methods.
- Use Qwen2.5-Coder for methods with Qwen/Qwen2 support.
- Use the official model family when neither CodeLlama nor Qwen is practical.
- Skip methods with too little official support unless explicitly assigned.

## Repository Layout

```text
configs/        Experiment plans.
data/           Dataset splits and model/upstream manifests.
docs/           Consolidated execution, method, run, and environment docs.
env/            Per-environment dependency snapshots.
patches/        Local patches against upstream submodules.
scripts/        Setup, adaptation, evaluation, audit, and run helpers.
third_party/    Upstream repositories as Git submodules.
results/stage1/ Compact R0-R4 progress and formal summaries.
results/raw/    Timestamped raw logs and generated outputs.
artifacts/      Local artifact policy; model weights stay out of Git.
```

## Quick Commands

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --list
```

For R4, follow `docs/runbook.md` and `docs/official_benchmarks.md` after confirming the relevant R3 method row is complete in `results/stage1/status.csv`.

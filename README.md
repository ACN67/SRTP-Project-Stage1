# SRTP Stage 1: Code LLM Pruning

This repository records Stage 1 of the SRTP code-LLM pruning reproduction project.
It is organized around three maintenance entry points:

- `data/`: benchmark splits, split roles, and model/upstream manifests.
- `methods/`: method adapters, method notes, and local patches.
- `results/`: immutable evidence, formal R4-half registries, Pan auxiliary results, and audit reports.

Cross-method execution code is under `workflows/`. Environment snapshots and WSL setup helpers are under `environment/`. Upstream projects remain Git submodules in `third_party/`.

## Current State

Stage 1 is not a clean success claim. It contains completed runs whose quality gates failed, partial runs, fallback/non-official paths, aggregate-only auxiliary results, and methods still pending.

常珂舒侧 WSL worktree content from `$SRTP_STAGE1_ROOT` has been preserved and integrated. Large model artifacts remain outside Git and are listed in `results/registries/artifacts.csv`.

Pan auxiliary full-eval results are preserved under `results/auxiliary/pan_full_eval/` and are not directly comparable with R4 half results.

## Key Registries

- `results/registries/methods.csv`
- `results/registries/runs.csv`
- `results/registries/scores.csv`
- `results/registries/artifacts.csv`

## Formal R4 Half

R4 half data lives in `data/benchmarks/r4_half/`.
Formal and diagnostic evidence lives in `results/evidence/r4_half/`.
Generated summaries live in `results/formal/r4_half/`.

## WSL Use

The experimental working location is:

```bash
cd $SRTP_STAGE1_ROOT
```

Run active helpers through `workflows/`, for example:

```bash
source environment/setup/env.sh
workflows/experiment/run_plan.sh --plan workflows/experiment/stage1_plan.yaml --list
```


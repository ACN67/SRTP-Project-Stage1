# SRTP Project Stage 1: Code LLM Pruning Reproduction

This repository contains the Stage 1 workspace for reproducing and standardizing multiple pruning methods for code large language models.

Stage 1 is not meant to prove which pruning method is best. Its main goal is to make each method reproducible, auditable, and easy for another teammate to rerun.

See the full execution book:

```text
docs/execution/stage1_execution.md
```

## Stage 1 Goals

The first stage focuses on six practical targets:

- Reproduce official or author-recognized implementations of classical and mainstream pruning methods.
- Select a realistic pruning model for each method based on upstream support: use CodeLlama for LLaMA-compatible methods, keep Qwen2.5-Coder for methods with Qwen support, and otherwise use the method's officially supported model.
- Build a unified experiment layout for HumanEval, MBPP, LiveCodeBench, and SWE-bench Lite.
- Keep scripts, configs, logs, failures, results, resource traces, metric summaries, and environment snapshots inside Git or Git LFS.
- Make every successful or failed run traceable through metadata and logs.
- Ensure at least one teammate can rerun another teammate's minimum reproduction.

The active model-selection and metric policy is documented in:

```text
docs/stage1_model_selection_and_metrics.md
```

## Method Scope

Stage 1 includes these methods:

| Family | Methods |
|---|---|
| Baseline and activation-aware pruning | Magnitude, Wanda |
| Reconstruction and search | SparseGPT, MaskLLM, Pruner-Zero, FLAP |
| Mask optimization and allocation | DSnoT, OWL |
| Structured, width, depth, and code-specific pruning | LLM-Pruner, SliceGPT, LaCo, Flab-Pruner |

Frontier methods such as IFPruning, Self-Pruner, and agent-guided pruning are deferred to backlog documentation.

## Team Ownership

| Member | Main Responsibility | Methods / Area |
|---|---|---|
| 常珂舒 | Repository integration and structured/code-specific methods | LLM-Pruner, SliceGPT, LaCo, Flab-Pruner |
| 潘阔 | Activation-aware pruning, mask optimization, benchmark data | Magnitude, Wanda, DSnoT, OWL, benchmark splits |
| 李长骏 | Reconstruction/search methods and environment audit | SparseGPT, MaskLLM, Pruner-Zero, FLAP, machine/resource checks |

Experiment jobs also store `owner`, `method_group`, `method`, `role`, `recommended_machine`, and `cross_reproduction_by` fields in their result metadata.

## Repository Layout

```text
docs/                         Execution book, method notes, troubleshooting
configs/experiments/           Manual run plans
scripts/setup/                 Environment setup helpers
scripts/run/                   Unified manual experiment runner
scripts/audit/                 Environment and result checks
env/                           pip freeze snapshots per environment
third_party/                   Official upstream repositories as Git submodules
data/manifests/                Upstream commit and model/data manifests
results/raw/                   Per-run logs, metadata, stdout, stderr, resource traces
results/tables/                Method status tables
reports/stage1/                Machine info, environment checks, run summaries
artifacts/                     Pruned models, patches, trajectories, manifests
```

## Quick Start

Use WSL. The project is expected to live inside the Linux filesystem, not under `/mnt/c`.

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
```

If this is a fresh clone, initialize submodules first:

```bash
git submodule update --init --recursive
```

Then run a light environment check:

```bash
source .venv-common/bin/activate
python scripts/audit/check_environment.py
```

The check writes:

```text
reports/stage1/environment_check.json
```

## Environment Setup

The local machine is configured with per-method Python virtual environments:

```text
.venv-common
.venv-magnitude
.venv-wanda
.venv-sparsegpt
.venv-llm_pruner
.venv-dsnot
.venv-maskllm
.venv-flap
.venv-slicegpt
.venv-laco
.venv-owl
.venv-pruner_zero
.venv-flab_pruner
.venv-livecodebench
```

Recreate them with:

```bash
scripts/setup/create_method_envs.sh
```

Environment snapshots are stored under:

```text
env/<name>/pip_freeze.txt
```

Heavy or fragile packages such as `apex`, `deepspeed`, `fairseq`, `mamba-ssm`, and `vllm` are intentionally not installed globally. Install them only inside the specific method environment when an actual reproduction command requires them.

## Running Experiments Manually

Long experiments should be run by a human in a terminal, not supervised continuously by Codex.

List available jobs:

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --list
```

Run only the default light checks:

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml
```

Run one explicit heavy job:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only sparsegpt_opt125m_sparsegpt_smoke \
  --include-disabled
```

Full manual running instructions are in:

```text
docs/RUN_EXPERIMENTS_MANUALLY.md
```

## Result Files

The main file to review after manual runs is:

```text
reports/stage1/manual_run_results.jsonl
```

A Markdown table is also generated:

```text
reports/stage1/manual_run_results.md
```

Each run has a dedicated directory:

```text
results/raw/<run_id>/
```

Typical contents:

```text
metadata.json
summary.json
summary.md
command.sh
stdout.log
stderr.log
resource.csv
resource_summary.json
```

If a run fails, keep the run directory. Failed runs are part of the Stage 1 evidence and should not be deleted.

## Git LFS

Large model and artifact files are tracked through Git LFS patterns in `.gitattributes`, including:

```text
*.safetensors
*.bin
*.pt
*.pth
*.ckpt
*.parquet
*.jsonl.gz
*.tar.gz
*.zip
```

Do not commit raw Hugging Face cache directories. Commit model manifests, revisions, hashes, and scripts needed to regenerate artifacts.

## Current Status

Configured on the current machine:

- WSL2 with GPU visibility.
- Docker Desktop usable from WSL through the wrapper in `scripts/setup/install_docker_wrapper.sh`.
- Git LFS initialized.
- Third-party repositories added as submodules.
- Common and method-specific Python environments created.
- Environment audit passed with CUDA available on RTX 5060 Laptop GPU.

Known caveats:

- Current WSL distro is Ubuntu 26.04, while the execution book recommends Ubuntu 22.04 LTS for team parity.
- Some upstream methods are LLaMA-family first. These should now use CodeLlama-family models where practical instead of forcing Qwen adapters.
- Flab-Pruner remains the Qwen2.5-Coder path because it already includes Qwen2 support and has a successful Qwen3B pruning run.
- SWE-bench Lite and LiveCodeBench integration may require additional benchmark-specific setup during formal evaluation.

## Recommended First Runs

Start with small official reproductions before method-family pruning:

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only sparsegpt_opt125m_gmp_smoke --include-disabled
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only sparsegpt_opt125m_sparsegpt_smoke --include-disabled
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only wanda_opt125m_magnitude_smoke --include-disabled
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only wanda_opt125m_wanda_smoke --include-disabled
```

After each run, commit the generated summary files and any relevant failure logs or manifests. Large artifacts should use Git LFS or be represented by a manifest if they can be regenerated.

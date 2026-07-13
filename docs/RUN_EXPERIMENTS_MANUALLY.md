# Manual Experiment Running Guide

This repo is configured so long experiments can run without Codex supervising them.
After running, the main file to inspect is:

```text
reports/stage1/manual_run_results.jsonl
```

Each line is one job summary. A readable table is also generated at:

```text
reports/stage1/manual_run_results.md
```

Detailed logs for each run are written under:

```text
results/raw/<run_id>/
```

## 1. Enter WSL Project

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
```

## 2. List Jobs

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --list
```


## Team Ownership

The run plan records responsibility fields for each job:

- `owner`: primary person responsible for the run.
- `method_group`: method family or shared pipeline area.
- `method`: concrete method or benchmark.
- `role`: why this run exists in the Stage 1 workflow.
- `recommended_machine`: suggested machine class.
- `cross_reproduction_by`: teammate expected to rerun or validate the minimum reproduction when applicable.

Current ownership follows the execution book:

| Owner | Scope |
|---|---|
| 潘阔 | Magnitude, Wanda, DSnoT, OWL, benchmark data pipeline |
| 李长骏 | SparseGPT, MaskLLM, Pruner-Zero, FLAP, environment audit |
| 常珂舒 | LLM-Pruner, SliceGPT, LaCo, Flab-Pruner, repository integration |
| shared | Common environment checks |

These fields are copied into `reports/stage1/manual_run_results.jsonl`, so results from three machines can be merged without losing responsibility information.

## 3. Run Only Light Checks

This should be quick and does not run pruning:

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml
```

## 4. Run One Specific Heavy Experiment

Disabled jobs require `--include-disabled`.

Example:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only sparsegpt_opt125m_sparsegpt_smoke \
  --include-disabled
```

Other useful starting jobs:

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only sparsegpt_opt125m_gmp_smoke --include-disabled
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only wanda_opt125m_magnitude_smoke --include-disabled
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only wanda_opt125m_wanda_smoke --include-disabled
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only prunerzero_opt125m_smoke --include-disabled
```

## 5. Template Jobs Requiring Model Access

Some methods use LLaMA-family examples upstream. Set a model variable before running:

```bash
export FLAP_MODEL="your-accessible-llama-compatible-model"
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only flap_llama_official_template --include-disabled
```

```bash
export DSNOT_MODEL="your-accessible-llama-compatible-model"
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --only dsnot_llama_official_template --include-disabled
```

## 6. What To Send Back For Review

Usually I only need:

```text
reports/stage1/manual_run_results.jsonl
```

If a job failed, I may also inspect the referenced run directory, especially:

```text
results/raw/<run_id>/stderr.log
results/raw/<run_id>/stdout.log
results/raw/<run_id>/summary.json
results/raw/<run_id>/resource.csv
```

## 7. Notes

- Do not run all heavy jobs at once. Start with one job.
- The OPT-125M jobs are the lowest-risk first targets.
- Large LLaMA/Qwen-style jobs can download many GB and may take a long time.
- If a run is interrupted, the partial run directory and summary logs are still useful.

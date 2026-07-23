# Runbook

## Enter The Project

From PowerShell:

```powershell
wsl -d Ubuntu
```

Then in Ubuntu:

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
```

## List Planned Jobs

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --list
```

## Run One Planned Job

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only <job_id> \
  --include-disabled
```

Each planned job writes a timestamped directory under `results/raw/` and appends to the manual run results.

## Run Evaluation Helpers Directly

Generate samples:

```bash
.venv-common/bin/python scripts/eval/generate_evalplus_samples.py \
  --model <model_or_local_path> \
  --split <split.jsonl> \
  --out-dir <run_dir> \
  --max-new-tokens 256 \
  --dtype fp16 \
  --device cuda:0
```

Score HumanEval:

```bash
.venv-common/bin/python scripts/eval/score_humaneval_smoke.py \
  --split <split.jsonl> \
  --samples <run_dir>/samples.jsonl \
  --out-dir <run_dir> \
  --base-only
```

Score MBPP:

```bash
.venv-common/bin/python scripts/eval/score_mbpp_smoke.py \
  --split <split.jsonl> \
  --samples <run_dir>/samples.jsonl \
  --out-dir <run_dir> \
  --base-only
```

## R4 Rule

Before a method enters R4:

1. Its R2 pruning row must be complete.
2. Its R3 HumanEval/MBPP smoke rows must be complete.
3. The R4 run must use guide data during pruning, not only during final scoring.
4. Save resource metrics and artifact manifests.

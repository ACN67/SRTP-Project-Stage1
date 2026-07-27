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

## Run Official Benchmark Evaluation

Formal benchmark runs must follow `docs/official_benchmarks.md`. Use the wrapper
below unless you have a specific reason to call the underlying scripts directly.

HumanEval:

```bash
scripts/eval/run_official_eval.sh \
  --benchmark humaneval \
  --model <model_or_local_path> \
  --split data/splits/humaneval_half/eval.jsonl \
  --out-dir results/raw/<run_id>/humaneval \
  --local-files-only
```

MBPP:

```bash
scripts/eval/run_official_eval.sh \
  --benchmark mbpp_evalplus \
  --model <model_or_local_path> \
  --split data/splits/mbpp_evalplus_half/eval.jsonl \
  --out-dir results/raw/<run_id>/mbpp_evalplus \
  --local-files-only
```

LiveCodeBench:

```bash
scripts/eval/run_official_eval.sh \
  --benchmark livecodebench \
  --model <model_or_local_path> \
  --split data/splits/livecodebench_half/eval.jsonl \
  --out-dir results/raw/<run_id>/livecodebench \
  --local-files-only \
  --lcb-release release_v1 \
  --lcb-config release_latest \
  --lcb-lm-style CodeQwenInstruct
```

## R4 Rule

Before a method enters R4:

1. Its R2 pruning row must be complete.
2. Its R3 HumanEval/MBPP smoke rows must be complete.
3. The R4 run must use guide data during pruning, not only during final scoring.
4. Save resource metrics and artifact manifests.

R4 uses three fixed guide/eval benchmark splits:

```text
data/splits/humaneval_half/
data/splits/mbpp_evalplus_half/
data/splits/livecodebench_half/
```

HumanEval and LiveCodeBench are split approximately 50/50 with stratification. MBPP uses official split semantics: prompt/train/validation as guide and test as eval.

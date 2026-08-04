# Runbook

## Enter The Project

From PowerShell:

```powershell
wsl -d Ubuntu
```

Then in Ubuntu:

```bash
cd ~/projects/srtp-code-llm-pruning
source environment/setup/env.sh
```

## List Planned Jobs

```bash
workflows/experiment/run_plan.sh \
  --plan workflows/experiment/stage1_plan.yaml \
  --list
```

## Run One Planned Job

```bash
workflows/experiment/run_plan.sh \
  --plan workflows/experiment/stage1_plan.yaml \
  --only <job_id> \
  --include-disabled
```

Each planned job writes a timestamped directory under `results/evidence/` and appends to the manual run results.

## Run Official Benchmark Evaluation

Formal benchmark runs must follow `docs/official_benchmarks.md`. Use the wrapper
below unless you have a specific reason to call the underlying scripts directly.

HumanEval:

```bash
workflows/evaluate/run.sh \
  --benchmark humaneval \
  --model <model_or_local_path> \
  --split data/benchmarks/r4_half/humaneval/eval.jsonl \
  --out-dir results/evidence/<run_id>/humaneval \
  --local-files-only
```

MBPP:

```bash
workflows/evaluate/run.sh \
  --benchmark mbpp_evalplus \
  --model <model_or_local_path> \
  --split data/benchmarks/r4_half/mbpp_evalplus/eval.jsonl \
  --out-dir results/evidence/<run_id>/mbpp_evalplus \
  --local-files-only
```

LiveCodeBench:

```bash
workflows/evaluate/run.sh \
  --benchmark livecodebench \
  --model <model_or_local_path> \
  --split data/benchmarks/r4_half/livecodebench/eval.jsonl \
  --out-dir results/evidence/<run_id>/livecodebench \
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
data/benchmarks/r4_half/humaneval/
data/benchmarks/r4_half/mbpp_evalplus/
data/benchmarks/r4_half/livecodebench/
```

HumanEval and LiveCodeBench are split approximately 50/50 with stratification. MBPP uses official split semantics: prompt/train/validation as guide and test as eval.

# Long Benchmark Split Jobs

This document contains commands for benchmark data jobs that may take longer than Codex should supervise continuously.

## Enter Project

If you are in PowerShell:

```powershell
wsl -d Ubuntu
```

Then in Ubuntu:

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
```

## LiveCodeBench And SWE-bench Lite Smoke Splits

Run:

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only create_lcb_swebench_smoke_splits \
  --include-disabled
```

Expected output at the end:

```text
create_lcb_swebench_smoke_splits: success (...)
```

Files to check afterward:

```text
data/splits/livecodebench/guide.jsonl
data/splits/livecodebench/eval.jsonl
data/splits/livecodebench/manifest.json
data/splits/swebench_lite/guide.jsonl
data/splits/swebench_lite/eval.jsonl
data/splits/swebench_lite/manifest.json
reports/stage1/manual_run_results.jsonl
results/raw/<run_id>/
```

If it fails, keep the generated run directory and send the latest `manual_run_results.jsonl` line or the `stderr.log` path for review.

## Why This Is Long

The job may download LiveCodeBench and SWE-bench Lite datasets from Hugging Face. It does not run model inference or benchmark evaluation, but dataset download and conversion can still take time.

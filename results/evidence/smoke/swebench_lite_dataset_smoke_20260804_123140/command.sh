#!/usr/bin/env bash
set -euo pipefail
cd /home/keshu/projects/srtp-code-llm-pruning
/home/keshu/projects/srtp-code-llm-pruning-pre-final-original-20260804-1015/.venv-common/bin/python -c 'import json; from pathlib import Path; p=Path('"'"'data/benchmarks/smoke/swebench_lite/eval.jsonl'"'"'); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]; print([r.get('"'"'task_id'"'"') for r in rows]); assert len(rows)==4'

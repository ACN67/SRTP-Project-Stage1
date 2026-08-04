#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL=""
SPLIT="data/benchmarks/smoke/mbpp/eval.jsonl"
OUTPUT_DIR="${RUN_DIR:-results/evidence/mbpp_eval_manual}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv-common/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv-common/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

args=(
  "$PYTHON_BIN" workflows/evaluate/legacy/generate_code_split.py
  --benchmark mbpp
  --model "$MODEL"
  --split "$SPLIT"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry-run)
fi

"${args[@]}"

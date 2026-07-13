#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source scripts/setup/env.sh
source .venv-common/bin/activate

python scripts/run/run_experiment_plan.py "$@"

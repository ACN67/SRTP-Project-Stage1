#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source environment/setup/env.sh
source .venv-common/bin/activate

python workflows/experiment/run_plan.py "$@"

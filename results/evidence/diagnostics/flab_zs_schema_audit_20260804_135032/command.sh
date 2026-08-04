#!/usr/bin/env bash
set -euo pipefail
python workflows/audit/check_flab_benchmark_guided.py --schema-audit

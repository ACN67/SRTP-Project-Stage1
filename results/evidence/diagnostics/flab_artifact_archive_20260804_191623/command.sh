#!/usr/bin/env bash
set -euo pipefail
# Archive existing Flab benchmark-guided artifacts if present; otherwise record missing /tmp availability.
python workflows/audit/check_keshu_final_cleanup.py --artifact-archive

#!/usr/bin/env bash
set -euo pipefail
cd /home/keshu/projects/srtp-code-llm-pruning
/home/keshu/projects/srtp-code-llm-pruning-pre-final-original-20260804-1015/.venv-common/bin/python -c 'from pathlib import Path; assert Path('"'"'third_party/laco/laco_llama-13b.ipynb'"'"').exists(); print('"'"'laco notebook present'"'"')'

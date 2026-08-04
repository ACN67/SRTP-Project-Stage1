#!/usr/bin/env bash
set -euo pipefail
pytest -q tests/test_flab_guide_contract.py tests/test_flab_zs_adapter.py tests/test_flab_tiny_prune.py tests/test_flab_guide_causality.py tests/test_flab_artifact_reload.py tests/test_flab_success_validator.py tests/test_flab_scope.py

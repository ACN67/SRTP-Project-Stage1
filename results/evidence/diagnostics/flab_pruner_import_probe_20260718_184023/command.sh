set -euo pipefail && cd /home/keshu/projects/srtp-code-llm-pruning && source scripts/setup/env.sh && source .venv-flab_pruner/bin/activate && cd /home/keshu/projects/srtp-code-llm-pruning/third_party/flab_pruner && python - <<'PY'
from hidden_prune_utils.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config
import ffn_prune, vocab_prune
print('flab_pruner_qwen2_import_ok')
print('model_class', Qwen2ForCausalLM.__name__)
print('config_class', Qwen2Config.__name__)
PY

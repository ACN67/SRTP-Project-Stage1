set -euo pipefail && cd /home/keshu/projects/srtp-code-llm-pruning && source scripts/setup/env.sh && source .venv-common/bin/activate && cd /home/keshu/projects/srtp-code-llm-pruning && .venv-wanda/bin/python -c "import torch, transformers, datasets; print('wanda', torch.cuda.is_available(), torch.__version__, transformers.__version__)"
.venv-sparsegpt/bin/python -c "import torch, transformers, datasets; print('sparsegpt', torch.cuda.is_available(), torch.__version__, transformers.__version__)"
.venv-llm_pruner/bin/python -c "import torch, transformers, peft; print('llm_pruner', torch.cuda.is_available(), torch.__version__, transformers.__version__)"
.venv-slicegpt/bin/python -c "import torch, transformers, slicegpt; print('slicegpt', torch.cuda.is_available(), torch.__version__, transformers.__version__)"
.venv-livecodebench/bin/python -c "import torch, lcb_runner; print('livecodebench', torch.cuda.is_available(), torch.__version__)"


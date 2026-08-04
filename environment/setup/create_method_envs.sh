#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
UV="${UV:-$HOME/.local/bin/uv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
base_packages=(transformers accelerate datasets evaluate safetensors sentencepiece protobuf tokenizers huggingface_hub pandas numpy scipy scikit-learn pyyaml jsonlines psutil tqdm rich einops matplotlib seaborn peft ml-collections deap nvitop pytest)
create_env() {
  local name="$1" env_dir=".venv-${name}"
  if [ ! -d "$env_dir" ]; then "$UV" venv --python "$PYTHON_VERSION" "$env_dir"; fi
  "$UV" pip install --python "$env_dir/bin/python" --index-url "$TORCH_INDEX" torch torchvision torchaudio
  "$UV" pip install --python "$env_dir/bin/python" "${base_packages[@]}"
}
for name in common magnitude wanda sparsegpt llm_pruner dsnot maskllm flap laco owl pruner_zero flab_pruner; do create_env "$name"; done
if [ ! -d .venv-slicegpt ]; then "$UV" venv --python "$PYTHON_VERSION" .venv-slicegpt; fi
"$UV" pip install --python .venv-slicegpt/bin/python --index-url "$TORCH_INDEX" torch torchvision torchaudio
"$UV" pip install --python .venv-slicegpt/bin/python -e third_party/slicegpt
if [ ! -d .venv-livecodebench ]; then "$UV" venv --python "$PYTHON_VERSION" .venv-livecodebench; fi
"$UV" pip install --python .venv-livecodebench/bin/python --index-url "$TORCH_INDEX" torch torchvision torchaudio
"$UV" pip install --python .venv-livecodebench/bin/python -e third_party/livecodebench --no-deps
"$UV" pip install --python .venv-livecodebench/bin/python annotated-types anthropic cohere datasets google-genai mistralai==0.4.2 openai pebble together

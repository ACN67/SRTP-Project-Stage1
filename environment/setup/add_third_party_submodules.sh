#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

repos=(
  "wanda https://github.com/locuslab/wanda.git"
  "sparsegpt https://github.com/IST-DASLab/sparsegpt.git"
  "llm_pruner https://github.com/horseee/LLM-Pruner.git"
  "dsnot https://github.com/zyxxmu/DSnoT.git"
  "maskllm https://github.com/NVlabs/MaskLLM.git"
  "flap https://github.com/CASIA-LMC-Lab/FLAP.git"
  "slicegpt https://github.com/microsoft/TransformerCompression.git"
  "laco https://github.com/yangyifei729/LaCo.git"
  "owl https://github.com/luuyin/OWL.git"
  "pruner_zero https://github.com/pprp/Pruner-Zero.git"
  "flab_pruner https://github.com/Flab-Pruner/Flab-Pruner.git"
  "livecodebench https://github.com/livecodebench/livecodebench.git"
)

for item in "${repos[@]}"; do
  name="${item%% *}"
  url="${item#* }"
  if git config --file .gitmodules --get-regexp "submodule.third_party/${name}.url" >/dev/null 2>&1; then
    echo "exists third_party/${name}"
  elif [ -e "third_party/${name}" ]; then
    echo "skip existing path third_party/${name}"
  else
    echo "adding ${name}"
    git submodule add "$url" "third_party/${name}"
  fi
done

mkdir -p data/manifests
{
  echo "method,upstream_repo,upstream_commit"
  for item in "${repos[@]}"; do
    name="${item%% *}"
    url="${item#* }"
    commit="$(git -C "third_party/${name}" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
    echo "${name},${url},${commit}"
  done
} > data/manifests/upstream_repos.csv

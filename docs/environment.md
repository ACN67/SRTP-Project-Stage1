# Environment And Artifacts

## Local Machine

| Item | Value |
|---|---|
| OS | Windows + WSL2 Ubuntu |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | about 8 GiB |
| WSL memory | about 13 GiB RAM + 32 GiB swap |
| Main model cache | `/home/keshu/.cache/huggingface/` |

The CodeLlama-7B Hugging Face snapshot is kept in the local cache and is not part of Git.

## Python Environments

Per-method virtual environments live at repository root:

```text
.venv-common
.venv-flab_pruner
.venv-llm_pruner
.venv-slicegpt
...
```

Dependency snapshots are stored under `env/<name>/pip_freeze.txt`.

## Local-Only Artifacts

Large model files are ignored by `.gitignore`:

- `results/**/*.safetensors`
- `results/**/*.pt`
- `results/**/tokenizer.json`
- `results/**/offload/`
- `results/**/tmp/`

Current useful local artifact:

```text
results/raw/flab_qwen3b_humaneval_prune_heavy_20260718_204006/flab_qwen3b_humaneval/pruned_model/
```

The directory contains the Flab-Pruner Qwen3B smoke-stage pruned model. Keep it local unless a formal artifact storage policy is adopted.

## Disk Notes

The largest local storage users are model caches and local pruned weights. Raw text logs are small by comparison.

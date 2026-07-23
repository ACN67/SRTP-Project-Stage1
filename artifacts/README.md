# Local Artifacts

This project keeps large model artifacts local by default.

The Git repository stores commands, summaries, scores, logs, hashes, and manifests. It does not store model weight files such as `.safetensors`, `.pt`, tokenizer dumps, or temporary offload folders.

## Current Local Artifacts

| Artifact | Local path | Git policy |
|---|---|---|
| Flab-Pruner Qwen3B pruned model | `results/raw/flab_qwen3b_humaneval_prune_heavy_20260718_204006/flab_qwen3b_humaneval/pruned_model/` | Keep local; ignored by `.gitignore`. |
| CodeLlama-7B HF snapshot | `/home/keshu/.cache/huggingface/hub/models--codellama--CodeLlama-7b-hf/` | Keep in Hugging Face cache; not part of repo. |
| SliceGPT OPT-125M smoke `.pt` artifact | `results/raw/slicegpt_opt125m_smoke_20260718_173646/sliced_model/opt-125m_0.1.pt` | Removed locally after hash was recorded; not needed for R4. |


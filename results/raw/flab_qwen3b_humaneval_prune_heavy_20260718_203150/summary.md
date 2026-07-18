# Flab-Pruner Qwen3B HumanEval Heavy Run

Status: failed

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide split: data/splits/humaneval/guide.jsonl
Run ID: flab_qwen3b_humaneval_prune_heavy_20260718_203150

What worked:

- The previous `_tied_weights_keys` issue was fixed.
- Qwen3B weights loaded successfully from cache.
- The run reached `model.prune()`.
- GPU memory rose to about 7.5 GB, confirming that the model entered the loading/pruning phase.

Failure point:

- Flab-Pruner failed inside `prune_linear_by_index` while replacing linear layer weights.
- Error: `RuntimeError: Attempted to call variable.set_data(tensor), but variable and tensor have incompatible tensor type.`
- Category: DEVICE/OFFLOAD compatibility.

Root cause:

The heavy job used `device_map=auto`, which offloaded some model parameters to CPU/meta tensors. Flab-Pruner's vendored Qwen2 pruning code mutates `module.weight.data` in place and is not compatible with offloaded/meta tensors.

Fix applied after this run:

- Changed `flab_qwen3b_humaneval_prune_heavy` to use `--device-map cuda:0`.
- Updated `docs/RUN_FLAB_QWEN3B_HEAVY.md` to explain why `auto` is avoided for this method.

Risk for next run:

Loading the full Qwen3B model on `cuda:0` may expose a real CUDA OOM. If that happens, record the failure and switch to a CPU/offline pruning strategy or lower-memory variant.

# Flab-Pruner Qwen3B HumanEval Heavy Run

Status: failed

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide split: data/splits/humaneval/guide.jsonl
Run ID: flab_qwen3b_humaneval_prune_heavy_20260718_194356

What worked:

- Qwen3B files were fetched from Hugging Face cache/HF Hub.
- Guide split was validated and recorded.
- Prune plan was written to `flab_qwen3b_humaneval/flab_qwen_prune_plan.json`.

Failure point:

- The upstream Flab-Pruner vendored Qwen2 model failed during model instantiation.
- Error: `AttributeError: 'Qwen2Config' object has no attribute 'rope_theta'`.
- Category: CODE/CONFIG compatibility.

Root cause:

Flab-Pruner's local `hidden_prune_utils/modeling_qwen2.py` expects `config.rope_theta`, but Qwen/Qwen2.5-Coder-3B-Instruct as loaded in the current environment does not expose that field.

Fix applied after this run:

- Updated `scripts/adapt/flab_qwen_prune.py` to patch missing Qwen2 compatibility config fields before passing config into Flab's vendored model class.
- Re-ran dry-run successfully in `flab_qwen3b_humaneval_dry_run_20260718_201552`.

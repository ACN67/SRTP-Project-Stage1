# Flab-Pruner Qwen3B HumanEval Heavy Run

Status: failed

Method: Flab-Pruner
Owner: 常珂舒
Model: Qwen/Qwen2.5-Coder-3B-Instruct
Guide split: data/splits/humaneval/guide.jsonl
Run ID: flab_qwen3b_humaneval_prune_heavy_20260718_202722

What worked:

- The previous missing `rope_theta` issue was fixed.
- The wrapper applied Qwen2 compatibility patches:
  - rope_theta: 1000000.0
  - sliding_window: null
- Guide split was validated and recorded.
- Prune plan was written to `flab_qwen3b_humaneval/flab_qwen_prune_plan.json`.

Failure point:

- The upstream Flab-Pruner vendored Qwen2 model failed during `post_init()` under Transformers 5.
- Error: `AttributeError: 'list' object has no attribute 'keys'`.
- Category: CODE/TRANSFORMERS compatibility.

Root cause:

Flab's vendored Qwen2 class uses the older Transformers convention where `_tied_weights_keys` is a list. Transformers 5 expects `_tied_weights_keys` to be a mapping such as `{"lm_head.weight": "model.embed_tokens.weight"}`.

Fix applied after this run:

- Updated `scripts/adapt/flab_qwen_prune.py` to patch `Qwen2ForCausalLM._tied_weights_keys` before calling `from_pretrained()`.

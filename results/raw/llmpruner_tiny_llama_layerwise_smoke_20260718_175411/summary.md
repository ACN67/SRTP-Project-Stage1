# LLM-Pruner Tiny LLaMA Layer-Wise Smoke

Status: failed

Method: LLM-Pruner
Owner: 常珂舒
Model: trl-internal-testing/tiny-random-LlamaForCausalLM
Run ID: llmpruner_tiny_llama_layerwise_smoke_20260718_175411

What worked:

- Tiny LLaMA tokenizer and model loaded.
- LLM-Pruner entered the l2 pruner path.
- Layer-wise pruning changed parameter count:
  - Before: 1,032,272
  - After: 1,028,144
  - Ratio: 99.6001%

Failure point:

- Final PPL evaluation failed while loading `ptb_text_only`.
- Error category: ENV/DATA
- Root cause: the installed `datasets` package was too new and no longer supported dataset scripts.

Fix applied after this run:

- Downgraded `datasets` in `.venv-llm_pruner` to `2.18.0`.
- Added `pyarrow-hotfix`.

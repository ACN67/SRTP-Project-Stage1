# LaCo Upstream Audit

Owner scope: Keshu-owned method evidence only.

Upstream path: `third_party/laco/laco_llama-13b.ipynb`
Upstream commit: `84835877f918dc972df1a1375e09209551e02f3d`
Notebook exists: `true`
Dependencies observed: `torch, transformers`
Model class: `unknown`
Hardcoded model route: `unknown`
Data needs: `unknown`
Scoring: `adjacent layer similarity`
Merge/collapse: `layer merge/collapse over transformer blocks`
Saving: `unknown`
Evaluation: `unknown`
Minimal executable unit: `methods/laco/run_smoke.py`

Conclusion: the previous file-presence probe is diagnostic only. The current smoke executes the core adjacent-layer similarity and collapse path with a tiny LLaMA-compatible model, but it is not formal CodeLlama R4 evidence.

# SliceGPT CodeLlama-7B Minimal Prune Smoke

Status: success

Method: SliceGPT
Owner: 常珂舒
Model: codellama/CodeLlama-7b-hf
Run ID: slicegpt_codellama7b_min_prune_smoke_20260722_235632

This run confirms that SliceGPT can complete a minimal CodeLlama-7B rotate-and-slice pruning smoke using CPU-resident weights and GPU layer work.

Key result:

- Hidden size: 4096 -> 4048
- Requested sparsity: 0.01
- Calibration: wikitext2, 1 sample, sequence length 32
- Rotate-and-slice completed for all 32 layers in 12m39s.
- Total wrapper elapsed time: 1291.040 seconds

Note:

The naive parameter count increased after replacing/fusing modules because SliceGPT adds shortcut rotation parameters in the compressed layer representation. For this run, use hidden-size reduction and runtime/resource records as smoke evidence rather than treating `parameter_reduction_rate` as a valid compression metric.

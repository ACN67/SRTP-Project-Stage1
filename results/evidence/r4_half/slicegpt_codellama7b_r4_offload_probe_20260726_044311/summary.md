# SliceGPT CodeLlama-7B R4 Offload Probe

Status: success

Method: SliceGPT
Owner: Chang Keshu
Model: codellama/CodeLlama-7b-hf
Run ID: slicegpt_codellama7b_r4_offload_probe_20260726_044311

This probe verified benchmark-guide calibration on HumanEval/MBPP/LiveCodeBench guide-half samples, 20% requested SliceGPT sparsity, and one HumanEval eval-half generation with GPU+CPU offload.

Key result:

- Calibration guide samples: 3
- Requested sparsity: 0.20
- Actual parameter keep ratio: 0.9061746335910298
- One HumanEval generation completed in 193.119 seconds
- Total elapsed time: 1447.121 seconds

Retention note:

- The sliced model state file and offload folder were deleted after the probe to save disk space.
- The small logs, summary, slicing config, generated sample, and generation record are retained.

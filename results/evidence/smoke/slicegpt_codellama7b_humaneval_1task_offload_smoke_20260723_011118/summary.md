# SliceGPT CodeLlama-7B HumanEval 1-task Offload Smoke

Status: partial_success

Method: SliceGPT
Owner: 常珂舒
Model: codellama/CodeLlama-7b-hf
Benchmark: HumanEval smoke diagnostic
Split: data/splits/humaneval/eval.jsonl
Run ID: slicegpt_codellama7b_humaneval_1task_offload_smoke_20260723_011118

This run confirms that SliceGPT CodeLlama-7B can generate one HumanEval smoke sample after minimal rotate-and-slice pruning using GPU/CPU offload generation.

Key result:

- Generated task: HumanEval/101
- Generation time: 629.704 seconds
- Base pass count on 4-task split-aware scorer: 0/4, with ungenerated tasks counted as missing
- Failure after generation: cleanup attempted to move an accelerate-offloaded model with meta tensors back to CPU
- Fix applied after this run: the wrapper now deletes offloaded model references instead of copying meta tensors back to CPU

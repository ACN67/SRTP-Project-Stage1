# LLM-Pruner CodeLlama-7B Layer-Wise CPU Smoke

Status: partial_success

Method: LLM-Pruner
Owner: 常珂舒
Model: codellama/CodeLlama-7b-hf
Run ID: llmpruner_codellama7b_layerwise_cpu_smoke_20260722_222955

This run attempted a minimal unquantized CodeLlama-7B layer-wise pruning workflow through LLM-Pruner on CPU.

What completed:

- CodeLlama-7B checkpoint loaded from the explicit local Hugging Face snapshot.
- LLM-Pruner entered the `l2` pruner path.
- Layer-wise pruning completed by keeping 31 of 32 layers.
- Parameters changed from 6,738,546,688 to 6,536,163,328.
- Parameter keep ratio was 96.9966%, so the reduction rate was about 3.0034%.

What did not complete:

- The upstream script then entered mandatory PPL evaluation on CPU.
- It reached 1/1334 wikitext2 samples after about 646.78 seconds for the first sample.
- The run was manually stopped because full CPU PPL evaluation was projected to take far too long.

Interpretation:

- This is valid R2 evidence that the unquantized CodeLlama-7B LLM-Pruner layer-wise pruning path can run locally through the pruning step.
- It is not an R3 eval completion because PPL and downstream benchmark evaluation did not finish.
- Next step should patch or wrap LLM-Pruner to skip mandatory full PPL or limit PPL samples for smoke/R3 runs.

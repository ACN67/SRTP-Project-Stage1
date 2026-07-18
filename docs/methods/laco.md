# LaCo Method Notes

Owner: 常珂舒  
Group: structured_depth_width  
Stage: 1  
Status: R1 blocked by upstream form

## Upstream State

LaCo is provided as an official repository, but the current upstream code is notebook-only.

Observed files:

```text
third_party/laco/LICENSE
third_party/laco/README.md
third_party/laco/laco-baic-13B.ipynb
third_party/laco/laco_llama-13b.ipynb
```

No standalone Python entry point was found under `third_party/laco`.

The upstream README describes the implementation as preliminary and points users to the LLaMA2 and Baichuan2 notebooks.

## Reproduction Assessment

For Stage 1, this means LaCo can be analyzed and documented, but it is not suitable for the same scripted R1 smoke workflow used for SliceGPT and LLM-Pruner without converting the notebook logic into a script.

Current decision:

- Do not run a heavy LaCo experiment in Stage 1.
- Record LaCo as method analyzed.
- Record R1 reproduction as blocked by upstream notebook-only implementation.
- Keep future work focused on extracting a scriptable minimal workflow if LaCo becomes necessary for Stage 2.

## Suggested Future Work

A future LaCo reproduction script should:

- Convert the LLaMA notebook into a Python entry point.
- Replace large gated models with an accessible tiny LLaMA-compatible model where possible.
- Add CLI arguments for model path, layer-removal ratio, output directory, and evaluation mode.
- Reuse the existing Stage 1 runner so results land under `results/raw/`.

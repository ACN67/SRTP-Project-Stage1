# Raw Result Archive

`results/raw/` contains timestamped raw outputs from experiment runs.

Use `results/stage1/` for normal progress review. Raw directories are retained as evidence for commands, logs, generated samples, score details, and resource traces.

Large model artifacts are local-only and ignored:

- `*.safetensors`
- `*.pt`
- `tokenizer.json`
- `offload/`
- `tmp/`


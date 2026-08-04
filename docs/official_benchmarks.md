# Official Benchmark Policy

This project treats benchmark evaluation as a formal experimental interface. New
paper-facing results must use the scripts in `workflows/evaluate/` documented here.
Historical logs under `results/evidence/` are retained as evidence, but old smoke or
diagnostic commands are not valid sources for final tables.

## Sources Of Authority

| Benchmark | Official source used by this repo | Prompt policy | Scoring policy |
|---|---|---|---|
| HumanEval | OpenAI HumanEval task format, evaluated through EvalPlus HumanEval | Use the raw task prompt from the split, which is generated from the official HumanEval problem prompt. | Use EvalPlus `humaneval` correctness checks through `score_humaneval_official.py`. |
| MBPP | EvalPlus MBPP task format with sanitized MBPP official split semantics | Use the raw EvalPlus MBPP prompt from the split. | Use EvalPlus `mbpp` correctness checks through `score_mbpp_evalplus_official.py`. |
| LiveCodeBench | Local LiveCodeBench `lcb_runner` code-generation implementation | Use `format_prompt_generation(problem, LMStyle.<style>)`; Qwen2.5-Coder defaults to `CodeQwenInstruct`. | Use LiveCodeBench `extract_code` and `codegen_metrics` through `score_livecodebench_official.py`. |

Reference URLs:

- HumanEval official repository: `https://github.com/openai/human-eval`
- EvalPlus official repository and CLI: `https://github.com/evalplus/evalplus`
- LiveCodeBench official repository: `https://github.com/LiveCodeBench/LiveCodeBench`

If a later paper or upstream benchmark release provides a newer public script,
prefer the benchmark's official repository first. Use paper-specific public
scripts only when they call the same official benchmark prompt/scoring APIs, or
when the paper explicitly defines a variant being reproduced.

## Canonical Scripts

| Script | Role |
|---|---|
| `workflows/evaluate/generate.py` | Generate samples with benchmark-specific official prompts and record audit fields. |
| `workflows/evaluate/score_humaneval.py` | Score HumanEval split rows with EvalPlus. |
| `workflows/evaluate/score_mbpp.py` | Score MBPP EvalPlus split rows with EvalPlus. |
| `workflows/evaluate/score_livecodebench.py` | Score LiveCodeBench split rows with LiveCodeBench metrics. |
| `workflows/evaluate/run.sh` | One-command wrapper that selects the correct official generation and scoring path. |

## Required Splits

Formal Stage 1 pruning/recovery results use:

```text
data/benchmarks/r4_half/humaneval/
data/benchmarks/r4_half/mbpp_evalplus/
data/benchmarks/r4_half/livecodebench/
```

Guide split rows may be used for pruning calibration and LoRA recovery data.
Eval split rows are reserved for final scoring. Do not train, calibrate, tune
prompts, select checkpoints, or choose decoding settings using eval rows.

## Decoding Rules

Default formal generation is greedy decoding:

```text
do_sample = False
temperature = not set
top_p = not set
```

The generator decodes only newly generated tokens, records the actual model
prompt in `model_prompt`, and writes both `raw_completion` and the benchmark
extracted `completion`. `completion_chars <= 1` is an audit warning, not a value
to patch by prompt rewriting.

## What Is Not Formal

Do not use the following for new paper-facing results:

- Smoke split generation commands.
- Diagnostic prompt wrappers that are not in the official benchmark code.
- Re-extraction scripts used to repair historical logs.
- Result directories produced before the official prompt/scoring policy was
  introduced, unless they are explicitly labeled as historical diagnostics.

## Minimal Examples

HumanEval:

```bash
workflows/evaluate/run.sh \
  --benchmark humaneval \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --split data/benchmarks/r4_half/humaneval/eval.jsonl \
  --out-dir results/evidence/example_humaneval_official \
  --local-files-only
```

MBPP:

```bash
workflows/evaluate/run.sh \
  --benchmark mbpp_evalplus \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --split data/benchmarks/r4_half/mbpp_evalplus/eval.jsonl \
  --out-dir results/evidence/example_mbpp_official \
  --local-files-only
```

LiveCodeBench:

```bash
workflows/evaluate/run.sh \
  --benchmark livecodebench \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --split data/benchmarks/r4_half/livecodebench/eval.jsonl \
  --out-dir results/evidence/example_livecodebench_official \
  --local-files-only \
  --lcb-release release_v1 \
  --lcb-config release_latest \
  --lcb-lm-style CodeQwenInstruct
```

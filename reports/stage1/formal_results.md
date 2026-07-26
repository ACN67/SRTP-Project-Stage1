# Formal Comparison

Source run: `results/raw/pan_formal_20260724_203248` (WSL GPU host)

## Split protocol note（重要）

本表 **不是** 按 GitHub 上已上传的 R4 半集 `*_half` 跑的：

| Split | Guide / Eval | 用途 |
|---|---|---|
| Hub 已有 `humaneval_half` / `mbpp_evalplus_half` / `livecodebench_half` | 82/82、154/224、200/200 | Stage1 R4 引导半集—验证半集 |
| 本 formal 用的 `humaneval_formal` / `mbpp_formal` | guide **32**；eval 为 **全量** HE164 / MBPP sanitized257 | 剪枝校准小 guide + 全量 Pass@1 |
| 本 formal 打分用的 MBPP | `mbpp_evalplus_full`（378，evalplus ID） | evalplus Pass@1 |

因此：组长 Flab 截图里 HE82 / MBPP224 / LCB200 才是半集 eval 规模；本表数字不能与半集结果直接横比。

## OPT-125M WikiText PPL (nsamples=128)

| Method | Sparsity | PPL |
|---|---:|---:|
| DSnoT | 0.3 | 28.201339721679688 |
| DSnoT | 0.5 | 41.66608428955078 |
| Magnitude | 0.3 | 34.51060485839844 |
| Magnitude | 0.5 | 193.34051513671875 |
| Wanda | 0.3 | 28.109424591064453 |
| Wanda | 0.5 | 38.96297073364258 |

## Qwen2.5-Coder-1.5B-Instruct Pass@1 (evalplus)

| Method | Sparsity | Calib | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---|---:|---:|---:|---:|
| Dense | 0.0 | none | 0.152 | 0.134 | 0.659 | 0.561 |
| Magnitude | 0.10 | humaneval_formal/guide | 0.165 | 0.140 | 0.664 | 0.561 |
| Magnitude | 0.10 | mbpp_formal/guide | 0.165 | 0.140 | 0.664 | 0.561 |
| Magnitude | 0.30 | humaneval_formal/guide | 0.134 | 0.122 | 0.601 | 0.513 |
| Wanda | 0.10 | humaneval_formal/guide | 0.152 | 0.134 | 0.680 | 0.577 |
| Wanda | 0.10 | mbpp_formal/guide | 0.152 | 0.134 | 0.669 | 0.571 |
| Wanda | 0.30 | humaneval_formal/guide | 0.159 | 0.140 | 0.648 | 0.550 |

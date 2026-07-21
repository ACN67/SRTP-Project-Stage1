# SRTP 第一阶段执行书 V3

## 面向软件工程 Agent 的代码大模型多范式剪枝复现

**阶段名称：** 第一阶段：多方法复现、模型策略定型与统一实验流程搭建  
**执行成员：** 常珂舒、潘阔、李长骏  
**设备条件：** 常珂舒 RTX 5060；潘阔 RTX 4060；李长骏 RTX 4060  
**统一环境：** Windows + WSL2 + Git LFS + Hugging Face cache  
**现行模型策略：** 按方法官方支持选择实验模型  
**文档状态：** 当前执行依据  
**更新日期：** 2026-07-22

---

## 1. 阶段定位

第一阶段的目标不是直接证明哪种剪枝方法最好，而是建立一个可复现、可审计、可扩展的实验闭环。

本阶段必须回答以下问题：

1. 各剪枝方法的官方代码、依赖、入口和最小复现实验能否跑通；
2. 每种方法应使用哪个代码模型族，避免强行适配不支持的模型；
3. 剪枝前后模型是否能保存、加载、生成和进入统一评测流程；
4. HumanEval、MBPP、LiveCodeBench、SWE-bench Lite 是否能形成 guide/eval 数据接口；
5. 结果是否完整保存模型、日志、参数变化、运行时长、显存占用和评测分数；
6. 一名成员的实验能否由其他成员依据仓库材料复跑。

第一阶段的验收逻辑是：

```text
官方流程可运行
→ 方法适配模型选择合理
→ 至少一个代表方法完成剪枝闭环
→ benchmark guide/eval 管线接通
→ 资源与能力指标完整入库
→ 后续可扩展到正式评测
```

Smoke 结果只用于证明流程跑通，不作为正式性能结论。

---

## 2. 核心策略调整

早期执行路线曾尝试让所有方法都适配 Qwen2.5-Coder。经过审计后，这一路线不再作为全局要求。

新的模型选择原则如下：

1. **能用 CodeLlama 的方法优先使用 CodeLlama。**  
   适用于官方代码支持 LLaMA-family Hugging Face 模型的方法，例如 LLM-Pruner、SliceGPT、FLAP，以及经确认支持 LLaMA 路线的 Wanda、SparseGPT、DSnoT、OWL、MaskLLM、Pruner-Zero。

2. **明确支持 Qwen/Qwen2 的方法继续使用 Qwen2.5-Coder。**  
   当前代表方法是 Flab-Pruner。它上游包含 Qwen2 相关实现，本项目已完成 Qwen2.5-Coder-3B 剪枝闭环。

3. **既不能稳定使用 CodeLlama，也没有 Qwen 支持的方法，使用官方支持模型。**  
   例如 OPT、Phi、官方 tiny model 或方法 README 指定的小模型。

4. **notebook-only 或入口不稳定的方法可以跳过。**  
   LaCo 当前仅保留 R0/R1 blocker 证据，暂不投入 notebook 转脚本。

现行模型策略的机器可读表为：

```text
data/manifests/stage1_model_policy.csv
```

详细指标口径见：

```text
docs/stage1_model_selection_and_metrics.md
```

---

## 3. 方法范围与分工

| 成员 | 方法范围 | 当前模型策略 |
|---|---|---|
| 常珂舒 | LLM-Pruner、SliceGPT、LaCo、Flab-Pruner | Flab-Pruner 用 Qwen2.5-Coder；LLM-Pruner/SliceGPT 转 CodeLlama；LaCo 跳过 |
| 潘阔 | Magnitude、Wanda、DSnoT、OWL、benchmark 数据 | 能用 CodeLlama 则用 CodeLlama；否则保留官方支持模型 |
| 李长骏 | SparseGPT、MaskLLM、Pruner-Zero、FLAP、环境/资源审计 | 能用 CodeLlama 则用 CodeLlama；否则保留官方支持模型 |

方法覆盖表：

| 方法 | 类别 | 第一阶段定位 |
|---|---|---|
| Magnitude | 最低基线 | 非结构化剪枝最低对照 |
| Wanda | 激活感知剪枝 | 激活强基线 |
| SparseGPT | 二阶重构剪枝 | 重构强基线 |
| LLM-Pruner | 结构化剪枝 | LLaMA-family 结构化代表 |
| DSnoT | 掩码优化 | 与基础剪枝器组合 |
| MaskLLM | 半结构化剪枝 | N:M / 掩码学习路线 |
| FLAP | 训练自由结构化剪枝 | 结构化代表 |
| SliceGPT | 隐藏维度压缩 | LLaMA-family 宽度压缩代表 |
| LaCo | 层合并 | notebook-only，第一阶段跳过 |
| OWL | 非均匀稀疏率分配 | 层间分配策略 |
| Pruner-Zero | 自动搜索重要性指标 | 自动化剪枝路线 |
| Flab-Pruner | 代码模型剪枝 | Qwen2.5-Coder 代表闭环 |

---

## 4. R0-R3 定义

### R0：官方材料与环境审计

目标：

- 确认官方仓库、commit、README、入口脚本；
- 建立或检查对应 Python 环境；
- 确认官方支持模型族；
- 记录依赖缺失、入口缺失、notebook-only 等 blocker。

产物：

```text
docs/methods/<method>.md
reports/failures/<method>_*.md
results/raw/<method>_audit_*/audit.log
```

### R1：官方最小复现

目标：

- 使用官方推荐或官方支持的小模型跑通最小剪枝；
- 不强行换成 Qwen 或 CodeLlama；
- 证明 upstream 路线和本机环境能执行。

产物：

```text
results/raw/<method>_<official_model>_smoke_*/
summary.md
summary.json
resource.csv
resource_summary.json
```

### R2：方法族代码模型剪枝

目标：

- 根据现行模型策略，选择方法适合的代码模型族；
- LLaMA-compatible 方法优先 CodeLlama；
- Qwen-supported 方法使用 Qwen2.5-Coder；
- 完成最小剪枝、保存或记录 artifact manifest、重载或生成检查。

不再要求所有方法都 Qwen adapter。

### R3：benchmark guide/eval 闭环

目标：

- 使用 HumanEval、MBPP、LiveCodeBench、SWE-bench Lite 的 guide/eval split；
- guide split 用于剪枝阶段的校准、激活、梯度、隐藏状态或候选选择；
- eval split 用于剪枝后验证；
- 保存 baseline 与 pruned 结果，用统一指标计算能力保留率和资源减负率。

---

## 5. 模型选择规则

优先级：

```text
CodeLlama-family
→ Qwen2.5-Coder
→ 官方支持模型
→ blocked/skipped
```

但优先级受方法支持边界约束：

| 方法状态 | 选择 |
|---|---|
| 官方 LLaMA-compatible | CodeLlama |
| 官方 Qwen/Qwen2-compatible | Qwen2.5-Coder |
| 官方仅 OPT/Phi/tiny model | 官方模型 |
| notebook-only | 跳过或单独立项脚本化 |

建议模型：

| 短名 | 模型 ID | 用途 |
|---|---|---|
| `qwen25c15b` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Flab-Pruner Qwen baseline |
| `qwen25c3b` | `Qwen/Qwen2.5-Coder-3B-Instruct` | Flab-Pruner Qwen 剪枝目标 |
| `codellama7b` | `codellama/CodeLlama-7b-hf` | LLaMA-compatible 方法主目标 |
| `opt125m` | `facebook/opt-125m` | OPT 路线 smoke |
| `tinyllama_random` | `trl-internal-testing/tiny-random-LlamaForCausalLM` | LLM-Pruner 极小 R1 smoke |

如果 CodeLlama-7B 在 8GB 显存上无法完成，应先记录失败与资源瓶颈，再考虑更小 LLaMA-compatible 替代模型。

---

## 6. Benchmark 数据策略

第一阶段保留四类 benchmark：

| Benchmark | 用途 |
|---|---|
| HumanEval | 代码生成函数题 smoke / formal eval |
| MBPP | 代码生成函数题 smoke / formal eval |
| LiveCodeBench | 后续更真实代码任务 |
| SWE-bench Lite | 后续软件工程任务 |

数据分为：

```text
guide split：剪枝阶段使用
eval split：评测阶段使用
```

已有 smoke split：

```text
data/splits/humaneval/
data/splits/mbpp/
data/splits/mbpp_evalplus/
data/splits/livecodebench/
data/splits/swebench_lite/
```

MBPP 注意：

- 原始 `data/splits/mbpp/` 保留自然语言任务；
- EvalPlus 打分使用 `data/splits/mbpp_evalplus/`；
- 这是数据接口对齐，不是修改题目。

---

## 7. 指标与结果保留

每个用于最终汇总的实验必须尽量保存以下原始指标：

| 指标 | 说明 |
|---|---|
| `duration_sec` | 运行总时长 |
| `peak_gpu_memory_used_mb` | 峰值 GPU 显存 |
| `peak_process_rss_mb` | 进程峰值内存 |
| `parameter_count_before` | 剪枝前参数量 |
| `parameter_count_after` | 剪枝后参数量 |
| `artifact_size_bytes_before` | 剪枝前模型文件大小 |
| `artifact_size_bytes_after` | 剪枝后模型文件大小 |
| `task_count` | 评测题数 |
| `pass_count` | 通过题数 |
| `pass_rate` | 通过率 |

派生指标：

| 指标 | 公式 |
|---|---|
| 能力保留率 | `pruned_pass_rate / baseline_pass_rate`，baseline 为 0 时记为 `N/A` |
| 参数减负率 | `1 - parameter_count_after / parameter_count_before` |
| 显存减负率 | `1 - pruned_peak_gpu_memory / baseline_peak_gpu_memory` |
| 运行时长减负率 | `1 - pruned_duration / baseline_duration` |
| 文件体积减负率 | `1 - artifact_size_after / artifact_size_before` |

跨不同模型族时，不能直接用 raw score 声称方法优劣，只能报告同模型族内的保留率和资源变化。

---

## 8. 结果目录规范

所有实验结果放在：

```text
results/raw/<run_id>/
```

推荐 run ID：

```text
<method>_<model_short>_<benchmark>_<task>_<yyyymmdd_hhmmss>
```

示例：

```text
flab_qwen25c3b_humaneval_prune_heavy_20260718_204006
slicegpt_codellama7b_humaneval_smoke_YYYYMMDD_HHMMSS
llmpruner_codellama7b_mbpp_smoke_YYYYMMDD_HHMMSS
wanda_codellama7b_humaneval_smoke_YYYYMMDD_HHMMSS
sparsegpt_codellama7b_mbpp_smoke_YYYYMMDD_HHMMSS
```

历史 run ID 不重命名。新实验按新规则命名。

标准文件：

```text
metadata.json
summary.json
summary.md
command.sh
stdout.log
stderr.log
resource.csv
resource_summary.json
generation_summary.json
score_summary.json
score_details.jsonl
artifact_manifest.sha256
```

不是每个 run 都会有全部文件，但用于最终比较的 run 必须尽可能补齐资源和评分字段。

---

## 9. 大文件策略

默认不直接提交大模型权重，除非团队明确决定使用 Git LFS 保存该 artifact。

必须保存：

```text
模型 ID
revision / commit
生成命令
文件名
文件大小
SHA256
是否本地保留
是否可重新生成
```

示例大文件：

```text
*.safetensors
*.bin
*.pt
*.pth
```

如果未提交权重，必须提交 `artifact_manifest.sha256` 或同等 manifest。

---

## 10. 重跑与补指标规则

不是所有旧实验都必须重跑。

判断规则：

1. 已有 `resource.csv` 的实验，先用 `scripts/audit/summarize_resources.py` 生成 `resource_summary.json`；
2. 没有 `resource.csv`，但只用于证明管线跑通的 smoke run，可以不重跑；
3. 没有 `resource.csv`，但要用于显存减负率或运行时长减负率，则必须补跑监控版；
4. 剪枝 artifact 已经有效、可加载、可评测时，不为补字段盲目重跑剪枝；
5. 后续所有重要实验尽量通过 `scripts/run/run_plan.sh` 执行，自动保留资源曲线。

当前补指标计划：

```text
reports/stage1/metric_backfill_plan.md
```

---

## 11. 统一运行方式

进入项目：

```bash
cd ~/projects/srtp-code-llm-pruning
source scripts/setup/env.sh
```

列出任务：

```bash
scripts/run/run_plan.sh --plan configs/experiments/stage1_manual_plan.yaml --list
```

运行单个任务：

```bash
scripts/run/run_plan.sh \
  --plan configs/experiments/stage1_manual_plan.yaml \
  --only <job_id> \
  --include-disabled
```

所有长任务由成员在本地终端运行。Codex 负责给命令、检查输出、补文档和提交结果。

---

## 12. 当前已完成事实

截至 2026-07-22：

| 项目 | 状态 |
|---|---|
| 环境检查 | 完成，CUDA 可见 |
| HumanEval / MBPP smoke split | 完成 |
| LiveCodeBench / SWE-bench Lite smoke split | 完成 |
| SliceGPT R1 OPT-125M smoke | 完成 |
| LLM-Pruner R1 tiny LLaMA smoke | 完成 |
| LaCo R0/R1 blocker | 完成，后续跳过 |
| Flab-Pruner Qwen3B 剪枝 | 完成 |
| Flab-Pruner pruned Qwen3B load/generate | 完成 |
| Qwen1.5B / Qwen3B baseline load/generate | 完成 |
| HumanEval smoke 三组对照 | 完成 |
| MBPP smoke 三组对照 | 完成 |
| SliceGPT Qwen adapter audit | 完成，确认缺 Qwen2 adapter |
| LLM-Pruner Qwen adapter audit | 完成，确认深度绑定 LLaMA |
| 模型策略调整 | 完成，转为 CodeLlama/Qwen/官方模型分流 |

Flab-Pruner 当前已完成第一阶段代表闭环。它不是正式性能结论，但已经证明 Qwen2.5-Coder 剪枝、保存、加载、生成和 smoke eval 管线可运行。

---

## 13. 下一步路线

### 常珂舒

1. 保留 Flab-Pruner 现有 Qwen 闭环；
2. 如果最终要计算资源减负率，补跑 monitored load/generate 或 eval；
3. 按新策略推进 SliceGPT CodeLlama 路线；
4. 按新策略推进 LLM-Pruner CodeLlama 路线；
5. LaCo 继续跳过，除非单独分配 notebook-to-script 工作。

### 潘阔

1. 确认 Wanda/Magnitude 是否能使用 CodeLlama；
2. DSnoT/OWL 优先尝试 LLaMA-compatible/CodeLlama 路线；
3. 如果方法只支持其他模型，记录模型选择原因；
4. 结果必须保留 guide split、eval split、资源和指标字段。

### 李长骏

1. 确认 SparseGPT 的 CodeLlama/LLaMA 路线；
2. FLAP 优先使用 CodeLlama；
3. MaskLLM/Pruner-Zero 能用 CodeLlama 则用，否则保留官方模型；
4. 重点补全资源记录和失败证据。

---

## 14. 验收标准

第一阶段结束时，仓库至少应具备：

1. 每个方法的 R0 审计记录；
2. 尽可能多的方法完成 R1 官方最小复现；
3. 至少一个方法完成代码模型剪枝闭环，当前为 Flab-Pruner/Qwen2.5-Coder；
4. LLaMA-compatible 方法明确 CodeLlama 路线或失败证据；
5. HumanEval/MBPP smoke 评测管线可运行；
6. LiveCodeBench/SWE-bench Lite 数据接口存在；
7. 所有关键 run 有 `summary.md`、日志、资源记录或补指标计划；
8. 大文件不误提交，必要 hash/manifest 已入库；
9. 三名成员的结果可在统一 Git 仓库追溯。

---

## 15. 禁止事项

- 不因分数低删除实验记录；
- 不跨模型族直接比较 raw score 并宣称方法优劣；
- 不把 smoke 结果写成正式性能结论；
- 不强迫所有方法适配 Qwen；
- 不提交 Hugging Face cache；
- 不提交大权重文件，除非团队明确决定使用 Git LFS；
- 不修改其他成员方法结果而不说明原因；
- 不覆盖历史 run ID。

---

## 16. 关键文件索引

```text
README.md
docs/stage1_model_selection_and_metrics.md
docs/execution/stage1_execution.md
docs/methods/qwen_r2_adaptation.md
docs/stage1_qwen3b_execution_route.md
reports/stage1/metric_backfill_plan.md
results/tables/method_status.csv
data/manifests/stage1_model_policy.csv
configs/experiments/stage1_manual_plan.yaml
scripts/run/run_plan.sh
scripts/run/run_experiment_plan.py
scripts/audit/probe_hf_model.py
scripts/audit/summarize_resources.py
```

本文件是第一阶段当前主执行依据；旧 Qwen-only 路线仅作为历史决策记录保留。

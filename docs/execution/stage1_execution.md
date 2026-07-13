# SRTP 第一阶段执行书（V2.0）

## 面向软件工程 Agent 的代码大模型多范式剪枝研究

**阶段名称：** 第一阶段——多范式剪枝方法复现与统一实验流程搭建  
**执行周期：** 2 周  
**时间管理原则：** 本执行书不规定按天、按周或固定会议安排，三名成员根据各自课程与设备使用情况自行协调并行实验顺序  
**执行成员：** 常珂舒、潘阔、李长骏  
**成员身份：** 学生  
**硬件条件：** 常珂舒 RTX 5060；潘阔 RTX 4060；李长骏 RTX 4060  
**统一系统：** Windows + WSL2，建议统一使用 Ubuntu 22.04 LTS  
**首选实验模型：** Qwen2.5-Coder-1.5B / 3B，优先使用 Instruct 版本  
**文档版本日期：** 2026-07-11

---

## 一、执行书定位

本执行书对应主计划书中的第一阶段“基础流程与多方法复现”。第一阶段的核心目标不是立即判断哪一种剪枝方法最好，也不要求形成论文级性能结论，而是尽可能完整地复现主计划书中列出的经典与主流剪枝方法，并建立后续阶段能够直接复用的统一工程体系。

第一阶段主要回答以下执行性问题：

1. 各剪枝方法的官方代码、依赖、模型接口和基本流程能否被成功复现；
2. 各方法能否迁移或适配到 Qwen2.5-Coder 小规模模型；
3. HumanEval、MBPP、LiveCodeBench 和 SWE-bench Lite 能否形成统一的“引导半集—验证半集”流程；
4. 三名成员产生的代码、配置、实验材料、日志和结果能否在统一 Git 仓库中完整保存和追溯；
5. 一名成员完成的方法能否被另一名成员依据仓库文档重新运行。

因此，本阶段的验收重点依次为：

> **方法覆盖完整 → 官方流程可运行 → Qwen 最小适配成功 → 四类 Benchmark 流程接通 → 实验材料完整入库 → 他人可以复跑。**

本阶段不以模型分数高低作为主要验收依据。只要方法按照原研究设定正确运行、输出结构正确、实验材料完整，即使性能下降明显，也属于有效复现结果，不得因结果“不理想”而删除实验记录。

---

## 二、第一阶段方法范围

### 2.1 必须纳入的经典基线与主流方法

第一阶段原则上覆盖主计划书中已经具有代表性、公开实现相对明确的经典与主流方法。

| 编号 | 方法 | 剪枝形态 | 第一阶段定位 |
|---|---|---|---|
| B0 | Magnitude Pruning | 非结构化最低基线 | 作为所有非结构化方法的最低比较基线 |
| M1 | Wanda | 激活感知非结构化 / N:M | 经典强基线 |
| M2 | SparseGPT | 二阶重构非结构化 / N:M | 经典强基线 |
| M3 | LLM-Pruner | 梯度驱动结构化剪枝 | 经典结构化方法 |
| M4 | DSnoT | 稀疏掩码优化 | 与 Wanda 等基础剪枝器组合复现 |
| M5 | MaskLLM | 半结构化剪枝 | 主流 N:M / 掩码学习路线 |
| M6 | FLAP | 训练自由结构化宽度剪枝 | 主流结构化方法 |
| M7 | SliceGPT | 隐藏维度压缩 | 主流结构化维度剪枝 |
| M8 | LaCo | 深度压缩 / 层合并 | 主流深度剪枝方法 |
| M9 | OWL | 非均匀层间稀疏率分配 | 与 Wanda 或 SparseGPT 组合复现 |
| M10 | Pruner-Zero | 自动搜索剪枝重要性指标 | 主流自动化剪枝方法 |
| M11 | Flab-Pruner | 面向代码大模型的剪枝方法 | 软件工程方向重点复现方法 |

Wanda、SparseGPT 是基础剪枝器；DSnoT、OWL、Pruner-Zero 主要改变掩码、层间稀疏率或重要性指标。执行时必须同时保存“基础剪枝器结果”和“组合策略结果”，不能只保存组合后的最好结果。

### 2.2 第一阶段暂不纳入的方法

以下较新、公开实现或模型兼容性仍需进一步确认的方法暂不列为第一阶段硬性任务：

- IFPruning；
- Self-Pruner；
- Agent-guided Pruning；
- 其他尚无稳定公开代码、复现成本明显超过当前设备能力的方法。

这些方法统一登记在 `docs/backlog/frontier_methods.md` 中，记录论文、代码状态、模型兼容性和后续是否纳入，不在第一阶段占用主要实验资源。

### 2.3 方法复现的完成层级

每种方法设置四级完成状态：

| 等级 | 名称 | 判定标准 |
|---|---|---|
| R0 | 资料确认 | 已找到论文、官方或作者认可代码、许可证、模型支持范围和原始命令 |
| R1 | 官方复现 | 在官方支持模型上跑通最小剪枝、保存和评价流程 |
| R2 | Qwen 最小复现 | 在 Qwen2.5-Coder-1.5B 上完成一次最小有效剪枝、保存、重载和生成 |
| R3 | 项目标准复现 | 使用项目统一数据格式，至少完成一组正式引导—验证实验并提交完整结果目录 |

第一阶段的总体目标是：

- 所有纳入方法必须达到 R1；
- 结构和接口可适配的方法尽量达到 R2；
- 每个方法家族至少有一个代表方法达到 R3；
- 无法达到 R2 或 R3 的方法必须提交可复现的失败证据和兼容性分析，不能仅标记“运行失败”。

---

## 三、三人并行分工

### 3.1 方法主分工

| 成员 | 设备 | 方法组 | 具体方法 |
|---|---|---|---|
| **潘阔** | RTX 4060 | 激活感知、掩码优化与非均匀分配 | Magnitude、Wanda、DSnoT、OWL |
| **李长骏** | RTX 4060 | 二阶重构、半结构化与自动搜索 | SparseGPT、MaskLLM、Pruner-Zero、FLAP |
| **常珂舒** | RTX 5060 | 结构化、维度/深度压缩与代码专用方法 | LLM-Pruner、SliceGPT、LaCo、Flab-Pruner |

以上分工代表“第一责任人”，不表示其他成员不得修改或复跑该方法。某一方法遇到长期阻塞时，负责人应先提交最小失败复现，再由其他成员协助排查。

### 3.2 公共职责

| 成员 | 公共职责 | 必须维护的内容 |
|---|---|---|
| 常珂舒 | 仓库集成与统一接口 | 仓库目录、公共模型接口、方法状态总表、正式版本标签 |
| 潘阔 | Benchmark 数据与划分 | 四个 Benchmark 的下载脚本、半集文件、数据检查和 Prompt 模板 |
| 李长骏 | WSL2 环境与实验审计 | 环境锁文件、机器信息、资源监控、结果完整性检查和交叉复跑清单 |

### 3.3 交叉复跑关系

| 原负责人 | 交叉复跑人 | 最低复跑要求 |
|---|---|---|
| 潘阔 | 李长骏 | 复跑 Wanda 或 Wanda+OWL 的最小配置 |
| 李长骏 | 常珂舒 | 复跑 SparseGPT 或 FLAP 的最小配置 |
| 常珂舒 | 潘阔 | 复跑 LLM-Pruner、SliceGPT 或 LaCo 中任一最小配置 |

交叉复跑只要求使用最小模型、最少样本和单一剪枝比例，不要求重复完整实验矩阵。复跑成功的判据是：命令可执行、输出结构一致、稀疏率或参数变化正确、模型可重新加载。

---

## 四、统一 Git 仓库与数据保存规范

### 4.1 仓库名称与权限

建议建立 GitHub 私有仓库：

```text
srtp-code-llm-pruning
```

三名成员均具有写入权限。正式结果必须进入仓库，禁止只通过聊天软件、网盘链接或个人电脑目录传递最终实验材料。

### 4.2 “所有共享信息和实验数据入库”的具体含义

以下内容必须保存在 Git 仓库中：

- 执行书、方法笔记、问题记录、兼容性分析；
- 所有自行编写和修改的脚本；
- 第三方仓库 commit、patch 和修改说明；
- Conda / pip / uv 环境文件与锁文件；
- Benchmark 下载脚本、任务 ID、半集划分、Prompt 和派生校准文本；
- 每次实验使用的配置文件；
- 完整 stdout、stderr 和资源监控日志；
- 生成结果、预测代码、补丁、Agent 轨迹和测试输出；
- 指标 JSON/CSV、汇总表、图表和阶段报告；
- 成功实验与失败实验的 metadata；
- 剪枝模型的 manifest、文件哈希和加载说明；
- 需要三人共享的剪枝权重或中间文件。

不得出现“本地有结果，但仓库中只有截图或口头描述”的情况。

### 4.3 普通 Git 与 Git LFS

仓库内统一启用 Git LFS。建议纳入 LFS 的文件类型：

```gitattributes
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.jsonl.gz filter=lfs diff=lfs merge=lfs -text
*.tar.gz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
```

保存原则：

1. 文档、代码、配置、CSV、JSON 和普通日志使用普通 Git；
2. 大型生成文件、轨迹压缩包、剪枝权重和必要二进制文件使用 Git LFS；
3. Hugging Face 原始模型缓存不要求重复提交，但必须提交模型 revision、文件清单和 SHA256；
4. 剪枝后模型若作为正式交付物，必须通过 Git LFS 保存，或至少保存可从仓库脚本重新生成的完整 manifest；
5. 任何正式实验数据不得只有一份未纳入版本控制的本地副本。

### 4.4 推荐目录结构

```text
srtp-code-llm-pruning/
├─ README.md
├─ .gitignore
├─ .gitattributes
├─ pyproject.toml
├─ docs/
│  ├─ execution/
│  │  └─ stage1_execution.md
│  ├─ methods/
│  │  ├─ magnitude.md
│  │  ├─ wanda.md
│  │  ├─ sparsegpt.md
│  │  ├─ llm_pruner.md
│  │  ├─ dsnot.md
│  │  ├─ maskllm.md
│  │  ├─ flap.md
│  │  ├─ slicegpt.md
│  │  ├─ laco.md
│  │  ├─ owl.md
│  │  ├─ pruner_zero.md
│  │  └─ flab_pruner.md
│  ├─ backlog/
│  │  └─ frontier_methods.md
│  └─ troubleshooting/
├─ env/
│  ├─ common/
│  ├─ wanda/
│  ├─ sparsegpt/
│  ├─ llm_pruner/
│  ├─ dsnot/
│  ├─ maskllm/
│  ├─ flap/
│  ├─ slicegpt/
│  ├─ laco/
│  ├─ owl/
│  ├─ pruner_zero/
│  └─ flab_pruner/
├─ third_party/
│  └─ <method_name>/
├─ patches/
│  └─ <method_name>/
├─ configs/
│  ├─ models/
│  ├─ benchmarks/
│  ├─ pruning/
│  └─ evaluation/
├─ data/
│  ├─ manifests/
│  ├─ splits/
│  │  ├─ humaneval/
│  │  ├─ mbpp/
│  │  ├─ livecodebench/
│  │  └─ swebench_lite/
│  ├─ calibration/
│  ├─ prompts/
│  ├─ fixtures/
│  └─ README.md
├─ src/
│  ├─ common/
│  ├─ adapters/
│  ├─ pruning/
│  ├─ benchmarks/
│  ├─ evaluation/
│  └─ audit/
├─ scripts/
│  ├─ setup/
│  ├─ download/
│  ├─ baseline/
│  ├─ prune/
│  ├─ evaluate/
│  ├─ smoke/
│  └─ audit/
├─ tests/
│  ├─ unit/
│  ├─ smoke/
│  └─ integration/
├─ results/
│  ├─ baseline/
│  ├─ raw/
│  ├─ metrics/
│  ├─ tables/
│  └─ figures/
├─ artifacts/
│  ├─ models/
│  ├─ patches/
│  ├─ trajectories/
│  └─ manifests/
└─ reports/
   ├─ methods/
   ├─ failures/
   ├─ cross_reproduction/
   └─ stage1/
```

删除原 V1 中的 `docs/meetings/`、`reports/weekly/` 和每周报告要求。本阶段不设置固定会议制度。

### 4.5 分支规范

```text
main                         仅保存通过基本检查的稳定版本
develop                      公共集成分支
feat/pan-activation-family   潘阔方法组
feat/li-reconstruction       李长骏方法组
feat/chang-structured        常珂舒方法组
feat/benchmark-pipeline      四个Benchmark公共流程
fix/<topic>                  公共修复
exp/<method>-<topic>         临时实验
```

提交格式：

```text
<type>(<scope>): <description>
```

示例：

```text
feat(wanda): add qwen2 linear layer adapter
feat(lcb): add time-ordered guide and eval split
fix(swebench): preserve repository commit in metadata
exp(flap): test qwen2.5-coder-1.5b minimal structured pruning
report(sparsegpt): add official reproduction record
```

### 4.6 Pull Request 最低内容

每个 PR 必须写明：

- 修改目的；
- 对应方法与复现等级 R0—R3；
- 上游仓库及 commit；
- WSL2 环境名称；
- 可直接复制的运行命令；
- 使用的模型 revision；
- 使用的 Benchmark 和 split hash；
- 输出目录；
- 是否通过 smoke test；
- 已知问题；
- 关联 Issue。

---

## 五、Windows + WSL2 统一环境

### 5.1 WSL2 基础要求

三台电脑统一按以下原则配置：

1. Windows 开启 WSL2；
2. 在 WSL2 中安装 Ubuntu 22.04 LTS；
3. 项目仓库放在 WSL2 Linux 文件系统内部，例如：

```bash
~/projects/srtp-code-llm-pruning
```

4. 不建议把仓库、模型缓存和大量小文件放在 `/mnt/c/` 下，以避免跨文件系统 I/O 变慢；
5. Windows 侧安装能够支持 WSL2 GPU 的 NVIDIA 驱动；
6. WSL2 内首先通过 `nvidia-smi` 和 PyTorch 检查 GPU；
7. Docker 相关 Benchmark 建议统一使用 Docker Desktop 的 WSL2 后端，或在 WSL2 内使用兼容的 Docker Engine；
8. 不在宿主环境直接执行来源不明的 Benchmark 生成代码，HumanEval、MBPP、LiveCodeBench 和 SWE-bench Lite 的代码执行均应使用隔离环境。

### 5.2 机器信息记录

每台电脑执行：

```bash
nvidia-smi
python --version
git --version
git lfs version
docker --version
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print('vram_gb:', torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else None)
PY
```

保存到：

```text
reports/stage1/machines/<name>_machine_info.md
```

### 5.3 环境隔离

禁止强制所有方法共用单一 Python 环境。建议：

```text
srtp-common
srtp-wanda
srtp-sparsegpt
srtp-llmpruner
srtp-dsnot
srtp-maskllm
srtp-flap
srtp-slicegpt
srtp-laco
srtp-owl
srtp-prunerzero
srtp-flabpruner
```

如果两个方法依赖兼容，可以共用环境，但必须在对应方法文档中明确记录。每个成功环境至少提交：

- `environment.yml` 或 `requirements.txt`；
- `pip_freeze.txt`；
- Python、CUDA、PyTorch、Transformers 版本；
- 安装命令；
- 已知冲突。

### 5.4 显存不足时的统一处理顺序

发生 OOM 时按以下顺序处理并记录：

1. 使用 Qwen2.5-Coder-1.5B；
2. batch size 降为 1；
3. 校准样本减少；
4. 序列长度从 2048 降至 1024，再降至 512；
5. 逐层剪枝；
6. CPU offload；
7. 关闭 KV cache；
8. 释放当前层缓存；
9. 将 smoke test 与正式复现分开；
10. 若仍无法运行，保留官方模型 R1 结果并提交 Qwen 失败报告。

不得未经记录直接改变算法关键参数，以免把“方法复现”变成不明来源的自定义变体。

---

## 六、统一模型规范

### 6.1 模型层级

| 层级 | 模型 | 用途 |
|---|---|---|
| O | 方法官方支持的小模型 | 完成官方 R1 复现 |
| Q1 | Qwen2.5-Coder-1.5B / Instruct | 所有方法的主要适配与最小复现模型 |
| Q3 | Qwen2.5-Coder-3B / Instruct | 资源允许时复现代表方法，验证流程可扩展性 |
| C | 原始 Qwen2.5-Coder-1.5B / 3B | 剪枝前基线 |

本阶段以 Q1 为“方法覆盖优先模型”。不能为了追求 3B 而放弃其他方法复现。Q3 主要用于 Wanda、SparseGPT、LLM-Pruner、FLAP、SliceGPT 等代表方法的扩展验证，不要求所有方法都在 3B 上运行。

### 6.2 Base 与 Instruct

- 代码生成和 SWE-bench Lite 验证优先使用 Instruct；
- 某官方方法只支持 Base 时，可以先在 Base 上达到 R1 或 R2；
- Base 和 Instruct 的结果必须分开保存，不得直接合并比较；
- 模型名称、revision、dtype 和 tokenizer revision 必须写入 metadata。

### 6.3 模型清单

每个模型生成：

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "revision": "<commit>",
  "tokenizer_revision": "<commit>",
  "dtype": "float16",
  "local_cache": "not_committed",
  "files": [
    {"name": "...safetensors", "size_bytes": 0, "sha256": "..."}
  ]
}
```

清单提交至：

```text
data/manifests/models/<model_short_name>.json
```

---

## 七、四个 Benchmark 的统一引导—验证设计

### 7.1 总体要求

第一阶段正式纳入：

- HumanEval；
- MBPP；
- LiveCodeBench；
- SWE-bench Lite。

四个 Benchmark 均需形成独立的引导集和验证集，并进入统一数据清单。第一阶段的“全面纳入”指：

1. 四个 Benchmark 均完成数据获取和版本固定；
2. 四个 Benchmark 均形成不可重叠的 guide/eval 划分；
3. 四个 Benchmark 均能生成剪枝所需的校准文本或任务输入；
4. 四个 Benchmark 均具有可运行的验证入口；
5. 正式方法至少使用对应 guide 集完成剪枝，并在对应 eval 集上验证；
6. 第一阶段不强制执行四个引导集与四个验证集的 4×4 全交叉矩阵。

为控制复现规模，第一阶段采用“同源引导—同源验证”为最低要求。例如 HumanEval-guide 引导的模型至少在 HumanEval-eval 上验证；跨 Benchmark 泛化比较留到后续阶段。

### 7.2 Smoke test 范围

Smoke test 只使用 HumanEval 和 MBPP，降低初始运行要求：

```text
HumanEval guide: 2—4题
HumanEval eval: 2—4题
MBPP guide: 2—4题
MBPP eval: 2—4题
```

Smoke test 只检查：数据加载、剪枝、保存、重载、生成和评价链路是否通畅，不将其结果纳入正式表格，也不要求 LiveCodeBench 和 SWE-bench Lite 参与每次 PR 的 smoke test。

### 7.3 HumanEval

划分原则：

- 按任务类型和基础难度尽量平衡划分；
- guide 与 eval 各自保存任务 ID；
- 参考答案和隐藏测试不能进入校准文本；
- 评价使用项目冻结的 EvalPlus 版本或等效固定工具。

文件：

```text
data/splits/humaneval/guide.jsonl
data/splits/humaneval/eval.jsonl
data/splits/humaneval/manifest.json
```

### 7.4 MBPP

划分原则：

- 按字符串、数组、数学、标准库、数据结构等任务类型平衡；
- 保留官方 task_id；
- 训练示例、测试样例和答案字段要按评价工具要求隔离；
- 正式校准文本不得包含目标实现。

文件：

```text
data/splits/mbpp/guide.jsonl
data/splits/mbpp/eval.jsonl
data/splits/mbpp/manifest.json
```

### 7.5 LiveCodeBench

LiveCodeBench 优先采用时间顺序划分：

- 较早任务作为 guide；
- 较新任务作为 eval；
- 保留 release/version、发布日期、任务类别和官方评测字段；
- 不进行破坏时间顺序的随机反向划分；
- 第一阶段至少跑通代码生成主任务；其他子任务按官方接口可用性逐步接入。

文件：

```text
data/splits/livecodebench/guide.jsonl
data/splits/livecodebench/eval.jsonl
data/splits/livecodebench/manifest.json
```

### 7.6 SWE-bench Lite

SWE-bench Lite 按仓库、问题类型和任务复杂度进行平衡划分：

- guide 与 eval 不共享同一个具体 issue；
- 尽量降低同一仓库高度相似任务跨集合造成的信息泄漏；
- guide 可使用 issue 文本、仓库元数据、目录树和公开代码上下文；
- 禁止将 gold patch、隐藏测试修复或验证集目标补丁放入校准文本；
- eval 使用固定仓库 commit 和官方测试环境；
- 第一阶段使用 mini-SWE-agent 或统一最小代理执行器作为 SWE-bench Lite 的标准运行入口，但不在本阶段深入分析 Agent 工作流机制；
- 保存每次运行的 trajectory、tool call、patch、test log 和最终状态。

文件：

```text
data/splits/swebench_lite/guide.jsonl
data/splits/swebench_lite/eval.jsonl
data/splits/swebench_lite/manifest.json
artifacts/trajectories/<run_id>/
artifacts/patches/<run_id>/
```

### 7.7 数据泄漏检查

必须编写自动检查：

- guide 与 eval 的 task_id 不重叠；
- LiveCodeBench guide 日期早于 eval；
- SWE-bench Lite issue ID 不重叠；
- 校准文本不含参考答案、gold patch 或隐藏测试；
- split 文件生成后计算 SHA256；
- 正式实验 metadata 必须记录 split hash；
- 任何 split 修改必须产生新版本号，不能覆盖旧文件。

---

## 八、统一校准文本与评价入口

### 8.1 校准文本统一结构

不同 Benchmark 转换为统一记录：

```json
{
  "benchmark": "humaneval|mbpp|livecodebench|swebench_lite",
  "task_id": "...",
  "split_role": "guide",
  "prompt": "...",
  "context": "...",
  "metadata": {},
  "contains_solution": false
}
```

推荐文本模板：

```text
### Benchmark
{benchmark_name}

### Task
{problem_or_issue_text}

### Available Context
{signature_examples_repository_context}

### Instruction
Complete the requested code task. Do not use unavailable tests or reference solutions.
```

### 8.2 各方法的引导信号映射

| 方法类型 | guide 集用途 |
|---|---|
| Wanda / OWL | 采集输入激活范数、层异常值和层间敏感度 |
| SparseGPT | 构建逐层输入统计和近似二阶重构信息 |
| LLM-Pruner | 计算梯度或结构重要性 |
| DSnoT | 在基础稀疏掩码上进行动态搜索或校正 |
| MaskLLM | 构造半结构化掩码学习或搜索输入 |
| FLAP | 统计激活波动和结构重要性 |
| SliceGPT | 获取代表性隐藏状态并执行维度变换/切片 |
| LaCo | 估计相邻层或候选层合并影响 |
| Pruner-Zero | 评价候选重要性指标的校准表现 |
| Flab-Pruner | 提取代码任务相关信号并执行其官方流程 |

### 8.3 统一验证入口

```text
scripts/evaluate/run_humaneval.sh
scripts/evaluate/run_mbpp.sh
scripts/evaluate/run_livecodebench.sh
scripts/evaluate/run_swebench_lite.sh
```

统一输出至少包括：

```text
predictions.jsonl
metrics.json
stdout.log
stderr.log
resource.csv
summary.md
```

HumanEval、MBPP、LiveCodeBench 和 SWE-bench Lite 的原始官方指标分别保存，不强行转换为同一种分数。

---

## 九、所有方法共同执行流程

每种方法均按以下顺序执行，不得只提交最后一条命令。

### 9.1 R0：资料与代码固定

1. 记录论文题目、年份和方法核心；
2. 确认官方或作者认可代码；
3. 固定上游仓库 URL 与 commit；
4. 记录许可证；
5. 记录官方支持模型、数据集和硬件；
6. 抄录并验证 README 中最小命令；
7. 建立 `docs/methods/<method>.md`。

### 9.2 R1：官方模型复现

1. 单独创建方法环境；
2. 在官方支持的小模型上运行 dense baseline；
3. 使用官方默认或最小剪枝配置；
4. 检查实际稀疏率、参数量或结构变化；
5. 保存模型；
6. 关闭进程后重新加载；
7. 运行官方评价或最小生成；
8. 提交完整命令、日志、环境和结果；
9. 说明与论文/README 数值是否一致；
10. 数值不一致时先说明原因，不要求人为调整到完全相同。

### 9.3 R2：Qwen 最小复现

1. 编写或复用 `src/adapters/qwen2_adapter.py`；
2. 枚举 Qwen attention 和 MLP 层；
3. 验证 forward 参数、position ids、attention mask 和 cache；
4. 使用 Qwen2.5-Coder-1.5B；
5. 使用 HumanEval/MBPP smoke 数据；
6. 采用最低有效剪枝比例；
7. 保存并重新加载；
8. 生成至少一个可执行代码结果；
9. 计算实际稀疏率或参数量；
10. 把所有 Qwen 特殊修改保存为 adapter 或 patch。

### 9.4 R3：项目标准复现

1. 选用冻结的 Qwen 模型 revision；
2. 选用固定 guide split；
3. 按原方法合理配置运行剪枝；
4. 在对应 eval split 上验证；
5. 保存完整 metadata；
6. 保存模型或模型 manifest；
7. 保存所有预测、补丁和轨迹；
8. 生成方法复现报告；
9. 由非负责人执行最小交叉复跑；
10. 更新 `results/tables/method_status.csv`。

---

## 十、潘阔方法组操作要求

### 10.1 Magnitude Pruning

用途：建立最低非结构化基线。

最低复现：

- 全局 magnitude 与逐层 magnitude 至少一种；
- 30% 和 50% 非结构化稀疏率至少各一组；
- 如基础代码支持，增加 2:4；
- 验证稀疏率和重载；
- 使用与 Wanda 相同模型和 guide 集。

### 10.2 Wanda

关键检查：

- 正确定位 Qwen 的 `q_proj/k_proj/v_proj/o_proj`；
- 正确定位 `gate_proj/up_proj/down_proj`；
- 激活 hook 触发次数正确；
- 激活统计无 NaN/Inf；
- 30%、50% 和 2:4 至少完成最小复现；
- 每层实际稀疏率与目标一致；
- 保存激活统计摘要和每层稀疏率。

### 10.3 DSnoT

执行原则：

1. 先生成固定基础掩码；
2. 分别以 Wanda 和 Magnitude 中至少一种为起点；
3. 保存优化前掩码、优化后掩码和变化比例；
4. 记录迭代次数、搜索参数和停止条件；
5. 不得只报告 DSnoT 后结果；
6. 复现失败时区分“基础掩码失败”和“掩码优化失败”。

### 10.4 OWL

执行原则：

1. 先复现统一稀疏率基础方法；
2. 采集层异常值或层敏感度；
3. 生成非均匀层间稀疏率配置；
4. 检查总稀疏率是否仍满足目标；
5. 保存每层分配表；
6. 至少与 Wanda 组合；
7. 如资源允许，再与 SparseGPT 组合，由李长骏协助。

---

## 十一、李长骏方法组操作要求

### 11.1 SparseGPT

关键检查：

- 第一层输入捕获；
- attention mask 和 position ids；
- 当前层逐层搬入 GPU；
- 二阶统计是否出现 NaN/Inf；
- blocksize、percdamp 和校准样本数；
- 逐层重构误差；
- 30%、50% 和 2:4 至少完成最小复现；
- 保存逐层时间、CPU 内存和 GPU 峰值。

### 11.2 MaskLLM

关键检查：

- 官方支持的 N:M pattern；
- 掩码参数与原始权重的保存关系；
- 是否需要训练、校准或额外 checkpoint；
- Qwen Linear 层是否完整纳入；
- 2:4 pattern 检查；
- 重载后掩码是否仍生效；
- 若官方模型结构无法直接迁移，至少完成官方 R1 并提交 Qwen 适配问题表。

### 11.3 Pruner-Zero

关键检查：

- 搜索空间定义；
- 候选重要性指标；
- 评价代理指标；
- 搜索随机种子；
- 搜索预算；
- 最终选中的指标表达式；
- 搜索过程日志；
- 选中指标应用到基础剪枝器后的完整结果。

第一阶段不要求大规模搜索。可采用官方最小搜索预算完成流程复现，但必须明确标记为“流程复现”，不能与论文完整搜索结果等同。

### 11.4 FLAP

关键检查：

- 激活波动统计；
- 注意力头或 FFN 通道的重要性；
- 补偿参数或 bias 处理；
- 结构化剪枝前后参数量；
- 模型 config 与权重 shape；
- 保存、标准加载和 generate；
- 结构化后实际显存和推理速度记录。

---

## 十二、常珂舒方法组操作要求

### 12.1 LLM-Pruner

关键检查：

- 结构发现、重要性估计和剪枝三个阶段；
- Qwen GQA 中 Q heads 与 KV heads 的分组关系；
- `o_proj` 输入维度同步；
- `gate_proj/up_proj/down_proj` 通道同步；
- RMSNorm、残差和输出层兼容；
- 剪枝前后真实参数量；
- config 更新；
- 标准加载和生成。

### 12.2 SliceGPT

关键检查：

- 正交变换或旋转步骤；
- hidden dimension 切片比例；
- embedding、attention、MLP 和 lm_head 的维度一致性；
- 旋转后数值稳定性；
- 模型保存和标准加载；
- 实际参数量、模型文件和显存变化。

### 12.3 LaCo

关键检查：

- 候选合并层；
- 层相似度或合并准则；
- 层合并次序；
- 合并后层数和 config；
- 残差路径和 cache；
- 保存与重载；
- 不同层合并数量至少完成一组最小配置。

### 12.4 Flab-Pruner

关键检查：

- 官方代码和论文设定是否针对特定代码模型；
- 代码任务信号构造；
- 方法使用的剪枝粒度；
- Qwen 适配点；
- 与通用方法不同的配置；
- 至少在 HumanEval/MBPP 和 LiveCodeBench 中接通代码任务流程；
- 如 SWE-bench Lite 无法直接用作其引导信号，应提交原因和替代的静态仓库上下文方案。

---

## 十三、方法覆盖矩阵

### 13.1 最低复现矩阵

| 方法 | R1 官方复现 | Qwen 1.5B 最小复现 | HE/MBPP smoke | 四 Benchmark 数据接口 | 正式结果材料入库 |
|---|---|---|---|---|---|
| Magnitude | 必须 | 必须 | 必须 | 必须 | 必须 |
| Wanda | 必须 | 必须 | 必须 | 必须 | 必须 |
| SparseGPT | 必须 | 必须 | 必须 | 必须 | 必须 |
| LLM-Pruner | 必须 | 必须或失败证据 | 必须 | 必须 | 必须 |
| DSnoT | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| MaskLLM | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| FLAP | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| SliceGPT | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| LaCo | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| OWL | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| Pruner-Zero | 必须 | 尽量完成 | 必须 | 必须 | 必须 |
| Flab-Pruner | 必须 | 尽量完成 | 必须 | 必须 | 必须 |

这里的“四 Benchmark 数据接口”是指该方法可以读取项目统一格式的四类 guide 数据。对于官方算法确实无法使用某类输入的情况，应在方法文档中说明并记录转换方式或限制。

### 13.2 正式引导—验证最低要求

为了避免两周内形成不可执行的全组合矩阵，第一阶段不强制所有方法完成 4×4 跨 Benchmark 比较。最低要求为：

- 每个方法至少完成一组项目标准 R3 复现；
- 每个方法组负责保证其代表方法覆盖四个 Benchmark；
- 三个方法组共同保证 HumanEval、MBPP、LiveCodeBench、SWE-bench Lite 均有剪枝模型和验证结果；
- 对同一方法进行四种 guide 复现时，使用相同模型、相同剪枝比例和相同方法参数；
- 跨 Benchmark 结论暂不在第一阶段做统计推断。

推荐代表方法：

| 方法组 | 四 Benchmark 代表方法 |
|---|---|
| 潘阔组 | Wanda |
| 李长骏组 | SparseGPT 或 FLAP |
| 常珂舒组 | LLM-Pruner 或 Flab-Pruner |

---

## 十四、统一 Smoke Test

### 14.1 Smoke Test 流程

每个方法在合并前完成：

```text
1. 加载官方极小模型或Qwen 1.5B
2. 加载2—4条HumanEval guide
3. 加载2—4条MBPP guide
4. 执行最低有效剪枝
5. 统计实际稀疏率或参数量
6. 保存临时模型
7. 关闭并重新加载
8. HumanEval生成1—2题
9. MBPP生成1—2题
10. 输出metadata、日志和结果
```

### 14.2 Smoke Test 不要求

- 不要求 LiveCodeBench；
- 不要求 SWE-bench Lite；
- 不要求完整 Pass@1；
- 不要求长序列；
- 不要求 3B；
- 不要求多随机种子；
- 不进入正式结论表。

---

## 十五、正式实验记录规范

### 15.1 Run ID

```text
<model>_<method>_<guide>_<ratio>_<pattern>_s<seed>_r<rep>
```

示例：

```text
qwen25c15b_wanda_he_s050_unstruct_s42_r01
qwen25c15b_sparsegpt_lcb_s050_unstruct_s42_r01
qwen25c15b_flap_swebl_p020_struct_s42_r01
qwen25c15b_laco_mbpp_l04_merge_s42_r01
```

Benchmark 简写：

```text
he      HumanEval
mbpp    MBPP
lcb     LiveCodeBench
swebl   SWE-bench Lite
```

### 15.2 每次实验目录

```text
results/raw/<run_id>/
├─ config.yaml
├─ metadata.json
├─ command.sh
├─ stdout.log
├─ stderr.log
├─ resource.csv
├─ structure_before.json
├─ structure_after.json
├─ sparsity_by_layer.csv
├─ predictions.jsonl
├─ metrics.json
├─ summary.md
└─ artifact_manifest.json
```

SWE-bench Lite 额外保存：

```text
artifacts/trajectories/<run_id>/
artifacts/patches/<run_id>/
results/raw/<run_id>/test_logs/
```

### 15.3 metadata 最低字段

```json
{
  "run_id": "...",
  "owner": "...",
  "status": "success|failed|partial",
  "reproduction_level": "R0|R1|R2|R3",
  "git_commit": "...",
  "upstream_repo": "...",
  "upstream_commit": "...",
  "machine": "...",
  "gpu": "...",
  "wsl_version": "...",
  "ubuntu": "...",
  "cuda": "...",
  "torch": "...",
  "transformers": "...",
  "model_id": "...",
  "model_revision": "...",
  "benchmark": "...",
  "guide_split_sha256": "...",
  "eval_split_sha256": "...",
  "method": "...",
  "target_ratio": 0.0,
  "actual_ratio": 0.0,
  "seed": 42,
  "start_time": "...",
  "end_time": "..."
}
```

### 15.4 失败实验

失败也必须提交：

- 最小命令；
- 完整报错；
- 环境；
- 输入数据；
- 发生阶段；
- 已尝试修复；
- 是否可重复；
- 初步原因；
- 后续建议。

统一分类：

```text
ENV      环境与依赖
DATA     数据加载、划分或泄漏
MODEL    模型结构或权重
MEM      GPU/CPU内存
PRUNE    剪枝算法
SAVE     保存或重载
EVAL     生成、测试或评价
DOCKER   容器和SWE-bench环境
REPRO    他人无法复跑
UNKNOWN  尚未定位
```

---

## 十六、统一评价与资源记录

### 16.1 能力结果

| Benchmark | 主要保存内容 |
|---|---|
| HumanEval | Pass@1、任务级通过状态、生成代码、测试日志 |
| MBPP | Pass@1/Accuracy、任务级通过状态、生成代码、测试日志 |
| LiveCodeBench | 官方代码生成指标、任务类别结果、运行日志 |
| SWE-bench Lite | Resolved Rate、patch、测试结果、trajectory、最终状态 |

### 16.2 剪枝结果

必须记录：

- 目标剪枝率；
- 实际稀疏率；
- 实际参数量；
- 非零参数量；
- 每层稀疏率或结构变化；
- 模型文件大小；
- 是否使用稀疏 kernel；
- 模型能否标准加载。

### 16.3 资源结果

必须记录：

- 剪枝峰值显存；
- 剪枝时 CPU 内存；
- 剪枝总耗时；
- 模型加载时间；
- 推理峰值显存；
- 首 token 延迟；
- tokens/s；
- SWE-bench Lite 端到端耗时；
- 生成 token 数和工具调用次数。

本阶段只要求如实记录，不要求根据这些指标立即得出方法优劣结论。

---

## 十七、测试与审计

### 17.1 单元测试

至少包括：

- Qwen decoder 层定位；
- Linear 层枚举；
- GQA 头数与维度检查；
- guide/eval 互斥；
- split hash 稳定；
- 校准文本不含答案；
- 稀疏率统计；
- N:M pattern；
- 结构化参数量统计；
- metadata 字段完整；
- Git LFS 跟踪检查。

### 17.2 正式结果入表前检查

- Git commit 存在；
- 上游 commit 存在；
- 环境文件存在；
- 模型 revision 存在；
- split hash 正确；
- 输出任务数正确；
- metrics 可重算；
- 模型可重载；
- 失败样本能定位 task_id；
- 正式数据已推送到仓库；
- 大文件已被 Git LFS 正确跟踪；
- 没有把 eval 数据用于剪枝或调参。

### 17.3 方法状态总表

```text
results/tables/method_status.csv
```

字段：

```text
method
owner
upstream_repo
upstream_commit
r0_status
r1_status
r2_status
r3_status
qwen_compatible
humaneval_ready
mbpp_ready
livecodebench_ready
swebench_lite_ready
artifact_complete
cross_reproduced
blocking_issue
notes
```

---

## 十八、阶段验收标准

### 18.1 必达标准

第一阶段完成需满足：

1. 统一 Git 私有仓库已实际使用；
2. 三台 Windows 电脑均通过 WSL2 GPU 检查；
3. 所有共享脚本、文档、配置、实验数据和结果均进入 Git/Git LFS；
4. HumanEval、MBPP、LiveCodeBench、SWE-bench Lite 均完成版本固定和 guide/eval 划分；
5. HumanEval 和 MBPP 的 smoke test 可重复运行；
6. Magnitude、Wanda、SparseGPT、LLM-Pruner、DSnoT、MaskLLM、FLAP、SliceGPT、LaCo、OWL、Pruner-Zero、Flab-Pruner 均达到 R1，或提交明确的上游代码缺失/不可运行证据；
7. Wanda、SparseGPT 和至少一种结构化方法达到 Qwen R2；
8. 三个方法组各至少有一个代表方法达到 R3；
9. 四个 Benchmark 均至少产生一组剪枝模型验证结果；
10. 每种方法均有方法文档、命令、环境、日志和状态记录；
11. 至少完成三次最小交叉复跑；
12. 成功与失败实验均可追溯，未删除不利结果。

### 18.2 不作为本阶段必达标准

- 不要求对所有方法进行完整 4×4 Benchmark 交叉比较；
- 不要求每个方法都完成 Qwen 3B；
- 不要求形成“最佳剪枝方法”结论；
- 不要求统计显著性分析；
- 不要求完整 Agent 工作流能力归因；
- 不要求大规模 LoRA 恢复；
- 不要求纳入 IFPruning、Self-Pruner 和 Agent-guided Pruning。

---

## 十九、第一阶段最终交付物

1. 第一阶段执行书最终版；
2. 统一 Git 私有仓库；
3. WSL2 环境和机器信息；
4. 十二种方法的资料与复现状态文档；
5. 各方法固定的上游 commit 和 patch；
6. Qwen 公共模型适配器；
7. HumanEval、MBPP、LiveCodeBench、SWE-bench Lite 数据清单；
8. 四个 Benchmark 的 guide/eval 划分与 hash；
9. HumanEval、MBPP smoke test；
10. 四个 Benchmark 的统一评价脚本；
11. 所有方法的官方复现日志；
12. Qwen 最小复现结果与失败证据；
13. 正式实验结果目录；
14. SWE-bench Lite patch、测试日志和轨迹；
15. 方法状态总表；
16. 三次交叉复跑记录；
17. 兼容性与问题清单；
18. 第一阶段复现总结。

---

## 二十、第一阶段总结的写作口径

第一阶段总结不直接写“某方法优于某方法”，而应围绕复现事实展开：

- 哪些方法官方代码可以直接运行；
- 哪些方法需要修改依赖；
- 哪些方法可直接识别 Qwen 结构；
- 哪些方法需要 Qwen adapter；
- 哪些方法在 RTX 5060/4060 上可以完成；
- 哪些方法受显存、CPU 内存或 Docker 环境限制；
- 四个 Benchmark 的引导与验证流程是否接通；
- 哪些方法保存后可以标准重载；
- 哪些方法存在结构化 config 不一致；
- 哪些方法值得进入下一阶段完整对比；
- 哪些方法只能保留为官方复现或兼容性案例。

第一阶段允许形成“复现难度和工程可行性判断”，但不提前形成论文级性能结论。

---

## 二十一、风险与处理

| 风险 | 表现 | 处理方式 |
|---|---|---|
| 两周内方法数量较多 | 个别方法只完成官方复现 | 优先保证 R1 全覆盖，再推进代表方法 R2/R3 |
| Qwen 不受官方支持 | 层名、GQA、forward 不兼容 | 建立公共 Qwen adapter；保存 patch 和失败证据 |
| RTX 4060/5060 显存不足 | OOM 或剪枝过慢 | Qwen 1.5B 优先、逐层处理、CPU offload、缩短校准序列 |
| WSL2 文件系统过慢 | 数据加载和 Git 操作缓慢 | 仓库与缓存放在 Linux 文件系统，不放 `/mnt/c/` |
| Git 仓库过大 | LFS 配额或 clone 过慢 | 压缩轨迹、使用 LFS、保留必要正式权重，原始缓存只存 manifest |
| SWE-bench Lite 环境复杂 | Docker 构建失败、测试不一致 | 固定镜像、repo commit 和 harness 版本；失败日志完整入库 |
| 数据泄漏 | eval 进入校准或调参 | 自动 split 检查、hash 校验、数据角色字段 |
| 只保存成功结果 | 复现结论失真 | 所有 run 均有 metadata；失败结果不得删除 |
| 方法定义混淆 | OWL/DSnoT 被当作独立基础剪枝器 | 同时保存基础剪枝器和组合策略结果 |
| 为追求分数偏离原方法 | 参数被随意修改 | 第一阶段优先遵循官方配置；自定义修改单独标记 variant |

---

## 当前进度

### 已确定

- 第一阶段执行周期调整为 2 周；
- 执行书不划分按天、按周时间表，也不设置固定会议制度；
- 三名成员均使用 Windows + WSL2；
- 第一阶段以复现覆盖为核心，不以方法优劣结论为核心；
- 纳入经典与主流方法：Magnitude、Wanda、SparseGPT、LLM-Pruner、DSnoT、MaskLLM、FLAP、SliceGPT、LaCo、OWL、Pruner-Zero、Flab-Pruner；
- IFPruning、Self-Pruner、Agent-guided Pruning 暂不纳入；
- HumanEval、MBPP、LiveCodeBench、SWE-bench Lite 全部进入正式引导—验证流程；
- Smoke test 只使用 HumanEval 和 MBPP 的极小样本；
- 所有共享资料、脚本、配置、日志、结果和必要实验文件均存入 Git/Git LFS 仓库；
- 以 Qwen2.5-Coder-1.5B 为方法覆盖优先模型，3B 用于代表方法扩展；
- 第一阶段采用 R0—R3 复现等级和交叉复跑进行验收。

### 后续由组员在执行中自行确定

- 各方法的具体运行先后顺序；
- 三人各自在两周内的每日安排；
- GitHub 私有仓库建立在个人账号还是组织账号；
- Git LFS 配额和正式剪枝权重的保留范围；
- Qwen Base 与 Instruct 在个别方法中的具体选择；
- 每个方法进入 Q3（3B）扩展验证的优先级。

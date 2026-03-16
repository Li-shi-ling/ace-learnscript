# ACE-LearnScript

ACE-LearnScript 是一个面向 **Prompt 优化与角色风格对齐** 的实验框架，基于 ACE（Adaptive Curation Engine）工作流，提供从数据预处理、离线/在线训练、评估到结果归档的完整闭环。其设计目标是：

- 以统一配置快速启动实验；
- 兼容 OpenAI 协议的多种模型服务；
- 降低调试与训练成本（速度、token、可观测性）；
- 支持长任务场景下的断点续训与进度继承。

---

## 1. 核心能力

- **一键启动**：通过单一 YAML 文件完成模型、数据、训练、输出配置。  
- **OpenAI 协议兼容**：不仅支持 OpenAI 官方接口，也支持各类兼容网关与私有部署。  
- **可扩展 CSV 数据加载**：在表头语义保持一致时可稳定处理数据；支持角色别名映射。  
- **训练可观测性增强**：提供逐步进度日志、提示词快照、阶段性 playbook 与最终结果归档。  
- **断点续训**：支持中断后从 checkpoint 自动恢复。  
- **成本控制**：可通过样本裁剪、输入截断、可选后置生成等参数降低耗时与 token 开销。

---

## 2. 目录结构

```text
ace-learnscript/
├── ace/                       # ACE 核心训练与优化流程
├── configs/
│   └── train_config.yaml      # 统一训练配置（推荐入口）
├── eval/roleplay/             # 角色扮演任务数据处理与评估逻辑
├── results/                   # 训练输出目录（按 run_name 区分）
├── run_one_click.py           # 一键运行入口
└── README.md
```

---

## 3. 环境准备

建议使用 Python 3.10+。

```bash
uv sync
```

若你未使用 `uv`，也可以使用 `pip` 安装依赖（以项目实际依赖声明为准）。

---

## 4. 快速开始

### 4.1 配置文件

默认配置文件路径：`configs/train_config.yaml`。  
你通常只需修改该文件即可启动实验。

### 4.2 一键运行

```bash
uv run python run_one_click.py --config configs/train_config.yaml
```

---

## 5. 统一配置说明（train_config.yaml）

以下为建议关注的关键配置块。

### 5.1 LLM 配置（最重要）

```yaml
llm:
  api_provider: openai_compatible
  model_name: gpt-4o-mini
  api_base: ${ENV:OPENAI_COMPATIBLE_BASE_URL}
  api_key: ${ENV:OPENAI_COMPATIBLE_API_KEY}
```

说明：

- `api_provider` 推荐使用 `openai_compatible`；
- `api_base` 与 `api_key` 支持 `${ENV:VAR_NAME}` 形式；
- 兼容所有 OpenAI 格式服务端点（官方、代理、网关、私有部署）。

### 5.2 数据配置

```yaml
data:
  source_data: ./Act_02.csv
  focus_role: hiro
  role_aliases_config: ./eval/roleplay/data/role_aliases.json
  train_ratio: 0.8
  val_ratio: 0.1
  split_seed: 42
```

说明：

- `focus_role` 为训练目标角色；
- `role_aliases_config` 用于角色别名规范化映射；
- 数据切分采用可复现随机种子。

### 5.3 训练配置

```yaml
training:
  num_epochs: 1
  eval_steps: 100
  save_steps: 50
  resume: false
  context_max_chars: 1200
  question_max_chars: 600
  enable_post_train_generation: false
```

说明：

- `resume=true` 可开启续训；
- `context_max_chars` / `question_max_chars` 控制输入长度；
- `enable_post_train_generation=false` 可显著减少一次额外生成调用。

### 5.4 输出配置

```yaml
output:
  base_dir: ./results
  run_name: roleplay_fast_demo
  resume_run_path: null
```

说明：

- `run_name` 用于组织实验输出目录；
- 续训时可直接指定 `resume_run_path` 指向历史结果目录。

---

## 6. 关于“5133 个批次”的说明

在角色任务中，`5133` 常见于如下过程：

1. 原始对话数据经角色筛选与样本构建后，得到约 `6417` 条有效样本；
2. 训练集比例为 `train_ratio=0.8`；
3. 训练样本数为 `int(6417 * 0.8) = 5133`。

这代表的是**训练样本条数**（step 数），并非传统深度学习意义上的 mini-batch 数。

---

## 7. 日志与产物说明

单次运行目录中，常见文件包括：

- `training_progress.jsonl`：逐步训练表现记录（正确性、token 等）；
- `checkpoint_state.json`：断点续训状态；
- `prompt_history/`：每步提示词快照（便于回溯优化轨迹）；
- `intermediate_playbooks/`：阶段性 playbook；
- `final_playbook.txt`：最终 playbook；
- `best_playbook.txt`：验证集最优 playbook；
- `train_results.json` / `val_results.json`：训练与验证聚合结果。

---

## 8. 断点续训

### 首次运行

```yaml
training:
  resume: false
output:
  run_name: roleplay_fast_demo
  resume_run_path: null
```

### 中断后恢复

```yaml
training:
  resume: true
output:
  resume_run_path: ./results/roleplay_fast_demo
```

系统会自动加载 `checkpoint_state.json` 并从上次进度继续执行。

---

## 9. 性能与成本优化建议

为平衡效果与资源消耗，建议采用以下策略：

1. 先用 `max_train_samples/max_val_samples/max_test_samples` 做小规模试跑；
2. 控制 `context_max_chars` 与 `question_max_chars`，减少不必要上下文；
3. 调参阶段关闭 `enable_post_train_generation`；
4. 合理增大 `eval_steps`，降低评估频率开销。

---

## 10. 适用场景

- 角色扮演风格提示词优化；
- 多轮任务中的生成策略迭代；
- 需要低门槛复现实验与可审计日志的 Prompt Engineering 工作流。

---

## 11. 许可与声明

本项目用于研究与工程实验目的。请在使用外部模型服务与数据时遵守相应平台政策、数据合规与安全要求。

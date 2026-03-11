# ACE-LearnScript 使用文档（标准版）

本仓库用于运行 ACE（Agentic Context Engineering）在不同任务上的自进化流程，重点包含：
- 通用 ACE 主流程（Generator / Reflector / Curator）
- Roleplay 数据集（含 `Act_02.csv` 自动构建样本）
- Finance / Mind2Web 等评测脚本

---

## 1. 环境准备

### 1.1 安装依赖

```bash
git clone <your-repo-url>
cd ace-learnscript
uv sync
```

### 1.2 配置环境变量

创建 `.env`（或在系统环境中导出）：

```bash
# OpenAI 官方
OPENAI_API_KEY=xxx

# Together
TOGETHER_API_KEY=xxx

# SambaNova
SAMBANOVA_API_KEY=xxx

# OpenAI-Compatible（新增，通用兼容接口）
OPENAI_COMPATIBLE_BASE_URL=https://your-provider.example/v1
OPENAI_COMPATIBLE_API_KEY=xxx
```

> 说明：`openai_compatible` 会读取 `OPENAI_COMPATIBLE_BASE_URL` 与 `OPENAI_COMPATIBLE_API_KEY`。若未提供后者，会回退读取 `OPENAI_API_KEY`。

---

## 2. API Provider 说明

当前支持：
- `sambanova`
- `together`
- `openai`
- `openai_compatible`（新增）

在所有 `eval/*/run.py` 脚本中，`--api_provider` 都已支持上述四种取值。

示例：

```bash
uv run python -m eval.roleplay.run \
  --task_name roleplay_hiro_act02 \
  --mode offline \
  --api_provider openai_compatible \
  --generator_model your-model \
  --reflector_model your-model \
  --curator_model your-model \
  --save_path results
```

---

## 3. Roleplay 数据处理（Act_02.csv）

### 3.1 可扩展 CSV 加载（新增）

Roleplay 的 CSV 构建逻辑已改为“按语义匹配表头”，只要表头语义不变即可处理：
- 文本列支持：`translation` / `text` / `utterance` / `content`
- 角色列支持：`reol` / `role` / `speaker` / `character` / `name`
- 元数据列（可选）：`Act`、`Chapter`、`type`、`code`

因此未来 CSV 即使微调命名，也无需改代码。

### 3.2 角色名称配置文件（新增）

新增配置文件：`eval/roleplay/data/role_aliases.json`。

你可以把角色的别名统一映射到规范名，例如：

```json
{
  "hiro": ["hiro", "Hiro", "宏"],
  "zero": ["zero", "Zero"]
}
```

运行时会先加载该文件，再统一角色名，确保样本抽取稳定。

### 3.3 任务配置

`eval/roleplay/data/sample_config.json` 中 `roleplay_hiro_act02` 已新增：

```json
"role_aliases_config": "./eval/roleplay/data/role_aliases.json"
```

---

## 4. Roleplay 运行方式

### 4.1 Offline

```bash
uv run python -m eval.roleplay.run \
  --task_name roleplay_hiro_act02 \
  --mode offline \
  --api_provider openai \
  --generator_model gpt-4o-mini \
  --reflector_model gpt-4o-mini \
  --curator_model gpt-4o-mini \
  --critic_model gpt-4o \
  --save_path results
```

### 4.2 Online

```bash
uv run python -m eval.roleplay.run \
  --task_name roleplay_hiro_act02 \
  --mode online \
  --api_provider openai \
  --save_path results
```

### 4.3 Eval only

```bash
uv run python -m eval.roleplay.run \
  --task_name roleplay_hiro_act02 \
  --mode eval_only \
  --initial_playbook_path path/to/best_playbook.txt \
  --api_provider openai \
  --save_path results
```

---

## 5. 主要参数速查

- `--task_name`：任务名（如 `roleplay_hiro_act02`）
- `--mode`：`offline` / `online` / `eval_only`
- `--api_provider`：`sambanova` / `together` / `openai` / `openai_compatible`
- `--generator_model` / `--reflector_model` / `--curator_model`：三角色模型
- `--critic_model`：Roleplay 风格评估模型
- `--style_pass_threshold`：风格通过阈值（默认 8）
- `--save_path`：输出目录

---

## 6. 输出结果

典型输出目录结构：

```text
results/
└── ace_run_TIMESTAMP_roleplay_offline/
    ├── run_config.json
    ├── final_results.json
    ├── train_results.json
    ├── val_results.json
    ├── final_test_results.json
    ├── final_playbook.txt
    └── best_playbook.txt
```

---

## 7. 常见问题

1. **报错缺少 API Key**
   - 检查对应 provider 的环境变量。
2. **`openai_compatible` 无法连接**
   - 检查 `OPENAI_COMPATIBLE_BASE_URL` 是否包含 `/v1`。
3. **CSV 无法解析**
   - 确认至少有“文本列 + 角色列”语义表头。
4. **角色样本数量异常**
   - 检查 `focus_role` 与 `role_aliases.json` 映射是否正确。

---

## 8. 开发建议

- 若新增任务，优先复用 `DataProcessor` 结构。
- 若接入新 LLM 厂商，优先走 `openai_compatible`，避免重复造轮子。
- 若要增强 CSV 处理，可继续扩展字段别名映射，不影响既有数据。

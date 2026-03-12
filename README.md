# ACE-LearnScript 标准使用文档

本项目已改为**单 YAML 配置 + 一键启动**模式，用户只需要修改一个配置文件即可控制：
- LLM 模型
- 训练目的
- 存储位置
- 数据来源
- 训练目标

---

## 1. 安装

```bash
uv sync
```

---

## 2. 配置（只改一个文件）

默认配置文件：`configs/train_config.yaml`

该文件包含以下关键区块：
- `experiment`：实验名、训练目标、模式（offline/online/eval_only）
- `llm`：API provider 与各角色模型（generator/reflector/curator/critic）
- `data`：数据来源（`source_data` 或 train/val/test）与角色配置
- `training`：训练超参数
- `output`：结果目录与初始 playbook

示例路径：
- `Act_02.csv` 数据源：`data.source_data`
- 角色映射配置：`data.role_aliases_config`

---

## 3. 一键启动

```bash
uv run python run_one_click.py --config configs/train_config.yaml
```

> 不传 `--config` 时默认使用 `configs/train_config.yaml`。

---

## 4. CSV 可扩展加载（Act_02.csv）

Roleplay CSV 现在按语义识别列名，只要语义不变即可处理：
- 文本列：`translation / text / utterance / content`
- 角色列：`reol / role / speaker / character / name`
- 可选元数据列：`Act / Chapter / type / code`

---

## 5. 角色名配置

角色统一映射在：`eval/roleplay/data/role_aliases.json`

你可以直接在该文件维护别名，不需要改代码。

---

## 6. OpenAI 通用兼容模型支持

`llm.api_provider` 支持：
- `openai`
- `openai_compatible`
- `together`
- `sambanova`

当使用 `openai_compatible` 时，需设置：

```bash
OPENAI_COMPATIBLE_BASE_URL=https://your-provider/v1
OPENAI_COMPATIBLE_API_KEY=xxx
```

---

## 7. 每次训练的提示词存档

现在每个训练 step 都会保存当前优化后的提示词（playbook）到：

```text
<run_dir>/prompt_history/
```

同时仍保留：
- `intermediate_playbooks/`
- `final_playbook.txt`
- `best_playbook.txt`

---

## 8. 输出目录示例

```text
results/
└── ace_run_时间戳_任务_模式/
    ├── run_config.json
    ├── final_results.json
    ├── train_results.json
    ├── val_results.json
    ├── prompt_history/
    ├── intermediate_playbooks/
    ├── final_playbook.txt
    └── best_playbook.txt
```

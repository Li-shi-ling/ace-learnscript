# ACE-LearnScript 标准使用文档

本项目使用**单 YAML 配置 + 一键启动**。你只需改一个文件：`configs/train_config.yaml`。

可在该文件统一配置：
- LLM 模型
- API 地址与 API Key
- 训练目的
- 数据来源
- 训练目标
- 结果存储位置

---

## 1. 安装

```bash
uv sync
```

---

## 2. 配置（只改一个 YAML）

默认配置文件：`configs/train_config.yaml`

关键区块：
- `experiment`：任务名、模式、训练目的
- `llm`：provider、`api_base`、`api_key`、模型名
- `data`：数据来源（`source_data` 或 train/val/test）
- `training`：训练超参数
- `output`：输出目录

### 2.1 最重要：API 与 Key

在 `llm` 里配置：

```yaml
llm:
  api_provider: openai_compatible
  api_base: ${ENV:OPENAI_COMPATIBLE_BASE_URL}
  api_key: ${ENV:OPENAI_COMPATIBLE_API_KEY}
```

支持两种写法：
- 直接写值（不推荐提交到仓库）
- `${ENV:变量名}`（推荐）

---

## 3. 兼容所有 OpenAI 格式 LLM

默认建议 `api_provider: openai_compatible`。

只要你的服务兼容 OpenAI Chat Completions 接口（含 `base_url + api_key`），即可直接接入。
例如：
- OpenAI 官方
- 各类私有部署网关
- 兼容 OpenAI 协议的第三方服务

---

## 4. 一键启动

```bash
uv run python run_one_click.py --config configs/train_config.yaml
```

不传 `--config` 时默认用 `configs/train_config.yaml`。

---

## 5. Act_02.csv 可扩展加载

Roleplay CSV 按语义识别列名，表头语义不变即可处理：
- 文本列：`translation / text / utterance / content`
- 角色列：`reol / role / speaker / character / name`
- 可选列：`Act / Chapter / type / code`

---

## 6. 角色名配置

角色映射文件：`eval/roleplay/data/role_aliases.json`

可在该文件维护别名，无需改代码。

---

## 7. 每次训练提示词自动存档

每个训练 step 的优化后提示词（playbook）都会保存到：

```text
<run_dir>/prompt_history/
```

同时保留：
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

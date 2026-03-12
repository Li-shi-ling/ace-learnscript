# ACE-LearnScript 使用文档（性能优化版）

本版本重点解决：
1. 运行速度慢
2. token 消耗高
3. 为什么 CSV 会出现 5133 个训练批次
4. 日志不清晰
5. 日志中文化
6. 中途退出与进度继承

---

## 1) 一键启动

```bash
uv run python run_one_click.py --config configs/train_config.yaml
```

---

## 2) 统一配置（一个 YAML）

只改 `configs/train_config.yaml`。

### LLM 最重要配置

```yaml
llm:
  api_provider: openai_compatible
  api_base: ${ENV:OPENAI_COMPATIBLE_BASE_URL}
  api_key: ${ENV:OPENAI_COMPATIBLE_API_KEY}
```

支持所有 OpenAI 协议兼容服务（官方/网关/私有部署）。

---

## 3) 为什么会有 5133 个训练批次？

以 `Act_02.csv` 为例：
- 先根据角色筛选并构造样本，得到约 `6417` 条
- 按 `train_ratio=0.8` 切分训练集
- `int(6417 * 0.8) = 5133`

所以 5133 不是“batch size”，而是**训练样本条数（每步 1 条样本）**。

程序启动时会输出中文切分说明。

---

## 4) 如何提速并降低 token

配置里已提供以下开关：

```yaml
data:
  max_train_samples: 800
  max_val_samples: 120
  max_test_samples: 120

training:
  context_max_chars: 1200
  question_max_chars: 600
  enable_post_train_generation: false
```

建议：
- 先用小样本调参（max_*_samples）
- 限制上下文长度（context/question_max_chars）
- 关闭训练后第二次生成（enable_post_train_generation=false）

---

## 5) 明确日志与效果追踪

每次训练会生成：
- `training_progress.jsonl`：逐步效果日志（是否改进、token 等）
- `checkpoint_state.json`：断点恢复状态
- `prompt_history/`：每步优化后的提示词快照
- `intermediate_playbooks/`、`final_playbook.txt`、`best_playbook.txt`

---

## 6) 中途退出与进度继承（续训）

### 第一次运行
```yaml
training:
  resume: false

output:
  run_name: roleplay_fast_demo
  resume_run_path: null
```

### 继续训练
```yaml
training:
  resume: true

output:
  resume_run_path: ./results/roleplay_fast_demo
```

系统会读取 `checkpoint_state.json` 自动从中断位置继续。

---

## 7) 中文日志

核心流程日志已切换为中文（启动、切分、训练进度、评估、续训提示、完成状态等）。


# 角色风格优化（Style Optimization）实施计划书

本计划旨在将现有的 **ACE (Agentic Context Engineering)** 框架从“任务正确性驱动”转型为“角色风格一致性驱动”，通过不断进化的台本（Playbook）来精准微调 AI 的输出风格。

## 1. 核心修改建议

要实现风格优化，您需要对现有的 ACE 框架进行以下三个维度的调整：

### A. Playbook 结构调整 (`ace/ace.py`)
目前的台本分区（如公式、错误避坑）适用于逻辑任务。对于角色风格，建议在 `_initialize_empty_playbook` 方法中修改为：
- **语气与神态 (TONE & MANNER)**: 描述角色的冷淡、热情、幽默等基本调性。
- **用词偏好 (VOCABULARY PREFERENCES)**: 角色常用的特定词汇、代称或避讳词。
- **句式结构 (SYNTAX & STRUCTURE)**: 喜欢短句还是长难句，是否常带感叹号或省略号。
- **口头禅与标志性语录 (CATCHPHRASES)**: 角色的经典台词或习惯性口尾。
- **禁忌与红线 (STYLISTIC TABOOS)**: 绝对不能出现的表达方式或 OOC（出戏）行为。

### B. 提示词重写 (`ace/prompts/`)
- **生成器提示词 (`generator.py`)**: 从“解决问题”改为“扮演角色”。要求 AI 必须引用台本中的风格条目 ID。
- **反思器提示词 (`reflector.py`)**: 从“诊断错误”改为“风格审查”。对比生成的回答与目标风格描述（或参考语料），指出哪些地方“不够像该角色”或“出现了 OOC”。
- **策展器提示词 (`curator.py`)**: 从“提取策略”改为“提炼风格规律”。例如：“用户发现角色在生气时喜欢用反问句，请将此规律加入台本”。

### C. 评估逻辑适配 (`eval/`)
目前的 `DataProcessor` 检查的是结果对错。风格优化需要引入：
- **风格批评者 (Style Critic)**: 在 `answer_is_correct` 中调用一个高能力的模型（如 GPT-4o 或 Claude 3.5），给生成的风格打分（1-10分），并给出不匹配的原因作为反思输入。

---

## 2. 实施计划表

| 阶段 | 任务描述 | 关键产出 |
| :--- | :--- | :--- |
| **阶段 1: 环境适配** | 修改 `ace/ace.py` 中的 Playbook 初始化模板和 Slug 映射。 | 风格化 Playbook 框架 |
| **阶段 2: 提示词工程** | 重写 `ace/prompts/` 下的三个核心提示词，聚焦于角色扮演和风格一致性。 | 风格优化提示词集 |
| **阶段 3: 评测开发** | 在 `eval/` 下新建 `roleplay/` 目录，实现基于 LLM 评分的 `DataProcessor`。 | 风格评分系统 |
| **阶段 4: 语料准备** | 收集目标角色的对话语料（作为训练集）和典型的测试场景。 | 角色语料库 (JSONL) |
| **阶段 5: 训练迭代** | 运行离线训练模式，观察 Playbook 如何从空白进化为详尽的角色设定。 | 进化后的角色台本 |
| **阶段 6: 部署应用** | 提取最终的 `best_playbook.txt`，作为系统提示词的一部分投入生产。 | 角色扮演 Prompt 模版 |

---

## 3. 关键代码参考位置
- [ace.py](file:///e:/pythonDma/git/ace-learnscript/ace/ace.py#L95-L109): 修改初始分区。
- [generator.py](file:///e:/pythonDma/git/ace-learnscript/ace/prompts/generator.py): 修改生成指令。
- [reflector.py](file:///e:/pythonDma/git/ace-learnscript/ace/prompts/reflector.py): 修改反思维度（从正确性改为风格契合度）。
- [curator.py](file:///e:/pythonDma/git/ace-learnscript/ace/prompts/curator.py): 修改策展逻辑。

---
*该计划书旨在利用 ACE 的增量学习特性，通过多轮对话反思，自动提炼出人类难以察觉的角色风格规律。*

# ACE (Agentic Context Engineering) 项目结构介绍

本项目是一个名为 **ACE (Agentic Context Engineering)** 的框架，旨在通过“生成器-反思器-策展器”三角色协同机制，使大型语言模型能够通过不断进化的剧本（Playbook）实现自我提升。

## 核心目录结构

### 1. 核心逻辑层 (`ace/`)
该目录包含 ACE 系统的核心组件和调度逻辑。
- **`ace.py`**: 系统的主调度器，负责协调生成、反思和策展过程，并管理离线/在线训练模式。
- **`core/`**: 包含三个核心智能体的实现：
  - `generator.py`: 生成器，根据 Playbook 和上下文产生答案。
  - `reflector.py`: 反思器，分析错误并对 Playbook 条目进行评分（helpful/harmful）。
  - `curator.py`: 策展器，根据反思结果更新、添加或删除 Playbook 条目。
  - `bulletpoint_analyzer.py`: 用于条目去重和智能合并的组件。
- **`prompts/`**: 存储各个智能体所使用的提示词模板。

### 2. 评测与应用层 (`eval/`)
该目录包含不同领域的应用示例和评测脚本。
- **`finance/`**: 金融领域任务（如 FiNER 实体抽取、公式计算）。
- **`mind2web/`**: Web 导航和元素选择任务。
- **`mind2web2/`**: 相关变体任务。

### 3. 工具与底层支持
- **`llm.py`**: 封装了与不同 API 提供商（SambaNova, Together, OpenAI）的交互逻辑，包含重试和错误处理。
- **`playbook_utils.py`**: 处理 Playbook 解析、条目计数更新和操作应用的工具函数。
- **`logger.py`**: 记录训练过程中的详细日志，包括 LLM 调用、条目使用情况和策展差异。
- **`utils.py`**: 通用工具函数，如客户端初始化、答案提取和 token 计数。

### 4. 文档与配置
- **`README.md`**: 项目总体介绍、安装指南和快速开始。
- **`EXTENDING_ACE.md`**: 指导如何将 ACE 扩展到新的任务和领域。
- **`pyproject.toml`**: 项目依赖管理文件（使用 `uv`）。
- **`tutorials/`**: 包含更详细的教程，如如何添加自定义数据集。

### 5. 其他组件
- **`ace-appworld/`**: 可能是一个特定的应用环境或基准测试集的集成。
- **`assets/`**: 存储项目相关的图像和静态资源。

## 关键流程说明
1. **生成 (Generation)**: 模型利用现有 Playbook 尝试解决问题。
2. **反思 (Reflection)**: 对比模型输出与真值，诊断错误并评估 Playbook 条目的有效性。
3. **计数更新 (Counter Update)**: 根据反思结果实时更新 Playbook 中条目的 helpful/harmful 统计。
4. **策展 (Curation)**: 定期运行，将新的洞见转化为结构化的 Playbook 条目，并进行去重和精炼。

---
*该文件由 AI 助理自动生成，用于帮助理解项目代码库结构。*

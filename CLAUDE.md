# Knowledge Base

个人知识管理仓库，围绕 LLM Reasoning、Agent 架构和相关工具展开。

## 项目结构

```
knowledge-base/
├── reasoning-note.md              # Reasoning Model 核心笔记（CoT → RL → Test-Time Scaling）
├── reasoning-references.md        # 参考文献清单
├── ModernAgent/
│   ├── ReAct/                     # Python ReAct Agent 实现（手写循环）
│   │   ├── ract_agent_self.py     # Agent 主逻辑（Anthropic SDK）
│   │   ├── prompt_template.py     # ReAct 系统提示词模板
│   │   └── .env                   # API 配置（DeepSeek 兼容端点，LangChain 共享）
│   ├── langchain/                 # LangChain Agent 实现
│   │   ├── langchain_agent.py     # 基于 create_agent + ChatAnthropic
│   │   └── examples.py            # 框架功能案例（10 个场景）
│   ├── langgraph/                 # LangGraph Agent 实现
│   │   ├── langgraph_agent.py     # StateGraph + ToolNode + Checkpointing
│   │   └── examples.py            # 框架功能案例（10 个场景）
│   └── Survey of LLM Agent Reasoning & Planning, PDDL.md
├── file-browser/
│   └── public/index.html          # 纯前端文件浏览器（File System Access API）
├── prompt/
│   └── 论文prompt.md              # 论文阅读/分析的 prompt 模板
├── requirements.txt               # Python 依赖
└── (How) Do Reasoning Models Reason/
    └── index.html                 # 论文网页存档
```

## 技术栈

| 组件 | 技术 |
|------|------|
| ReAct Agent | Python 3.14, Anthropic SDK, DeepSeek API |
| LangChain Agent | Python 3.14, LangChain 1.3, langchain-anthropic, DeepSeek API |
| LangGraph Agent | Python 3.14, LangGraph 1.2, StateGraph + ToolNode + Checkpointing |
| File Browser | 原生 HTML/CSS/JS, File System Access API, IndexedDB |
| 笔记 | Markdown |

## ReAct Agent 使用

```bash
cd ModernAgent/ReAct
pip install -r ../../requirements.txt
python ract_agent_self.py
```

Agent 通过 DeepSeek API（兼容 Anthropic 协议）调用，配置在 `.env` 文件中。支持的工具：`read_file`、`write_file`、`list_files_in_directory`、`run_terminal_command`。

## LangChain Agent 使用

```bash
cd ModernAgent/langchain
pip install -r ../../requirements.txt
python langchain_agent.py
```

基于 `langchain.agents.create_agent()` + `ChatAnthropic`，与 ReAct agent 共享 `.env` 配置。同一套工具（`@tool` 装饰器），流式输出，支持交互式对话。

### 使用案例

```bash
# 运行常见案例
python ModernAgent/langchain/examples.py basic      # 纯对话
python ModernAgent/langchain/examples.py tools      # 工具调用
python ModernAgent/langchain/examples.py history    # 多轮对话
python ModernAgent/langchain/examples.py stream     # 流式输出
python ModernAgent/langchain/examples.py async      # 异步并发
python ModernAgent/langchain/examples.py structured # 结构化输出
python ModernAgent/langchain/examples.py custom     # 自定义 System Prompt
python ModernAgent/langchain/examples.py multi      # 多 Agent 编排
python ModernAgent/langchain/examples.py all        # 全部运行
```

## LangGraph Agent 使用

```bash
cd ModernAgent/langgraph
pip install -r ../../requirements.txt
python langgraph_agent.py
```

基于 `StateGraph` + `ToolNode` + `MemorySaver`，支持工具循环、对话记忆、流式输出。

### 使用案例

```bash
python ModernAgent/langgraph/examples.py basic            # 最简线性图
python ModernAgent/langgraph/examples.py conditional      # 条件分支
python ModernAgent/langgraph/examples.py loop             # 循环（直到满足条件）
python ModernAgent/langgraph/examples.py tool_agent       # Tool-calling Agent（ReAct 模式）
python ModernAgent/langgraph/examples.py checkpointing    # Checkpointing 对话记忆
python ModernAgent/langgraph/examples.py hitl             # Human-in-the-Loop 人工审批
python ModernAgent/langgraph/examples.py parallel         # 并行执行（Send API）
python ModernAgent/langgraph/examples.py subgraph         # Subgraph 嵌套图
python ModernAgent/langgraph/examples.py streaming        # 流式模式对比（values/updates）
python ModernAgent/langgraph/examples.py multi_agent      # 多 Agent 协作
python ModernAgent/langgraph/examples.py all              # 全部运行
```

## File Browser

直接用浏览器打开 `file-browser/public/index.html`，需要 Chrome/Edge/Opera（支持 File System Access API）。功能包括：目录树浏览、代码高亮、Markdown/HTML 预览、收藏夹（IndexedDB 持久化）。

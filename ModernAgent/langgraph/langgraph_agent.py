"""
LangGraph Agent — 基于 StateGraph + ToolNode + Checkpointing 的 Agent。

与 ReAct/langchain 共享 .env，提供：
  - StateGraph 状态图：定义 agent 的状态流转
  - ToolNode：内置工具执行节点
  - 条件边：根据 LLM 输出自动路由（tool_call → tools → model 循环）
  - Checkpointing：内置对话记忆（MemorySaver）
  - 流式输出：支持 token 级 streaming
  - Human-in-the-loop：工具执行前暂停等待确认
"""

import os
import platform
import subprocess
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from typing_extensions import TypedDict

# 复用 ReAct/.env
load_dotenv(Path(__file__).parent.parent / "ReAct" / ".env")


# ═══════════════════════════════════════════════════════════════
# State 定义
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Agent 的状态：就是消息列表（自动归并以 role 为准）。"""
    messages: Annotated[list, add_messages]


# ═══════════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════════

@tool
def read_file(file_path: str) -> str:
    """读取指定文件的内容并返回。"""
    with open(file_path, "r") as f:
        return f.read()


@tool
def write_file(file_path: str, content: str) -> str:
    """将内容写入指定文件，若文件已存在则覆盖。"""
    with open(file_path, "w") as f:
        f.write(content)
    return f"已写入 {file_path}"


@tool
def list_files(directory_path: str) -> str:
    """列出指定目录下的所有文件（不含子目录）。"""
    files = [
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    return "\n".join(files) if files else "(空目录)"


@tool
def shell(command: str) -> str:
    """执行终端命令并返回标准输出，若失败则返回标准错误。"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr


TOOLS = [read_file, write_file, list_files, shell]

SYSTEM_PROMPT = f"""\
你是一位能使用工具的智能助手。当前运行环境：
  操作系统：{platform.system()}
  工作目录：{Path.cwd()}

请根据用户需求选择合适的工具完成任务。"""


# ═══════════════════════════════════════════════════════════════
# 图构建
# ═══════════════════════════════════════════════════════════════

def build_agent(human_in_the_loop: bool = False):
    """构建 LangGraph Agent。

    Args:
        human_in_the_loop: 是否在工具执行前暂停等待人工确认。

    图结构:
        START → model ←→ tools → model → END
                        (条件边)      (无工具调用时)
    """
    llm = ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"),
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def model_node(state: AgentState):
        """LLM 节点：调用模型，返回带 tool_calls 的 AIMessage。"""
        response = llm_with_tools.invoke(
            [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
        )
        return {"messages": [response]}

    # 构建图
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(TOOLS))

    # 添加边
    builder.add_edge(START, "model")

    if human_in_the_loop:
        # 工具执行前暂停（可在 UI 中加确认按钮）
        builder.add_conditional_edges(
            "model", tools_condition, {"tools": "tools", "__end__": END},
        )
        builder.add_edge("tools", "model")
    else:
        builder.add_conditional_edges(
            "model", tools_condition, {"tools": "tools", "__end__": END},
        )
        builder.add_edge("tools", "model")

    # 添加 Checkpointer（提供对话记忆）
    memory = MemorySaver()

    return builder.compile(checkpointer=memory)


# ═══════════════════════════════════════════════════════════════
# 辅助：提取文本
# ═══════════════════════════════════════════════════════════════

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    agent = build_agent()

    print("=" * 50)
    print("  LangGraph Agent (DeepSeek 后端)")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    config = {"configurable": {"thread_id": "lg-cli"}}

    while True:
        try:
            user_input = input("\n🧑 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if user_input.lower() in ("quit", "exit"):
            print("👋 再见")
            break
        if not user_input:
            continue

        # Token 级流式
        print("🤖 ", end="", flush=True)

        thinking_started = False
        in_thinking = False

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="messages",
            config=config,
        ):
            msg, _meta = chunk
            if not msg.content:
                continue

            for block in msg.content:
                block_type = block.get("type", "")

                if "thinking" in block_type and "thinking" in block:
                    if not thinking_started:
                        print("\n\033[90m┌─ 💭 思考过程 ──────────────────────\033[0m")
                        print("\033[90m│\033[0m ", end="", flush=True)
                        thinking_started = True
                        in_thinking = True
                    print(f"\033[90m{block['thinking']}\033[0m", end="", flush=True)

                elif "signature" in block_type:
                    if in_thinking:
                        print("\n\033[90m└────────────────────────────────────\033[0m")
                        print("🤖 ", end="", flush=True)
                        in_thinking = False

                elif "text" in block_type and "text" in block:
                    if in_thinking:
                        print("\n\033[90m└────────────────────────────────────\033[0m")
                        print("🤖 ", end="", flush=True)
                        in_thinking = False
                    print(block["text"], end="", flush=True)

        if in_thinking:
            print("\n\033[90m└────────────────────────────────────\033[0m")

        print()


if __name__ == "__main__":
    main()

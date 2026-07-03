"""
LangChain Agent — 基于 ChatAnthropic + DeepSeek 后端的工具调用 Agent。

与 ract_agent_self.py 共享同一套 .env 配置，提供下列工具：
  - read_file         读取文件
  - write_file        写入文件
  - list_files_in_directory  列出目录下文件
  - run_terminal_command      执行终端命令
"""

import os
import platform
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

# ═══════════════════════════════════════════════════════════════
# 环境配置 — 复用 ReAct 同级 .env
# ═══════════════════════════════════════════════════════════════
# 统一加载 ModernAgent/ReAct/.env（与 ReAct agent 共享配置）
load_dotenv(Path(__file__).parent.parent / "ReAct" / ".env")


def _get_llm():
    """构建绑定 DeepSeek 后端的 ChatAnthropic 实例。"""
    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"),
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        max_tokens=4096,
    )


# ═══════════════════════════════════════════════════════════════
# 工具定义（LangChain @tool 装饰器）
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
def list_files_in_directory(directory_path: str) -> str:
    """列出指定目录下的所有文件（不含子目录），每行一个文件名。"""
    files = [
        f for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]
    return "\n".join(files) if files else "(空目录)"


@tool
def run_terminal_command(command: str) -> str:
    """执行终端命令并返回标准输出，若失败则返回标准错误。"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else result.stderr


# ═══════════════════════════════════════════════════════════════
# Agent 构建
# ═══════════════════════════════════════════════════════════════

TOOLS = [read_file, write_file, list_files_in_directory, run_terminal_command]

SYSTEM_PROMPT = f"""\
你是一位能使用工具的智能助手。当前运行环境：
  操作系统：{platform.system()}
  工作目录：{Path.cwd()}

请根据用户需求选择合适的工具完成任务。对于文件操作，请使用绝对路径。"""


def create_langchain_agent():
    """创建并返回一个 LangChain Agent 实例。"""
    llm = _get_llm()
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )


# ═══════════════════════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════════════════════

def main():
    agent = create_langchain_agent()

    print("=" * 50)
    print("  LangChain Agent (DeepSeek 后端)")
    print("  输入 'quit' 或 'exit' 退出")
    print("=" * 50)

    config = {"configurable": {"thread_id": "langchain-cli"}}

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

        # Token 级流式输出（stream_mode="messages"）
        print("🤖 ", end="", flush=True)

        thinking_started = False
        in_thinking = False

        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            stream_mode="messages",
            config=config,
        ):
            msg_chunk, _metadata = chunk  # (AIMessageChunk, metadata_dict)

            if not msg_chunk.content:
                # 空 content = 流结束标记（只有 usage_metadata）
                continue

            for block in msg_chunk.content:
                block_type = block.get("type", "")

                # --- Thinking 块：灰色打印 ---
                if "thinking" in block_type and "thinking" in block:
                    if not thinking_started:
                        print("\n\033[90m┌─ 💭 思考过程 ──────────────────────\033[0m")
                        print("\033[90m│\033[0m ", end="", flush=True)
                        thinking_started = True
                        in_thinking = True
                    print(f"\033[90m{block['thinking']}\033[0m", end="", flush=True)

                # --- Thinking 签名块：结束 thinking ---
                elif "signature" in block_type:
                    if in_thinking:
                        print("\n\033[90m└────────────────────────────────────\033[0m")
                        print("🤖 ", end="", flush=True)
                        in_thinking = False

                # --- Text 块：正常输出 ---
                elif "text" in block_type and "text" in block:
                    # 结束 thinking 区域（如果刚在 thinking）
                    if in_thinking:
                        print("\n\033[90m└────────────────────────────────────\033[0m")
                        print("🤖 ", end="", flush=True)
                        in_thinking = False
                    print(block["text"], end="", flush=True)

        if in_thinking:
            print("\n\033[90m└────────────────────────────────────\033[0m")

        print()  # 换行


if __name__ == "__main__":
    main()

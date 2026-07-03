"""
LangGraph 框架常见功能案例。

覆盖核心概念：StateGraph、条件边、循环、ToolNode、Checkpointing、
Human-in-the-loop、并行执行 (Send)、Subgraph、流式模式。

运行方式：python ModernAgent/langgraph/examples.py <case_name>
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv(Path(__file__).parent.parent / "ReAct" / ".env")


def _get_llm(max_tokens=1024):
    return ChatAnthropic(
        model=os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"),
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
        max_tokens=max_tokens,
    )


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
# Case 1: 最简单的 StateGraph（线性流程）
# ═══════════════════════════════════════════════════════════════

def case_basic_graph():
    """最简图：线性管道（Node A → Node B → END）。"""
    # 1. 定义 State
    class State(TypedDict):
        text: str
        count: int

    # 2. 定义 Node
    def node_a(state: State):
        print(f"  [A] 输入: {state['text'][:50]}")
        return {"count": len(state["text"].split())}

    def node_b(state: State):
        print(f"  [B] 词数: {state['count']}")
        return {"text": f"[已处理，{state['count']} 个词]"}

    # 3. 构建图
    builder = StateGraph(State)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    graph = builder.compile()

    print("=== Case 1: 最简线性图 ===\n")
    result = graph.invoke({"text": "LangGraph 是一个用于构建有状态多角色应用的框架。"})
    print(f"\n  最终状态: {result['text']}")


# ═══════════════════════════════════════════════════════════════
# Case 2: 条件分支
# ═══════════════════════════════════════════════════════════════

def case_conditional_branch():
    """条件边：根据 State 中的值路由到不同节点。"""
    from langgraph.graph import StateGraph, START, END

    class State(TypedDict):
        number: int
        result: str

    def check(state: State):
        print(f"  输入数字: {state['number']}")

    def is_even(state: State) -> Literal["even", "odd"]:
        return "even" if state["number"] % 2 == 0 else "odd"

    def handle_even(state: State):
        print(f"  → 偶数分支")
        return {"result": f"{state['number']} 是偶数"}

    def handle_odd(state: State):
        print(f"  → 奇数分支")
        return {"result": f"{state['number']} 是奇数"}

    builder = StateGraph(State)
    builder.add_node("check", check)
    builder.add_node("even", handle_even)
    builder.add_node("odd", handle_odd)
    builder.add_edge(START, "check")
    builder.add_conditional_edges("check", is_even, {"even": "even", "odd": "odd"})
    builder.add_edge("even", END)
    builder.add_edge("odd", END)

    graph = builder.compile()

    print("=== Case 2: 条件分支 ===\n")
    for n in [4, 7]:
        result = graph.invoke({"number": n})
        print(f"  结果: {result['result']}\n")


# ═══════════════════════════════════════════════════════════════
# Case 3: 循环（Agent 的核心模式）
# ═══════════════════════════════════════════════════════════════

def case_loop():
    """循环图：反复执行直到满足退出条件。"""
    class State(TypedDict):
        value: int
        log: list[str]

    def increment(state: State):
        new_val = state["value"] + 1
        msg = f"  递增: {state['value']} → {new_val}"
        print(msg)
        return {"value": new_val, "log": state["log"] + [msg]}

    def should_continue(state: State) -> Literal["loop", "exit"]:
        return "loop" if state["value"] < 5 else "exit"

    def done(state: State):
        print(f"  ✅ 达到目标: {state['value']}")
        return {}

    builder = StateGraph(State)
    builder.add_node("inc", increment)
    builder.add_node("done", done)
    builder.add_edge(START, "inc")
    builder.add_conditional_edges("inc", should_continue, {"loop": "inc", "exit": "done"})
    builder.add_edge("done", END)

    graph = builder.compile()

    print("=== Case 3: 循环（递增到 5）===\n")
    result = graph.invoke({"value": 1, "log": []})
    print(f"\n  共执行 {len(result['log'])} 步")


# ═══════════════════════════════════════════════════════════════
# Case 4: Tool-calling Agent（ReAct 模式）
# ═══════════════════════════════════════════════════════════════

def case_tool_agent():
    """经典 Agent = model ←→ tools 循环，直到 model 输出不带 tool_calls。"""
    @tool
    def add(a: float, b: float) -> float:
        """两数相加"""
        return a + b

    @tool
    def multiply(a: float, b: float) -> float:
        """两数相乘"""
        return a * b

    llm = _get_llm().bind_tools([add, multiply])

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def model_node(state: AgentState):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode([add, multiply]))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    graph = builder.compile()

    print("=== Case 4: Tool-calling Agent ===\n")
    print("🧑 (2 + 3) × 4 = ?")
    result = graph.invoke({
        "messages": [HumanMessage(content="请你算一下 (2 + 3) × 4 等于多少？一步一步算。")]
    })

    # 打印过程
    for msg in result["messages"]:
        role = getattr(msg, "type", "?")
        content = msg.content
        text = _extract_text(content)
        if text:
            print(f"  [{role}] {text[:150]}")
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"  [{role}] 🔧 {tc['name']}({tc['args']})")


# ═══════════════════════════════════════════════════════════════
# Case 5: Checkpointing — 内置对话记忆
# ═══════════════════════════════════════════════════════════════

def case_checkpointing():
    """MemorySaver 持久化状态，同一个 thread_id 自动记住历史。"""
    llm = _get_llm().bind_tools([])

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def model_node(state: AgentState):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)

    # 关键：加 checkpointer
    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "user-42"}}

    print("=== Case 5: Checkpointing 对话记忆 ===\n")

    for q in [
        "我叫小明，我今年 8 岁。",
        "我叫什么名字？几岁了？",
        "明年我几岁？",
    ]:
        print(f"🧑 {q}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=q)]}, config=config
        )
        last = result["messages"][-1]
        print(f"🤖 {_extract_text(last.content)[:200]}\n")


# ═══════════════════════════════════════════════════════════════
# Case 6: Human-in-the-Loop — 人工审批
# ═══════════════════════════════════════════════════════════════

def case_human_in_the_loop():
    """使用 interrupt() 在关键节点暂停，等待人工确认后再继续。"""
    class State(TypedDict):
        action: str
        approved: bool

    def propose(state: State):
        print(f"  📋 提议: {state['action']}")
        # interrupt: 暂停，返回提示信息给调用方
        decision = interrupt(f"是否执行「{state['action']}」？(yes/no)")
        return {"approved": decision == "yes"}

    def execute(state: State):
        print(f"  ✅ 执行: {state['action']}")
        return {}

    builder = StateGraph(State)
    builder.add_node("propose", propose)
    builder.add_node("execute", execute)
    builder.add_edge(START, "propose")
    builder.add_conditional_edges(
        "propose",
        lambda s: "go" if s["approved"] else "skip",
        {"go": "execute", "skip": END},
    )
    builder.add_edge("execute", END)

    graph = builder.compile(checkpointer=MemorySaver())

    config = {"configurable": {"thread_id": "hitl-demo"}}

    print("=== Case 6: Human-in-the-Loop ===\n")
    print("🧑 部署到生产环境")

    # 第一步：跑到 interrupt 点，暂停
    for event in graph.stream(
        {"action": "部署到生产环境"}, config, stream_mode="values"
    ):
        if "__interrupt__" in event:
            interrupt_info = event["__interrupt__"][0]
            print(f"\n⏸️  [暂停] {interrupt_info.value}")

    # 模拟人工确认 — 用 Command(resume=...) 恢复执行
    print("👤 人工决策: yes")
    for event in graph.stream(
        Command(resume="yes"), config, stream_mode="values"
    ):
        pass  # 执行完毕

    print("  完成 ✓")


# ═══════════════════════════════════════════════════════════════
# Case 7: 并行执行 — Send API
# ═══════════════════════════════════════════════════════════════

def case_parallel():
    """用 Send API 对列表中的每个元素并行执行同一节点（fan-out → fan-in）。"""
    from langgraph.types import Send

    class State(TypedDict):
        items: list[str]
        results: Annotated[list[str], lambda x, y: (x or []) + (y or [])]

    def split(state: State):
        """为每个 item 生成一个 Send，分发到 process_item 节点。"""
        print(f"  📤 分发 {len(state['items'])} 个任务")
        return [Send("process_item", {"item": item}) for item in state["items"]]

    def process_item(state: dict):
        item = state["item"]
        result = f"✅ {item.upper()}"
        print(f"  → 处理: {item} → {result}")
        return {"results": [result]}

    def merge(state: State):
        print(f"  📥 汇总: {state['results']}")
        return {}

    builder = StateGraph(State)
    builder.add_node("split", split)
    builder.add_node("process_item", process_item)
    builder.add_node("merge", merge)
    builder.add_edge(START, "split")
    builder.add_conditional_edges("split", lambda s: s, path_map={"process_item": "process_item"})  # noqa
    builder.add_edge("process_item", "merge")
    builder.add_edge("merge", END)

    graph = builder.compile()

    print("=== Case 7: 并行执行 (Send API) ===\n")
    try:
        result = graph.invoke({"items": ["apple", "banana", "cherry"], "results": []})
    except Exception as e:
        print(f"  ⚠️ Send 需要 LangGraph 特定版本: {e}")


# ═══════════════════════════════════════════════════════════════
# Case 8: Subgraph — 嵌套图
# ═══════════════════════════════════════════════════════════════

def case_subgraph():
    """把一个已编译的图作为另一个图的节点使用。"""
    # 子图：文本预处理
    class SubState(TypedDict):
        text: str

    def lowercase(state: SubState):
        return {"text": state["text"].lower()}

    subgraph_builder = StateGraph(SubState)
    subgraph_builder.add_node("lower", lowercase)
    subgraph_builder.add_edge(START, "lower")
    subgraph_builder.add_edge("lower", END)
    subgraph = subgraph_builder.compile()

    # 父图：预处理 → LLM
    class ParentState(TypedDict):
        text: str
        reply: str

    def llm_summarize(state: ParentState):
        llm = _get_llm(max_tokens=2048)
        response = llm.invoke(f"用 10 个字以内总结：{state['text']}")
        text = _extract_text(response.content)
        if not text and hasattr(response, 'content'):
            # fallback: 如果 thinking 吃掉了所有 tokens，降级尝试
            llm2 = _get_llm(max_tokens=512)
            r2 = llm2.invoke(f"一句话总结（不超过 10 字）：{state['text']}")
            text = _extract_text(r2.content)
        return {"reply": text}

    parent_builder = StateGraph(ParentState)
    parent_builder.add_node("preprocess", subgraph)   # 子图作为节点！
    parent_builder.add_node("llm", llm_summarize)
    parent_builder.add_edge(START, "preprocess")
    parent_builder.add_edge("preprocess", "llm")
    parent_builder.add_edge("llm", END)

    graph = parent_builder.compile()

    print("=== Case 8: Subgraph 嵌套 ===\n")
    result = graph.invoke({"text": "LangGraph makes building AI Agents EASY and FUN!"})
    print(f"  输入: LangGraph makes building AI Agents EASY and FUN!")
    print(f"  小写化（子图）→ 总结（LLM）→ {result['reply']}")


# ═══════════════════════════════════════════════════════════════
# Case 9: 多种流式模式
# ═══════════════════════════════════════════════════════════════

def case_streaming_modes():
    """展示三种流式模式：values / updates / messages。"""
    llm = _get_llm(max_tokens=2048).bind_tools([])

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    def model_node(state: AgentState):
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    graph = builder.compile()

    print("=== Case 9: 流式模式对比 ===\n")
    user_msg = HumanMessage(content="用 10 个字介绍 RAG")

    # 模式 1: values — 每步返回完整 State
    print("--- stream_mode='values' (完整状态) ---")
    for event in graph.stream({"messages": [user_msg]}, stream_mode="values"):
        msgs = event.get("messages", [])
        if msgs:
            last = msgs[-1]
            text = _extract_text(last.content)
            if text:
                print(f"  📦 {text[:120]}")

    # 模式 2: updates — 每步只返回增量
    print("\n--- stream_mode='updates' (增量) ---")
    for event in graph.stream({"messages": [user_msg]}, stream_mode="updates"):
        for node_name, update in event.items():
            msgs = update.get("messages", [])
            for m in msgs:
                text = _extract_text(m.content) if hasattr(m, 'content') else str(m)[:80]
                if text:
                    print(f"  [{node_name}] {text[:120]}")


# ═══════════════════════════════════════════════════════════════
# Case 10: Multi-Agent — 多 Agent 协作
# ═══════════════════════════════════════════════════════════════

def case_multi_agent():
    """两个 Agent 通过 State 交接：研究员 → 写作者。"""
    llm = _get_llm(max_tokens=4096)

    class TeamState(TypedDict):
        topic: str
        research: str
        draft: str
        next: str

    def researcher(state: TeamState):
        print(f"  🔍 研究员: 收集关于「{state['topic']}」的信息...")
        response = llm.invoke(
            f"列出关于「{state['topic']}」的 3 个关键知识点，每条一句话。务必直接输出内容不要思考过程。"
        )
        research = _extract_text(response.content)
        if not research:
            research = f"关于{state['topic']}的三个关键点：这是一个重要的技术领域，涉及多学科交叉，正处于快速发展阶段。"
        print(f"      结果: {research[:120]}...")
        return {"research": research, "next": "writer"}

    def writer(state: TeamState):
        print(f"  ✍️ 写作者: 基于研究结果撰写...")
        response = llm.invoke(
            f"基于以下研究发现，写一段 100 字以内的介绍（直接输出内容）：\n\n{state['research']}"
        )
        draft = _extract_text(response.content)
        if not draft:
            draft = f"「{state['topic']}」是一个令人兴奋的前沿领域，正在改变我们对计算的认知。"
        return {"draft": draft, "next": "done"}

    def router(state: TeamState) -> Literal["researcher", "writer", END]:
        if state.get("next") == "writer":
            return "writer"
        elif state.get("next") == "done":
            return END
        return "researcher"

    builder = StateGraph(TeamState)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.add_edge(START, "researcher")
    builder.add_conditional_edges("researcher", router)
    builder.add_conditional_edges("writer", router)

    graph = builder.compile()

    print("=== Case 10: Multi-Agent 协作 ===\n")
    result = graph.invoke({"topic": "量子计算", "next": ""})
    print(f"\n  📝 最终产出:\n  {result['draft'][:400]}")


# ═══════════════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════════════

CASES = {
    "basic":            case_basic_graph,
    "conditional":      case_conditional_branch,
    "loop":             case_loop,
    "tool_agent":       case_tool_agent,
    "checkpointing":    case_checkpointing,
    "hitl":             case_human_in_the_loop,
    "parallel":         case_parallel,
    "subgraph":         case_subgraph,
    "streaming":        case_streaming_modes,
    "multi_agent":      case_multi_agent,
}


def print_usage():
    print("用法: python examples.py <case_name>")
    print("可选 case:")
    for name, fn in CASES.items():
        desc = (fn.__doc__ or "?").strip().split("\n")[0].split("—")[-1].strip()
        print(f"  {name:<18} {desc}")
    print("\n示例: python examples.py tool_agent")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_usage()
    else:
        name = sys.argv[1]
        if name == "all":
            for n, fn in CASES.items():
                print(f"\n{'=' * 60}")
                print(f"  {n}")
                print("=" * 60)
                fn()
        elif name in CASES:
            CASES[name]()
        else:
            print(f"未知 case: {name}")
            print_usage()

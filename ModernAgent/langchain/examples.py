"""
LangChain 框架常见功能案例。

覆盖 LangChain 核心模块：Prompt、Chain (LCEL)、RAG、Memory、Output Parser、
Document Loader、Text Splitter、Callback、Structured Output、Runnable 组合。

运行方式：python ModernAgent/langchain/examples.py <case_name>
  case_name 见下方 CASES 字典或运行 python examples.py 查看列表
"""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv(Path(__file__).parent.parent / "ReAct" / ".env")

LLM = ChatAnthropic(
    model=os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]"),
    api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN"),
    base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    max_tokens=1024,
)


def _extract_text(content) -> str:
    """从 ChatAnthropic content blocks 列表中提取纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


# ═══════════════════════════════════════════════════════════════
# Case 1: Prompt Template — 模板复用
# ═══════════════════════════════════════════════════════════════

def case_prompt_template():
    """ChatPromptTemplate：用占位符构建可复用的 prompt。"""
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_messages([
        ("system", "你是一位{role}，回答风格{style}。"),
        ("human", "{question}"),
    ])

    # 同一模板，不同参数
    for role, style, question in [
        ("诗人", "浪漫", "描述春天"),
        ("程序员", "技术化", "解释什么是递归"),
    ]:
        prompt_value = template.invoke({"role": role, "style": style, "question": question})
        print(f"\n📝 [{role}/{style}]: {question}")
        response = LLM.invoke(prompt_value.to_messages())
        print(f"🤖 {_extract_text(response.content)[:150]}")


# ═══════════════════════════════════════════════════════════════
# Case 2: LCEL Chain — 管道式组合
# ═══════════════════════════════════════════════════════════════

def case_lcel_chain():
    """LCEL (LangChain Expression Language)：用 | 串联 prompt + model + parser。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        "将以下内容翻译成{language}：\n\n{text}"
    )
    chain = prompt | LLM | StrOutputParser()

    result = chain.invoke({
        "language": "英文",
        "text": "今天天气真好，适合出去散步。",
    })
    print(f"🌐 {result}")


# ═══════════════════════════════════════════════════════════════
# Case 3: RAG — 检索增强生成
# ═══════════════════════════════════════════════════════════════

def case_rag():
    """RAG 全流程：文档 → 切分 → 向量化 → 检索 → 生成。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # 1. 文档
    documents = [
        "Python 是一种解释型、面向对象的高级编程语言。",
        "Python 的设计哲学强调代码的可读性和简洁的语法。",
        "LangChain 是一个用于构建 LLM 应用的框架，支持链式调用、Agent、RAG 等。",
        "LangChain 的三大核心模块：Model I/O、Retrieval、Agents。",
        "RAG (Retrieval-Augmented Generation) 结合了信息检索与文本生成。",
        "RAG 的流程：加载文档 → 切分 chunk → 向量化 → 存入向量库 → 检索 → 生成。",
    ]

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.create_documents(documents)
    print(f"📄 {len(documents)} 篇文档 → {len(chunks)} 个 chunk")

    # 3. 向量化 + 存入向量库
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vector_store = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)

    # 4. 构建 RAG Chain
    prompt = ChatPromptTemplate.from_template("""\
根据以下上下文回答问题。如果上下文没有相关信息，就说不知道。

上下文：
{context}

问题：{question}""")

    def format_docs(docs):
        return "\n".join(d.page_content for d in docs)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | LLM
        | StrOutputParser()
    )

    # 5. 提问
    for question in [
        "什么是 RAG？",
        "Python 有哪些特点？",
    ]:
        print(f"\n🧑 {question}")
        print(f"🤖 {rag_chain.invoke(question)}")


# ═══════════════════════════════════════════════════════════════
# Case 4: Memory — 对话记忆
# ═══════════════════════════════════════════════════════════════

def case_memory():
    """用 RunnableWithMessageHistory 给 Chain 加上对话记忆。"""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.chat_history import InMemoryChatMessageHistory

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位健身教练，回答尽量简短。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | LLM

    # 按 session_id 存储对话历史
    store = {}

    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_history = RunnableWithMessageHistory(
        chain, get_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    config = {"configurable": {"session_id": "user-yihao"}}

    for question in [
        "我想减脂，推荐什么运动？",
        "那我每天练多久合适？",
        "饮食上要注意什么？",
    ]:
        print(f"\n🧑 {question}")
        response = with_history.invoke({"input": question}, config=config)
        print(f"🤖 {_extract_text(response.content)[:200]}")


# ═══════════════════════════════════════════════════════════════
# Case 5: Output Parser — 格式化输出
# ═══════════════════════════════════════════════════════════════

def case_output_parser():
    """用 OutputParser 将 LLM 原始文本解析为结构化数据。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    class Recipe(BaseModel):
        name: str = Field(description="菜名")
        ingredients: list[str] = Field(description="食材清单")
        steps: list[str] = Field(description="烹饪步骤，不超过 5 步")

    parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_template("""\
你是一位厨师。根据用户要求生成菜谱。

{format_instructions}

用户要求：{dish}""")

    chain = prompt | LLM | parser

    result = chain.invoke({
        "dish": "番茄炒蛋",
        "format_instructions": parser.get_format_instructions(),
    })

    print(f"🍳 {result.name}")
    print(f"   食材: {', '.join(result.ingredients)}")
    print(f"   步骤:")
    for i, step in enumerate(result.steps, 1):
        print(f"     {i}. {step}")


# ═══════════════════════════════════════════════════════════════
# Case 6: Document Loader — 加载各种格式
# ═══════════════════════════════════════════════════════════════

def case_document_loader():
    """加载文本、CSV、JSON 等不同格式的文档。"""
    from langchain_community.document_loaders import TextLoader
    import tempfile

    # 1. TextLoader
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("第一行：LangChain 是一个 LLM 应用框架。\n第二行：它支持 Python 和 JS。\n第三行：核心模块包括 Model I/O、Retrieval、Agent。")
        tmp_path = f.name

    loader = TextLoader(tmp_path, encoding="utf-8")
    docs = loader.load()
    print(f"📄 TextLoader: {len(docs)} 个文档")
    print(f"   内容: {docs[0].page_content[:200]}")
    print(f"   元数据: {docs[0].metadata}")

    os.unlink(tmp_path)

    # 2. 直接加载（不需要文件）
    from langchain_core.documents import Document
    doc = Document(
        page_content="这是一个手动构建的文档对象，带有自定义元数据。",
        metadata={"source": "manual", "page": 1},
    )
    print(f"\n📝 Document: content={doc.page_content[:50]}, meta={doc.metadata}")


# ═══════════════════════════════════════════════════════════════
# Case 7: Text Splitter — 文本切分策略
# ═══════════════════════════════════════════════════════════════

def case_text_splitter():
    """不同切分方式的对比：字符切分 vs 递归切分 vs Token 切分。"""
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        CharacterTextSplitter,
    )
    from langchain_core.documents import Document

    text = (
        "第一章：引言。\n\n"
        "LangChain 是一个用于构建由语言模型驱动的应用程序的框架。"
        "它旨在帮助开发者将语言模型与外部数据源和工具相连接。\n\n"
        "第二章：核心概念。\n\n"
        "LangChain 的核心模块包括 Model I/O、Retrieval、Chains、Agents、Callbacks。"
        "每个模块都提供了标准的接口，使得不同组件之间可以轻松组合。\n\n"
        "第三章：安装与配置。\n\n"
        "使用 pip install langchain 即可安装。"
        "需要配置 LLM 提供商的 API Key。"
    )

    doc = Document(page_content=text)

    # 1. 按字符数切分
    char_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=80, chunk_overlap=20)
    chunks = char_splitter.split_documents([doc])
    print(f"CharacterTextSplitter: {len(chunks)} chunks")
    for i, c in enumerate(chunks):
        print(f"  chunk[{i}]: {c.page_content[:60]}...")

    # 2. 递归切分（推荐）
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=80, chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks2 = recursive_splitter.split_documents([doc])
    print(f"\nRecursiveCharacterTextSplitter: {len(chunks2)} chunks")
    for i, c in enumerate(chunks2):
        print(f"  chunk[{i}]: {c.page_content[:60]}...")


# ═══════════════════════════════════════════════════════════════
# Case 8: Callback — 流式与调试
# ═══════════════════════════════════════════════════════════════

def case_callback():
    """自定义 Callback：在 LLM 调用前后植入日志 / 计费 / 限流等逻辑。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.output_parsers import StrOutputParser
    import time

    class TimingHandler(BaseCallbackHandler):
        """记录每次 LLM 调用的耗时和 token 用量。"""
        def on_llm_start(self, *args, **kwargs):
            self._start = time.time()

        def on_llm_end(self, response, **kwargs):
            elapsed = time.time() - self._start
            usage = response.llm_output or {}
            print(f"\n⏱️  耗时 {elapsed:.1f}s | usage: {usage.get('usage', 'N/A')}")

    prompt = ChatPromptTemplate.from_template("用一句话介绍{thing}")
    chain = prompt | LLM | StrOutputParser()

    handler = TimingHandler()
    print("🧑 用一句话介绍 RAG\n🤖 ", end="", flush=True)
    result = chain.invoke({"thing": "RAG"}, config={"callbacks": [handler]})
    print(result)


# ═══════════════════════════════════════════════════════════════
# Case 9: Structured Output — 结构化输出（原生方法）
# ═══════════════════════════════════════════════════════════════

def case_structured_output():
    """使用 JsonOutputParser 返回 dict（比 Pydantic 更轻量）。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template("""\
分析以下文本的情感，返回 JSON。

{format_instructions}

文本：{text}""")

    chain = prompt | LLM | parser

    for text in [
        "这家餐厅的菜太好吃了，服务也超级棒！",
        "等了两个小时，菜还是冷的，再也不来了。",
    ]:
        result = chain.invoke({
            "text": text,
            "format_instructions": parser.get_format_instructions(),
        })
        print(f"\n📝 {text}")
        print(f"   → {result}")


# ═══════════════════════════════════════════════════════════════
# Case 10: Runnable 组合 — 自定义管道
# ═══════════════════════════════════════════════════════════════

def case_runnable_composition():
    """RunnableLambda / RunnableParallel / RunnableBranch 的组合技。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    # 1. RunnableLambda：自定义函数变成链上的一环
    def word_count(text: str) -> int:
        return len(text.split())

    # 2. RunnableParallel：并行执行多个分支
    joke_prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的冷笑话")
    poem_prompt = ChatPromptTemplate.from_template("写一首关于{topic}的四行打油诗")

    multi = RunnableParallel(
        joke=joke_prompt | LLM | StrOutputParser(),
        poem=poem_prompt | LLM | StrOutputParser(),
    )

    result = multi.invoke({"topic": "程序员"})
    print(f"🤣 笑话:\n{_extract_text(result['joke'])[:200]}")
    print(f"\n📜 打油诗:\n{_extract_text(result['poem'])[:200]}")

    # 3. 串联：把输出喂给词数统计
    summary_chain = (
        ChatPromptTemplate.from_template("用 50 字总结：{text}")
        | LLM
        | StrOutputParser()
    )
    wc_lambda = RunnableLambda(word_count)

    pipeline = summary_chain | wc_lambda
    wc = pipeline.invoke({"text": result['joke']})
    print(f"\n📏 总结后的词数: {wc}")


# ═══════════════════════════════════════════════════════════════
# 路由
# ═══════════════════════════════════════════════════════════════

CASES = {
    "prompt":            case_prompt_template,
    "chain":             case_lcel_chain,
    "rag":               case_rag,
    "memory":            case_memory,
    "parser":            case_output_parser,
    "loader":            case_document_loader,
    "splitter":          case_text_splitter,
    "callback":          case_callback,
    "structured_output": case_structured_output,
    "runnable":          case_runnable_composition,
}


def print_usage():
    print("用法: python examples.py <case_name>")
    print("可选 case:")
    for name, fn in CASES.items():
        desc = (fn.__doc__ or "?").strip().split("\n")[0].split("—")[-1].strip()
        print(f"  {name:<20} {desc}")
    print("\n示例: python examples.py rag")


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

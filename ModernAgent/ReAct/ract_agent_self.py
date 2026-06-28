import os
import platform
from pathlib import Path
import inspect
import re

from anthropic import Anthropic
from dotenv import load_dotenv
import ast

from prompt_template import react_system_prompt_template


def _extract_tag(text: str, tag: str) -> str | None:
    """从文本中提取 XML 标签内容，返回 None 如果未找到。"""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else None


# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")


class ReActAgent:
    """ReAct Agent that uses Claude-compatible LLM for reasoning and acting."""

    def __init__(self, tools, project_dir):
        # LLM configuration loaded from environment variables
        self.api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL")
        self.model = os.environ.get("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
        self.opus_model = os.environ.get(
            "ANTHROPIC_DEFAULT_OPUS_MODEL", "deepseek-v4-pro[1m]"
        )
        self.sonnet_model = os.environ.get(
            "ANTHROPIC_DEFAULT_SONNET_MODEL", "deepseek-v4-pro[1m]"
        )
        self.haiku_model = os.environ.get(
            "ANTHROPIC_DEFAULT_HAIKU_MODEL", "deepseek-v4-flash"
        )
        self.subagent_model = os.environ.get(
            "CLAUDE_CODE_SUBAGENT_MODEL", "deepseek-v4-flash"
        )
        self.effort_level = os.environ.get("CLAUDE_CODE_EFFORT_LEVEL", "max")

        # 通过名称字符串从实例方法中绑定工具
        self.tools = {name: getattr(self, name) for name in tools}

        self.project_dir = project_dir

        # Anthropic client
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, messages_params: str, system: str = "You are a helpful assistant."):
        """Send a message to the LLM, print thinking blocks separately, and return the text response."""
        response = self.client.messages.create(
            model=self.model, max_tokens=1000, system=system, messages=messages_params
        )

        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                # 用 ANSI 灰色 + 前缀标记打印思考内容
                thinking_text = getattr(block, "thinking", str(block))
                print("\033[90m┌─ 💭 思考过程 ──────────────────────\033[0m")
                for line in thinking_text.split("\n"):
                    print(f"\033[90m│\033[0m {line}")
                print("\033[90m└────────────────────────────────────\033[0m")
            elif block.type == "redacted_thinking":
                print("\033[90m┌─ 💭 思考过程 [已编辑] ──────────────\033[0m")
                print("\033[90m│ (内容被模型提供方隐藏)\033[0m")
                print("\033[90m└────────────────────────────────────\033[0m")

        return "\n".join(text_parts)

    def parse_action(self, action: str) -> tuple[str, list, dict]:
        """将 '<action> 标签内的字符串解析为 (函数名, 位置参数列表, 关键字参数字典)。

        示例：
            'read_file("/tmp/foo.txt")'       → ('read_file', ['/tmp/foo.txt'], {})
            'write_file("/a.txt", "hello")'    → ('write_file', ['/a.txt', 'hello'], {})
            'find_recipe(dish="番茄炒蛋")'      → ('find_recipe', [], {'dish': '番茄炒蛋'})
        """
        action = action.strip()

        # 抽取函数名：第一个 ( 之前的部分
        paren_idx = action.find("(")
        if paren_idx == -1:
            raise ValueError(f"无法解析 action（缺少括号）: {action!r}")
        func_name = action[:paren_idx].strip()
        args_str = action[paren_idx:].strip()  # "(arg1, arg2, ...)" 包含括号

        # 安全模式：用 ast 解析为函数调用表达式
        # 将 func_name(args_str) 包装为完整表达式后解析
        wrapper = f"{func_name}{args_str}"
        try:
            tree = ast.parse(wrapper, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"action 语法无效: {wrapper!r}") from e

        call = tree.body
        if not isinstance(call, ast.Call):
            raise ValueError(f"action 不是函数调用: {wrapper!r}")

        args = [ast.literal_eval(arg) for arg in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}

        return func_name, args, kwargs

    def run(self, user_message: str):
        """Run the ReAct agent with a user message."""
        system_prompt = self.render_system_prompt()
        messages = [{"role": "user", "content": user_message}]

        for _ in range(20):  # 最多 20 轮防止死循环（用来测试的）
            content = self.chat(messages_params=messages, system=system_prompt)

            # 检测 Final Answer
            if "<final_answer>" in content:
                match = _extract_tag(content, "final_answer")
                return match if match else content

            # 检测 Thought
            thought = _extract_tag(content, "thought")
            if thought:
                print(f"\n💭 {thought}")

            # 检测 Action
            action_str = _extract_tag(content, "action")
            if not action_str:
                raise RuntimeError(f"模型未输出 <action>，原始响应:\n{content}")

            func_name, args, kwargs = self.parse_action(action_str)
            arg_strs = [repr(a) for a in args] + [
                f"{k}={v!r}" for k, v in kwargs.items()
            ]
            print(f"🔧 {func_name}({', '.join(arg_strs)})")
            # 每次执行工具前都询问用户确认
            should_continue = input("\n\n是否继续？（Y/N）")
            if should_continue.lower() != "y":
                print("\n\n操作已取消。")
                return "操作被用户取消"

            # 执行工具
            if func_name not in self.tools:
                observation = f"错误：未找到工具 '{func_name}'，可用工具: {list(self.tools.keys())}"
            else:
                try:
                    observation = self.tools[func_name](*args, **kwargs)
                except Exception as e:
                    observation = f"工具执行错误: {e}"

            print(f"👁️ {observation}")

            # 将本轮交互追加到消息历史
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {"role": "user", "content": f"<observation>{observation}</observation>"}
            )

        raise RuntimeError("达到最大轮次上限")

    def get_operating_system_name(self):
        os_map = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}
        return os_map.get(platform.system(), "Unknown")

    def render_system_prompt(self) -> str:
        """Render the system prompt with the tool list and environment information."""
        # 1. 工具列表（含签名 + 简要说明）
        tool_list = self.get_tool_list()

        # 2. 操作系统
        operating_system = self.get_operating_system_name()

        # 3. 当前目录下文件列表
        files = (
            os.path.abspath(os.path.join(self.project_dir, f))
            for f in os.listdir(self.project_dir)
        )
        file_list = ", ".join(files)

        # 4. 替换模板中的三个占位符
        return (
            react_system_prompt_template.replace("${tool_list}", tool_list)
            .replace("${operating_system}", operating_system)
            .replace("${file_list}", file_list)
        )

    def read_file(self, file_path: str) -> str:
        """读取指定文件的内容并返回。"""
        with open(file_path, "r") as file:
            return file.read()

    def write_file(self, file_path: str, content: str):
        """将内容写入指定文件，若文件已存在则覆盖。"""
        with open(file_path, "w") as file:
            file.write(content)

    def list_files_in_directory(self, directory_path: str) -> list[str]:
        """列出指定目录下的所有文件（不含子目录）。"""
        return [
            f
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]

    def run_terminal_command(self):
        """执行终端命令并返回标准输出，若失败则返回标准错误。"""
        import subprocess

        command = input("Enter a terminal command to run: ")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout if result.returncode == 0 else result.stderr

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)


def main():
    agent = ReActAgent()
    reply = agent.chat("Hi, how are you?")
    print(reply)


if __name__ == "__main__":
    tools = [
        "read_file",
        "write_file",
        "run_terminal_command",
        "list_files_in_directory",
    ]
    project_dir = Path(__file__).parent
    print(f"Project directory: {project_dir}")
    agent = ReActAgent(tools, project_dir)
    agent.run("帮我打开本地的ReAct.md文件")
    # s= agent.render_system_prompt()
    # print(s)

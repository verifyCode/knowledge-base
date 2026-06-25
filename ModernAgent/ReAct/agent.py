import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# Load .env from the same directory as this file
load_dotenv(Path(__file__).parent / ".env")


class ReActAgent:
    """ReAct Agent that uses Claude-compatible LLM for reasoning and acting."""

    def __init__(self):
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

        # Anthropic client
        self.client = Anthropic(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def chat(self, user_message: str, system: str = "You are a helpful assistant."):
        """Send a message to the LLM and return the text response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )

        # Extract text from TextBlock(s), skipping ThinkingBlock(s)
        text_blocks: list[str] = [
            block.text for block in response.content if block.type == "text"
        ]
        return "\n".join(text_blocks)


def main():
    agent = ReActAgent()
    reply = agent.chat("Hi, how are you?")
    print(reply)


if __name__ == "__main__":
    main()

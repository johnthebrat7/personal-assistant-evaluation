# assistant/memory.py
"""Short-term conversational memory: a rolling window of recent turns."""
from typing import List, Dict
from .config import MEMORY_TURNS, SYSTEM_PROMPT


class ConversationMemory:
    def __init__(self, max_turns: int = MEMORY_TURNS):
        self.max_turns = max_turns
        self.messages: List[Dict[str, str]] = []

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def build_context(self) -> List[Dict[str, str]]:
        """System prompt + recent turns, ready to send to a model."""
        return [{"role": "system", "content": SYSTEM_PROMPT}] + self.messages

    def reset(self) -> None:
        self.messages = []

    def _trim(self) -> None:
        limit = self.max_turns * 2  # user + assistant per turn
        if len(self.messages) > limit:
            self.messages = self.messages[-limit:]
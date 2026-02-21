"""Context compression using ConversationSummaryBufferMemory.

Keeps recent messages verbatim up to max_token_limit; older messages are
summarized by an LLM call. This balances context quality with token cost.

Note: ConversationSummaryBufferMemory is marked deprecated in LangChain 0.3.1
but is not removed. We use it deliberately for its token-aware auto-summarization.
Migration to LangGraph-native trim_messages + summarization nodes is planned
for a future version.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage


class ContextCompressor:
    """Wraps ConversationSummaryBufferMemory for automatic context compression."""

    def __init__(self, llm: BaseChatModel, max_token_limit: int = 4000):
        self._llm = llm
        self._max_token_limit = max_token_limit
        self._memory = self._build_memory()

    def _build_memory(self):
        # In langchain 1.x, legacy memory classes moved to langchain_classic
        try:
            from langchain_classic.memory import ConversationSummaryBufferMemory  # noqa: PLC0415
        except ImportError:
            from langchain.memory import ConversationSummaryBufferMemory  # noqa: PLC0415
        return ConversationSummaryBufferMemory(
            llm=self._llm,
            max_token_limit=self._max_token_limit,
            return_messages=True,
            memory_key="chat_history",
        )

    def add_interaction(self, human_input: str, ai_output: str) -> None:
        """Record one turn of conversation for compression tracking."""
        if human_input or ai_output:
            self._memory.save_context(
                {"input": human_input or ""},
                {"output": ai_output or ""},
            )

    def get_messages(self) -> list[BaseMessage]:
        """Return the compressed message history (summary + recent messages)."""
        return self._memory.load_memory_variables({}).get("chat_history", [])

    def get_summary(self) -> str:
        """Return the running summary of older context."""
        return self._memory.moving_summary_buffer or ""

    def clear(self) -> None:
        """Reset the memory."""
        self._memory.clear()

    @property
    def max_token_limit(self) -> int:
        return self._max_token_limit

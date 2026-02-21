"""Router agent — classifies user intent into a task type.

Uses the simple (Haiku) model with a direct prompt chain (no tools).
This is intentionally lightweight: it makes one cheap LLM call to
classify the user's message before dispatching to a specialized agent.
"""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ROUTER_SYSTEM_PROMPT = """You are a task classifier for a coding assistant.
Classify the user's latest message into exactly one category.

Categories:
- code: Writing new code, modifying existing code, debugging, refactoring, code review
- plan: Architecture decisions, project planning, breaking down tasks, design discussions
- search: Looking up documentation, finding libraries, researching technologies, web queries
- ask: General questions, explanations, how-things-work, code understanding, everything else

Rules:
- Respond with ONLY the single lowercase word: code, plan, search, or ask
- When in doubt, use 'ask'
- If the message is about writing or changing code files, use 'code'"""


class RouterAgent:
    """Lightweight classifier — one LLM call, no tools, no retries on classification."""

    def __init__(self, llm: BaseChatModel):
        self._chain = (
            ChatPromptTemplate.from_messages([
                ("system", ROUTER_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
            ])
            | llm
            | StrOutputParser()
        )

    def classify(self, messages: list[BaseMessage]) -> str:
        """Classify the user's intent. Returns one of: code, plan, search, ask."""
        if not messages:
            return "ask"

        # Only send the last 3 messages to keep this cheap
        recent = messages[-3:] if len(messages) > 3 else messages
        try:
            result = self._chain.invoke({"messages": recent})
            task_type = result.strip().lower().split()[0]  # take first word only
            if task_type in ("code", "plan", "search", "ask"):
                return task_type
            return "ask"
        except Exception:
            return "ask"  # Safe default

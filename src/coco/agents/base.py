"""Base agent class for all coco specialized agents.

All agents are thin wrappers around langgraph.prebuilt.create_react_agent.
They share a common pattern:
1. A system prompt that defines the agent's role and capabilities
2. A curated tool list appropriate for the task
3. A [CLARIFY]...[/CLARIFY] convention for requesting human input

The [CLARIFY] protocol:
    When an agent needs user clarification, it includes the tag in its response:
        [CLARIFY]What framework should I use for the API?[/CLARIFY]
    BaseCocoAgent.run() detects this and sets human_feedback_needed=True,
    which triggers the LangGraph human_feedback_node → interrupt() flow.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent


class BaseCocoAgent(ABC):
    """Base class for all coco agents.

    Subclasses must define:
        name: str — unique agent identifier
        system_prompt: str — the agent's system instructions
    """

    name: str = "base"
    system_prompt: str = ""

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        self._llm = llm
        self._tools = tools
        self._agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=self.system_prompt,
        )

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the agent against the current conversation state.

        Returns a dict with:
            messages: list[BaseMessage] — new messages to add to state
            human_feedback_needed: bool — True if agent needs user input
            clarification_question: str — the question (if needed)
        """
        try:
            result = self._agent.invoke({"messages": state["messages"]})
        except Exception as e:
            # Return error as an AI message rather than crashing
            error_msg = AIMessage(
                content=f"I encountered an error: {e}\n\nPlease try rephrasing your request."
            )
            return {
                "messages": [error_msg],
                "human_feedback_needed": False,
                "clarification_question": "",
            }

        messages: list[BaseMessage] = result.get("messages", [])
        human_feedback_needed = False
        clarification_question = ""

        # Check if the last AI message contains a clarification request
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                if "[CLARIFY]" in content:
                    human_feedback_needed = True
                    clarification_question = self._extract_clarification(content)
                break

        return {
            "messages": messages,
            "human_feedback_needed": human_feedback_needed,
            "clarification_question": clarification_question,
        }

    @staticmethod
    def _extract_clarification(content: str) -> str:
        """Extract question from [CLARIFY]...[/CLARIFY] tags."""
        match = re.search(r"\[CLARIFY\](.*?)\[/CLARIFY\]", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return content after [CLARIFY] if closing tag is missing
        match = re.search(r"\[CLARIFY\](.*)", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content

    @abstractmethod
    def get_tool_names(self) -> list[str]:
        """Return the names of tools this agent has access to."""
        ...

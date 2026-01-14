"""Utility helpers for constructing LangChain message histories."""

from __future__ import annotations

from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.models.query import ChatRequest


def build_history_messages(chat_request: ChatRequest) -> List[SystemMessage | HumanMessage | AIMessage]:
    """Convert chat history entries into LangChain message objects."""

    history_messages: List[SystemMessage | HumanMessage | AIMessage] = []

    if not chat_request.chat_history:
        return history_messages

    for item in chat_request.chat_history:
        if item.role == "user":
            history_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            history_messages.append(AIMessage(content=item.content))
        else:
            # Skip non user/assistant roles to avoid stacking system prompts.
            continue

    return history_messages


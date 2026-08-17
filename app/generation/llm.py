"""LLM configuration and factory functions."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def create_groq_llm() -> ChatGroq:
    """Create a configured Groq chat model from environment variables."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("GROQ_MODEL", "").strip()

    if not api_key:
        raise ValueError("GROQ_API_KEY is required")

    if not model:
        raise ValueError("GROQ_MODEL is required")

    return ChatGroq(
        model=model,
        api_key=api_key,
    )
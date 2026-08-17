"""Base interface for all knowledge ingestion loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document


class BaseLoader(ABC):
    """Common interface for every knowledge source loader."""

    @abstractmethod
    def load(self, source: str | Path) -> list[Document]:
        """Load a source and return normalized LangChain documents."""
        raise NotImplementedError
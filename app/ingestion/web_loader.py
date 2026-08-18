"""Load web-based knowledge sources into LangChain Documents."""

from __future__ import annotations

from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

from app.ingestion.base import BaseLoader


class URLLoader(BaseLoader):
    """Load web pages while preserving source metadata."""

    def load(self, source: str) -> list[Document]:
        """Load a URL and return normalized LangChain Documents."""
        if not source.startswith(("http://", "https://")):
            raise ValueError(f"Expected an HTTP/HTTPS URL, got: {source}")

        parsed_url = urlparse(source)

        if not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {source}")

        loader = WebBaseLoader(web_paths=[source])
        documents = loader.load()

        for document in documents:
            document.metadata.update(
                {
                    "source_id": source,
                    "source_type": "web",
                    "source_name": parsed_url.netloc,
                    "location": source,
                }
            )

        return documents


class WikipediaBackend:
    """Small wrapper around the Wikipedia Python package."""

    def load(self, query: str) -> list[Document]:
        """Fetch a Wikipedia page and return it as a LangChain Document."""
        import wikipedia

        page = wikipedia.page(query, auto_suggest=False)

        return [
            Document(
                page_content=page.content,
                metadata={
                    "title": page.title,
                    "url": page.url,
                },
            )
        ]


class WikipediaLoader(BaseLoader):
    """Load Wikipedia pages while preserving source metadata."""

    def load(self, source: str) -> list[Document]:
        """Load a Wikipedia page by title/query."""
        if not source.strip():
            raise ValueError("Wikipedia query cannot be empty")

        backend = WikipediaBackend()
        documents = backend.load(source)

        for document in documents:
            document.metadata.update(
                {
                    "source_id": f"wikipedia:{source}",
                    "source_type": "wikipedia",
                    "source_name": source,
                    "location": "Wikipedia",
                }
            )

        return documents
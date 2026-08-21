"""Knowledge-base version stamp.

Every cached retrieval / answer is bound to the current version of the
knowledge base. When new knowledge is ingested the version is bumped, which
changes every cache key and therefore guarantees that stale answers computed
against an older knowledge base are never served.

The stamp is a small counter file stored next to the vector store so it
survives restarts and works even when Redis is unavailable.
"""

from __future__ import annotations

from pathlib import Path

from app.config import settings

_FILENAME = "kb_version.txt"


class KnowledgeVersion:
    """Persisted monotonic counter describing the knowledge-base state."""

    def __init__(self, persist_dir: str | None = None) -> None:
        self._dir = Path(persist_dir or settings.CHROMA_PERSIST_DIR)
        self._path = self._dir / _FILENAME

    def current(self) -> int:
        """Return the current knowledge-base version (0 if never ingested)."""
        try:
            return int(self._path.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            return 0

    def bump(self) -> int:
        """Increment and persist the version. Returns the new value."""
        version = self.current() + 1
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._path.write_text(str(version), encoding="utf-8")
        except OSError:
            # Non-fatal: caching simply keeps using the previous stamp.
            return self.current()
        return version

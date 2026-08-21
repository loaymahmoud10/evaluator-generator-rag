"""Load audio sources and transcribe them into LangChain Documents."""

from __future__ import annotations

import os
from pathlib import Path

from groq import Groq
from langchain_core.documents import Document

from app.config import settings
from app.ingestion.base import BaseLoader


class AudioLoader(BaseLoader):
    """Transcribe WAV files using Groq Whisper."""

    def load(self, source: str | Path) -> list[Document]:
        """Transcribe a WAV file and return it as a LangChain Document."""
        path = Path(source)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a WAV file, got: {path.suffix}")

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")

        client = Groq(api_key=api_key)

        with path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=settings.WHISPER_MODEL or "whisper-large-v3-turbo",
                response_format="json",
                temperature=0.0,
            )

        document = Document(
            page_content=transcription.text,
            metadata={
                "source_id": path.resolve().as_posix(),
                "source_type": "wav",
                "source_name": path.name,
                "location": "audio",
            },
        )

        return [document]
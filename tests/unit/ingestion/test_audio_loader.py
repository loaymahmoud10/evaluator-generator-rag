from pathlib import Path
from unittest.mock import MagicMock, patch

from app.ingestion.audio_loader import AudioLoader


SAMPLE_WAV = Path("tests/data/sample.wav")


def test_audio_loader_transcribes_wav_and_tracks_source():
    fake_transcription = MagicMock()
    fake_transcription.text = (
        "Artificial intelligence is a field of computer science."
    )

    with (
        patch("app.ingestion.audio_loader.Groq") as mock_groq_class,
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
    ):
        mock_client = mock_groq_class.return_value
        mock_client.audio.transcriptions.create.return_value = fake_transcription

        loader = AudioLoader()
        documents = loader.load(SAMPLE_WAV)

    assert len(documents) == 1

    document = documents[0]

    assert (
        "Artificial intelligence is a field of computer science."
        in document.page_content
    )

    assert document.metadata["source_type"] == "wav"
    assert document.metadata["source_name"] == "sample.wav"
    assert document.metadata["location"] == "audio"
    assert document.metadata["source_id"]

    mock_groq_class.assert_called_once_with(api_key="test-key")
    mock_client.audio.transcriptions.create.assert_called_once()
from unittest.mock import patch

import pytest

from app.generation.llm import create_groq_llm


def test_create_groq_llm_uses_environment_configuration():
    with patch(
        "app.generation.llm.ChatGroq"
    ) as mock_chat_groq:
        with patch.dict(
            "os.environ",
            {
                "GROQ_API_KEY": "test-api-key",
                "GROQ_MODEL": "test-model",
            },
            clear=False,
        ):
            create_groq_llm()

    mock_chat_groq.assert_called_once_with(
        model="test-model",
        api_key="test-api-key",
    )


def test_create_groq_llm_requires_api_key():
    with patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "",
            "GROQ_MODEL": "test-model",
        },
        clear=False,
    ):
        with pytest.raises(
            ValueError,
            match="GROQ_API_KEY is required",
        ):
            create_groq_llm()
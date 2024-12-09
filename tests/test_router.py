from src.router import AiRouter
from src.enum import ModelEnum
from src.errors import MissingMessagesError, StreamingNotSupportedInPrivacyModeError

import pytest
from numpy import random
from unittest.mock import patch


@pytest.fixture
def ai_router():
    return AiRouter(api_key="test_api_key", base_url="https://test_base_url.com")


def test_chat_standard_request(ai_router):
    messages = [{"role": "user", "content": "Hello"}]
    stream = False
    model = random.choice(list(ModelEnum))
    other_model = [other_model for other_model in ModelEnum if model != other_model]
    models = [model, other_model]

    with patch.object(
        ai_router.ai_router_api, "standard_request", return_value="response"
    ) as mock_standard_request:
        result = ai_router.chat(
            messages=messages, stream=stream, models=models, privacy_mode=False
        )
        mock_standard_request.assert_called_once_with(
            messages=messages, stream=stream, models=models
        )
        assert result == "response"


def test_chat_privacy_mode_request(ai_router):
    messages = [{"role": "user", "content": "Hello"}]
    stream = False
    model = random.choice(list(ModelEnum))
    other_model = [other_model for other_model in ModelEnum if model != other_model]
    models = [model, other_model]

    with (
        patch.object(
            ai_router, "_generate_embeddings", return_value=[1, 2, 3]
        ) as mock_generate_embeddings,
        patch.object(
            ai_router.ai_router_api,
            "privacy_mode_request",
            return_value=model,
        ) as mock_privacy_mode_request,
    ):
        result = ai_router.chat(
            messages=messages, stream=stream, models=models, privacy_mode=True
        )
        mock_generate_embeddings.assert_called_once_with(messages)
        mock_privacy_mode_request.assert_called_once_with(embeddings=[1, 2, 3])
        assert result == model


def test_chat_privacy_mode_with_streaming(ai_router):
    messages = [{"role": "user", "content": "Hello"}]
    stream = True
    model = random.choice(list(ModelEnum))
    other_model = [other_model for other_model in ModelEnum if model != other_model]
    models = [model, other_model]

    with pytest.raises(StreamingNotSupportedInPrivacyModeError):
        ai_router.chat(
            messages=messages, stream=stream, models=models, privacy_mode=True
        )


def test_chat_missing_messages(ai_router):
    messages = []

    with pytest.raises(MissingMessagesError):
        ai_router.chat(messages=messages)


def test_generate_embeddings(ai_router):
    messages = [{"role": "user", "content": "Hello"}]

    with patch("src.router.TextEmbedding") as MockTextEmbedding:
        mock_embedder = MockTextEmbedding.return_value
        mock_embedder.query_embed.return_value.__next__.return_value.tolist.return_value = [
            1,
            2,
            3,
        ]

        embeddings = ai_router._generate_embeddings(messages)
        assert embeddings == [1, 2, 3]


#
#
def test_execute_proxy_request_not_implemented(ai_router):
    messages = [{"role": "user", "content": "Hello"}]
    model = random.choice(list(ModelEnum))
    stream = False

    with pytest.raises(NotImplementedError):
        ai_router._execute_proxy_request(model=model, messages=messages, stream=stream)


@pytest.mark.integration
def test_router():
    api_key = "sk12345"

    router = AiRouter(api_key=api_key, base_url="http://localhost:4000")

    result = router.chat(
        privacy_mode=True,
        messages=[{"role": "system", "content": "What is the meaning of life?"}],
    )

    print(result)

    assert True

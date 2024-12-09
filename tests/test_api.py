import pytest
from numpy import random
from unittest.mock import patch, MagicMock

from src.api import AiRouterApi
from src.enum import ModelEnum
from src.errors import MissingAPIKeyError
from src.types import AiRouterModelWeighting


@pytest.fixture
def mock_openai_client():
    with patch("src.api.OpenAI") as MockOpenAI:
        yield MockOpenAI


@pytest.fixture
def ai_router_api(mock_openai_client):
    return AiRouterApi(api_key="test_api_key", base_url="https://test_base_url.com")


def test_get_api_key_from_env():
    with patch("src.api.getenv", return_value=None):
        with pytest.raises(MissingAPIKeyError):
            AiRouterApi.get_api_key_from_env()

    with patch("src.api.getenv", return_value="test_api_key"):
        assert AiRouterApi.get_api_key_from_env() == "test_api_key"


def test_privacy_mode_request(ai_router_api, mock_openai_client):
    mock_response = MagicMock()
    model = random.choice(list(ModelEnum))
    other_model = [other_model for other_model in ModelEnum if model != other_model]
    mock_response.choices[0].message.content = model
    ai_router_api.ai_router_client.chat.completions.create.return_value = mock_response

    embeddings = [1, 2, 3]
    models = [model, other_model]
    weighting = AiRouterModelWeighting()

    result = ai_router_api.privacy_mode_request(embeddings, models, weighting)
    assert result == model


#
#
def test_standard_request(ai_router_api, mock_openai_client):
    mock_response = MagicMock()
    ai_router_api.ai_router_client.chat.completions.create.return_value = mock_response
    model = random.choice(list(ModelEnum))
    other_model = [other_model for other_model in ModelEnum if model != other_model]

    messages = [{"role": "user", "content": "Hello"}]
    stream = False
    models = [model, other_model]
    weighting = AiRouterModelWeighting()

    result = ai_router_api.standard_request(messages, stream, models, weighting)
    assert result == mock_response


def test_make_chat_request(ai_router_api, mock_openai_client):
    mock_response = MagicMock()
    ai_router_api.ai_router_client.chat.completions.create.return_value = mock_response

    model = random.choice(list(ModelEnum))
    other_model = [other_model for other_model in ModelEnum if model != other_model]

    messages = [{"role": "user", "content": "Hello"}]
    stream = False
    models = [model, other_model]
    weighting = AiRouterModelWeighting()
    privacy_mode = True
    embeddings = [1, 2, 3]

    result = ai_router_api._make_chat_request(
        messages=messages,
        stream=stream,
        models=models,
        weighting=weighting,
        privacy_mode=privacy_mode,
        embeddings=embeddings,
    )
    assert result == mock_response

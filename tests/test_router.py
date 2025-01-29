import os
import pytest
from airouter import AiRouter, EmbeddingType
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def ai_router():
    os.environ["AIROUTER_HOST"] = "http://localhost:4000"
    return AiRouter()


@pytest.mark.integration
def test_default_model_routing(ai_router):
    response = ai_router.chat.completions.create(
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    assert response is not None


@pytest.mark.integration
def test_model_selection_mode(ai_router):
    best_model = ai_router.get_best_model(
        messages=[{"role": "user", "content": "Hello, world!"}],
    )
    assert best_model is not None


@pytest.mark.integration
def test_full_privacy_mode(ai_router):
    best_model = ai_router.get_best_model(
        messages=[{"role": "user", "content": "Hello, world!"}],
        full_privacy=True,
    )
    assert best_model is not None


@pytest.mark.integration
def test_full_privacy_mode_with_existing_embedding(ai_router):
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="Hello, world!",
        dimensions=1536
    )
    embedding = embedding.data[0].embedding
    best_model = ai_router.get_best_model(
        full_privacy=True,
        embedding=embedding,
        embedding_type=EmbeddingType.TEXT_EMBEDDING_3_SMALL,
    )
    assert best_model is not None

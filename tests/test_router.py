import os
import pytest
from airouter import AiRouter


@pytest.fixture
def ai_router():
    os.environ["AIROUTER_HOST"] = "http://localhost:4000"
    return AiRouter(api_key="sk-khGHG1HlJjoNmgq_a8DVNw")


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

from src.config import AiRouterConfig


def test_embedder_model():
    assert (
        AiRouterConfig.embedder_model()
        == "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )


def test_base_url_prod():
    assert AiRouterConfig.base_url_prod() == "https://api.airouter.io"


def test_supported_models():
    expected_models = [
        "router/gpt-4o-2024-05-13",
        "router/gpt-4o-mini-2024-07-18",
        "gpt-4-0125-preview",
        "mistralai/open-mixtral-8x22b-2404",
        "mistral-large-2407",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama-3.1-8b",
        "llama-3.1-70b",
        "llama-3.1-405b",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "command-r-plus",
        "gemini-1.5-flash-001",
        "gemini-1.5-pro-001",
        "qwen2.5-72B",
    ]
    assert AiRouterConfig.supported_models() == expected_models

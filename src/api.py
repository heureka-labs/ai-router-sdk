from os import getenv
from typing import Iterable, List

from src.config import AiRouterConfig
from src.enum import ModelEnum
from src.errors import MissingAPIKeyError
from src.types import AiRouterModelWeighting

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class AiRouterApi:
    def __init__(self, api_key: str = None, base_url: str = None):
        api_key = api_key if api_key else AiRouterApi.get_api_key_from_env()
        base_url = base_url if base_url else AiRouterConfig.base_url_prod()
        self.ai_router_client = OpenAI(api_key=api_key, base_url=base_url)

    def privacy_mode_request(
        self,
        embeddings: list[int],
        models: List[ModelEnum] = None,
        weighting: AiRouterModelWeighting = None,
    ) -> ModelEnum:
        response = self._make_chat_request(
            stream=False,
            models=models,
            weighting=weighting,
            privacy_mode=True,
            embeddings=embeddings,
        )
        model = response.choices[0].message.content

        return model

    def standard_request(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool,
        models: list[ModelEnum],
        weighting: AiRouterModelWeighting,
        **kwargs,
    ):
        return self._make_chat_request(
            messages=messages,
            stream=stream,
            models=models,
            weighting=weighting,
            privacy_mode=False,
            embeddings=None,
            **kwargs,
        )

    def _make_chat_request(
        self,
        messages: Iterable[ChatCompletionMessageParam] | None = None,
        stream: bool = False,
        models: list[ModelEnum] = None,
        weighting: AiRouterModelWeighting = None,
        privacy_mode: bool = False,
        embeddings: list[int] = None,
        **kwargs,
    ):
        extra_body = {}
        if privacy_mode:
            extra_body["privacy"] = privacy_mode
            extra_body["embeddings"] = embeddings

        if models:
            extra_body["models"] = [model for model in models]

        if weighting:
            extra_body["weighting"] = weighting

        return self.ai_router_client.chat.completions.create(
            model="router",
            messages=messages,
            stream=stream,
            extra_body=extra_body,
            **kwargs,
        )

    @staticmethod
    def get_api_key_from_env():
        api_key = getenv("AI_ROUTER_API_KEY", None)
        if api_key is None:
            raise MissingAPIKeyError()

        return api_key

    @staticmethod
    def get_base_url_from_env():
        getenv("AI_ROUTER_BASE_URL", AiRouterConfig.base_url_prod())

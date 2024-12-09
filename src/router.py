from typing import Iterable

from src.api import AiRouterApi
from src.config import AiRouterConfig
from src.enum import ModelEnum
from src.errors import (
    MissingMessagesError,
    StreamingNotSupportedInPrivacyModeError,
)

from openai.types.chat import ChatCompletionMessageParam
from fastembed import TextEmbedding


class AiRouter:
    embedder = None

    def __init__(self, api_key: str = None, base_url: str = None):
        self.ai_router_api = AiRouterApi(api_key=api_key, base_url=base_url)

    def chat(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool = False,
        models: list[ModelEnum] = None,
        privacy_mode: bool = False,
        embeddings: list[int] = None,
        exceute_request: bool = False,
        **kwargs,
    ):
        self._validate_messages(messages)

        if not privacy_mode:
            return self.ai_router_api.standard_request(
                messages=messages, stream=stream, models=models, **kwargs
            )
        else:
            self._verify_streaming_off(stream)

            embeddings = self._generate_embeddings(messages)
            model = self.ai_router_api.privacy_mode_request(embeddings=embeddings)

            if exceute_request:
                return self._execute_proxy_request(
                    model=model, messages=messages, stream=stream, **kwargs
                )
            else:
                return model

    def _validate_messages(self, messages):
        if messages is None or len(messages) == 0:
            raise MissingMessagesError()

    def _verify_streaming_off(self, stream):
        if stream:
            raise StreamingNotSupportedInPrivacyModeError()

    def _generate_embeddings(self, messages: Iterable[ChatCompletionMessageParam]):
        if not self.embedder:
            self.embedder = TextEmbedding(model_name=AiRouterConfig.embedder_model())

        query = " ".join(message["content"] for message in messages)
        return self.embedder.query_embed(query).__next__().tolist()

    def _execute_proxy_request(
        self,
        model: ModelEnum,
        messages: Iterable[ChatCompletionMessageParam],
        stream: bool,
        **kwargs,
    ):
        raise NotImplementedError(
            "Request excecution in privacy mode is not yet implemented. You will have to make the request yourself for now."
        )

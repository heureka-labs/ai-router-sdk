import os
from typing import Iterable
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from airouter.chat import AiRouterChat
from airouter.types import EmbeddingType, Model

AI_ROUTER_API_KEY_ENV_VAR_NAME = "AIROUTER_API_KEY"
AI_ROUTER_BASE_URL_ENV_VAR_NAME = "AIROUTER_HOST"
AI_ROUTER_BASE_URL = "https://api.airouter.io"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class AiRouter(OpenAI):
    chat: AiRouterChat

    def __init__(self, **kwargs):
        # adjust the base url to the airouter
        kwargs["base_url"] = os.getenv(
            AI_ROUTER_BASE_URL_ENV_VAR_NAME, AI_ROUTER_BASE_URL
        )
        # set the airouter api key
        if "api_key" not in kwargs:
            kwargs["api_key"] = os.getenv(AI_ROUTER_API_KEY_ENV_VAR_NAME)
        if kwargs["api_key"] is None:
            raise OpenAIError(
                f"The airouter api key must be set either by passing api_key to the client or by setting the {AI_ROUTER_API_KEY_ENV_VAR_NAME} environment variable"
            )

        super().__init__(**kwargs)

        self.chat = AiRouterChat(self)
        self.embedder = None

    def get_best_model(
        self,
        messages: Iterable[ChatCompletionMessageParam] = None,
        full_privacy: bool = False,
        embedding: list[float] = None,
        embedding_type: EmbeddingType = None,
        **kwargs,
    ) -> Model:
        """
        Get the best model for the given messages.

        :param messages: The messages to identify the best model for.
        :param full_privacy: Whether to use full privacy mode where an embedding is used instead of messages.
        :param embedding: The embedding to use.
        :param embedding_type: The embedding type of the embedding.
        :param kwargs: Additional arguments for the generation.
        :return: The best model for the given messages.
        """
        self._validate_inputs(messages, embedding, embedding_type)

        # explicitly set full privacy if an embedding is provided
        if embedding:
            full_privacy = True

        # deactivate model routing to directly receive the best model
        kwargs["extra_body"] = {
            "model_routing": False,
            **(kwargs.get("extra_body", {})),
        }

        # if no explicit default model is set, use the model routing default
        kwargs["model"] = kwargs.get("model", "auto")

        if full_privacy:
            if not embedding:
                # generate embedding locally to avoid sending the messages to the airouter
                embedding = self._generate_embedding(messages)
                embedding_type = EmbeddingType.PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2

            kwargs["extra_body"] = {
                **(kwargs.get("extra_body", {})),
                "embedding": embedding,
                "embedding_type": embedding_type.value,
            }
            messages = None

        response = self.chat.completions.create(
            messages=messages,
            **kwargs,
        )

        return Model.from_string(response.choices[0].message.content)

    def _validate_inputs(
        self,
        messages: Iterable[ChatCompletionMessageParam] | None,
        embedding: list[float] | None,
        embedding_type: EmbeddingType | None,
    ) -> None:
        """
        Validate the inputs for model selection.

        :param messages: The messages to validate.
        :param embedding: The embedding to validate.
        :param embedding_type: The embedding type to validate.
        :raises ValueError: If the validation fails.
        """
        if (messages is None and embedding is None) or (messages is not None and embedding is not None):
            raise ValueError("Either messages or embedding must be provided, but not both")

        if embedding is not None and embedding_type is None:
            raise ValueError("embedding_type must be provided when using explicit embedding")

    def _generate_embedding(
        self, messages: Iterable[ChatCompletionMessageParam]
    ) -> list[float]:
        # ensure privacy extra is installed
        try:
            if not self.embedder:
                from fastembed import TextEmbedding

                self.embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)

            query = " ".join(message["content"] for message in messages)
            return self.embedder.query_embed(query).__next__().tolist()
        except ImportError:
            raise ImportError(
                "The fully privacy mode requires the 'privacy' extra. Install with: pip install airouter-sdk[privacy]."
            )

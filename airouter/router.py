import os
from typing import Iterable
from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from airouter.chat import AiRouterChat
from airouter.types import Model

AI_ROUTER_API_KEY_ENV_VAR_NAME = "AIROUTER_API_KEY"
AI_ROUTER_BASE_URL_ENV_VAR_NAME = "AIROUTER_HOST"
AI_ROUTER_BASE_URL = "https://api.airouter.io"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class AiRouter(OpenAI):
    def __init__(self, **kwargs):
        # adjust the base url to the airouter
        kwargs['base_url'] = os.getenv(AI_ROUTER_BASE_URL_ENV_VAR_NAME, AI_ROUTER_BASE_URL)
        # set the airouter api key
        if 'api_key' not in kwargs:
            kwargs['api_key'] = os.getenv(AI_ROUTER_API_KEY_ENV_VAR_NAME)
        if kwargs['api_key'] is None:
            raise OpenAIError(
                f"The airouter api key must be set either by passing api_key to the client or by setting the {AI_ROUTER_API_KEY_ENV_VAR_NAME} environment variable"
            )

        super().__init__(**kwargs)

        self.chat = AiRouterChat(self)
        self.embedder = None

    def get_best_model(self, messages: Iterable[ChatCompletionMessageParam], full_privacy: bool = False, **kwargs) -> Model:
        # deactivate model routing to directly receive the best model
        kwargs['extra_body'] = {'model_routing': False, **(kwargs.get('extra_body', {}))}

        # if no explicit default model is set, use the model routing default
        kwargs['model'] = kwargs.get('model', 'auto')

        if full_privacy:
            # generate embeddings locally to avoid sending the messages to the airouter
            embeddings = self._generate_embeddings(messages)

            response = self.chat.completions.create(
                messages=None,
                **kwargs,
                extra_body={
                    'embedding': embeddings
                }
            )
        else:
            response = self.chat.completions.create(
                messages=messages,
                **kwargs,
            )

        return Model.get(response.choices[0].message.content)

    def _generate_embeddings(self, messages: Iterable[ChatCompletionMessageParam]) -> list[float]:
        # ensure privacy extra is installed
        try:
            from fastembed import TextEmbedding
            if not self.embedder:
                self.embedder = TextEmbedding(
                    model_name=EMBEDDING_MODEL_NAME
                )

            query = ' '.join(message['content'] for message in messages)
            return self.embedder.query_embed(query).__next__().tolist()
        except ImportError:
            raise ImportError("The fully privacy mode requires the 'privacy' extra. Install with: pip install airouter-sdk[privacy].")


if __name__ == "__main__":
    ai_router = AiRouter(api_key="sk12345")
    print(
        'Default model routing with the message "Hello, world!"...',
        ai_router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, world!"}],
        ),
    )

    print(
        'Model selection mode returning the best model for the message "Hello, world!"...',
        ai_router.get_best_model(
            messages=[{"role": "user", "content": "Hello, world!"}],
        ),
    )

    print(
        'Full privacy mode returning the best model for the message "Hello, world!" without exposing the message...',
        ai_router.get_best_model(
            messages=[{"role": "user", "content": "Hello, world!"}],
            full_privacy=True,
        ),
    )
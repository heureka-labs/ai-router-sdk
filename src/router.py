import os
import re
from typing import Iterable
from openai import DEFAULT_MAX_RETRIES, BadRequestError, OpenAI, OpenAIError
from openai.resources.chat import Chat, Completions
from fastembed import TextEmbedding
from openai.types.chat import ChatCompletionMessageParam

AI_ROUTER_API_KEY_ENV_VAR_NAME = "AI_ROUTER_API_KEY"
AI_ROUTER_BASE_URL_ENV_VAR_NAME = "AI_ROUTER_BASE_URL"
AI_ROUTER_BASE_URL_DEV = "http://localhost:4000"
AI_ROUTER_BASE_URL_PROD = "https://api.airouter.io"


class AiRouterCompletions(Completions):
    embedder: TextEmbedding | None

    def __init__(self, client):
        super().__init__(client)
        self.embedder = None

    def create(
        self,
        messages: Iterable[ChatCompletionMessageParam],
        full_privacy_mode: bool = False,
        model_selection_mode: bool = False,
        **kwargs,
    ) -> str:
        if full_privacy_mode:
            return self._run_full_privacy_mode(messages, **kwargs)
        elif model_selection_mode:
            return self._run_model_selection_mode(messages, **kwargs)
        else:
            return super().create(model="router", messages=messages, **kwargs)

    def _run_full_privacy_mode(
        self, messages: Iterable[ChatCompletionMessageParam], **kwargs
    ):
        embeddings = self._generate_embeddings(messages)
        try:
            response = super().create(
                model="router",
                messages=[],
                stream=False,
                extra_body={"privacy": True, "embedding": embeddings},
                **kwargs,
            )
        except BadRequestError as e:
            selected_model_response = e.message
            return self._extract_model_name(selected_model_response)
        if not selected_model_response:
            selected_model = response.choices[0].message.content

        return selected_model

    def _generate_embeddings(self, messages):
        if not self.embedder:
            self.embedder = TextEmbedding(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            )
        query = " ".join(message["content"] for message in messages)
        return self.embedder.query_embed(query).__next__().tolist()

    def _extract_model_name(self, response):
        pattern = r"\['(.*?)'\]"
        match = re.search(pattern, response)
        selected_model = match.group(1)
        return selected_model

    def _run_model_selection_mode(self, messages, **kwargs):
        pass


class AiRouterChat(Chat):
    @property
    def completions(self):
        return AiRouterCompletions(self._client)


class AiRouter(OpenAI):
    chat: AiRouterChat

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        if api_key is None:
            api_key = os.environ.get(AI_ROUTER_API_KEY_ENV_VAR_NAME)
        if api_key is None:
            raise OpenAIError(
                f"The api_key client option must be set either by passing api_key to the client or by setting the {AI_ROUTER_API_KEY_ENV_VAR_NAME} environment variable"
            )
        self.api_key = api_key

        if base_url is None:
            base_url = os.environ.get(AI_ROUTER_BASE_URL_ENV_VAR_NAME)
        if base_url is None:
            # base_url = AI_ROUTER_BASE_URL_PROD
            base_url = AI_ROUTER_BASE_URL_DEV

        super().__init__(api_key=api_key, base_url=base_url, max_retries=max_retries)

        self.chat = AiRouterChat(self)


if __name__ == "__main__":
    ai_router = AiRouter(api_key="sk12345")
    print(
        'running full privacy mode with message "Hello, world!"...',
        ai_router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, world!"}],
            full_privacy_mode=True,
        ),
    )

    print(
        'running model selection mode with message "Hello, world!"...',
        ai_router.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, world!"}],
        ),
    )

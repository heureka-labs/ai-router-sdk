import os

from openai import DEFAULT_MAX_RETRIES, OpenAI, OpenAIError

from chat import AiRouterChat

AI_ROUTER_API_KEY_ENV_VAR_NAME = "AI_ROUTER_API_KEY"
AI_ROUTER_BASE_URL_ENV_VAR_NAME = "AI_ROUTER_BASE_URL"
AI_ROUTER_BASE_URL_DEV = "http://localhost:4000"
AI_ROUTER_BASE_URL_PROD = "https://api.airouter.io"


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

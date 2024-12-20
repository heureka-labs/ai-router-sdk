from openai.resources.chat import Chat

from airouter.completions import AiRouterCompletions


class AiRouterChat(Chat):
    @property
    def completions(self) -> AiRouterCompletions:
        return AiRouterCompletions(self._client)

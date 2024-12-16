from openai.resources.chat import Chat

from completions import AiRouterCompletions


class AiRouterChat(Chat):
    @property
    def completions(self):
        return AiRouterCompletions(self._client)

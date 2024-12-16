import re
from typing import Iterable

from fastembed import TextEmbedding
from openai import BadRequestError
from openai.resources.chat import Completions
from openai.types.chat import ChatCompletionMessageParam


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
        raise NotImplementedError("Model selection mode is not yet implemented")

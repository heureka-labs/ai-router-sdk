from typing import Union
from openai.resources.chat import Completions

from airouter.types import Model


class AiRouterCompletions(Completions):
    def create(
        self,
        model: Union[str, Model] = 'auto',
        models: list[Union[str, Model]] = [],
        weighting: dict[str, float] = {},
        **kwargs,
    ) -> str:
        """
        Create a completion for the given messages and parameters.

        :param models: The models to use for the completion.
        :param weighting: The weighting of quality, costs and latency. Boost a weighting with numbers greater than 1.0, penalize with numbers less than 1.0.
        """
        # convert model instances to strings
        if isinstance(model, Model):
            model = model.to_string()
        models = [model.to_string() if isinstance(model, Model) else model for model in models]

        # add the models and weighting to the extra body
        kwargs['extra_body'] = {
            **(kwargs.get('extra_body', {})),
            'models': models,
            'weighting': weighting,
        }

        return super().create(model=model, **kwargs)

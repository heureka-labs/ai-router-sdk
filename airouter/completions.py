from openai.resources.chat import Completions


class AiRouterCompletions(Completions):
    def create(
        self,
        models: list[str] = [],
        weighting: dict[str, float] = {},
        **kwargs,
    ) -> str:
        """
        Create a completion for the given messages and parameters.

        :param models: The models to use for the completion.
        :param weighting: The weighting of quality, costs and latency. Boost a weighting with numbers greater than 1.0, penalize with numbers less than 1.0.
        """
        # if no explicit default model is set, use the model routing default
        kwargs['model'] = kwargs.get('model', 'auto')

        # add the models and weighting to the extra body
        kwargs['extra_body'] = {
            **(kwargs.get('extra_body', {})),
            'models': models,
            'weighting': weighting,
        }

        return super().create(**kwargs)

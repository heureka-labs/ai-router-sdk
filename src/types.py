from typing import Literal


AiRouterModelWeighting = dict[
    Literal["quality"] | Literal["costs"] | Literal["latency"], float
]

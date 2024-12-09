from enum import Enum

from src.config import AiRouterConfig

ModelEnum = Enum(
    "ModelEnum", {model: model for model in list(AiRouterConfig.supported_models())}
)

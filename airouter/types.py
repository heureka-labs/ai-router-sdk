from enum import Enum
from typing import Dict, ClassVar


class Model(Enum):
    GPT_4O = 'gpt-4o-2024-08-06'
    GPT_4O_MINI = 'gpt-4o-mini-2024-07-18'
    MISTRAL_LARGE = 'mistral-large-2407'
    LLAMA_31_8B = 'llama-3.1-8b'
    LLAMA_31_70B = 'llama-3.1-70b'
    LLAMA_31_405B = 'llama-3.1-405b'
    CLAUDE_3_HAIKU = 'claude-3-haiku-20240307'
    CLAUDE_3_OPUS = 'claude-3-opus-20240229'
    CLAUDE_35_SONNET = 'claude-3-5-sonnet-20241022'
    COMMAND_R_PLUS = 'command-r-plus'
    GEMINI_15_FLASH = 'gemini-1.5-flash-001'
    GEMINI_15_PRO = 'gemini-1.5-pro-001'
    QWEN25_72B = 'qwen2.5-72B'

    _value_map: ClassVar[Dict[str, 'Model']] = {}

    @classmethod
    def get(cls, value: str) -> 'Model':
        """Convert a model string value to its corresponding Model enum."""
        if not cls._value_map:
            cls._value_map = {model.value: model for model in cls}
        try:
            return cls._value_map[value]
        except KeyError:
            raise ValueError(f'Unknown model: {value}')

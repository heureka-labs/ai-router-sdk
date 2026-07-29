from enum import Enum


class Model(Enum):
    GPT_5_2 = "gpt-5.2"
    GPT_5_1 = "gpt-5.1"
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_OSS_120B = "gpt-oss-120b"
    MISTRAL_LARGE = "mistral-large"
    MISTRAL_SMALL = "mistral-small"
    CLAUDE_4_1_OPUS = "claude-4-1-opus"
    CLAUDE_4_5_SONNET = "claude-4-5-sonnet"
    CLAUDE_4_5_HAIKU = "claude-4-5-haiku"
    GEMINI_31_FLASH_LITE = "gemini-3.1-flash-lite"
    GEMINI_35_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_35_FLASH = "gemini-3.5-flash"
    GEMINI_36_FLASH = "gemini-3.6-flash"
    GEMINI_31_PRO_PREVIEW = "gemini-3.1-pro-preview"
    GEMINI_25_PRO = "gemini-2.5-pro"
    LLAMA_33_70B = "llama-3.3-70b"
    LLAMA_33_70B_FAST = "llama-3.3-70b-fast"
    QWEN3_235B_A22B = "qwen-3-235B-A22B"
    QWEN3_30B_A3B = "qwen-3-30B-A3B"
    PHI_4 = "phi-4"
    UNKNOWN = "unknown"

    def __init__(self, value):
        super().__init__()
        self._original_value = None

    @classmethod
    def _get_value_map(cls):
        # Create value map only if needed and cache it
        if not hasattr(cls, "_value_map"):
            cls._value_map = {m.value: m for m in cls.__members__.values()}
        return cls._value_map

    @classmethod
    def from_string(cls, value: str) -> "Model":
        value_map = cls._get_value_map()
        model = value_map.get(value, cls.UNKNOWN)
        if model == cls.UNKNOWN:
            # Create a new instance for unknown models to store the original value
            model = cls.UNKNOWN
            model._original_value = value
        return model

    def to_string(self):
        if self._original_value is not None:
            return self._original_value
        return self.value


class EmbeddingType(Enum):
    TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
    PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2 = "paraphrase-multilingual-mpnet-base-v2"

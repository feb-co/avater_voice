"""LLaMA ASR model configuration"""
import os

from avatar_infer.models.voice import AvatarVoiceConfig


class LlamaVoiceConfig(AvatarVoiceConfig):
    model_type = "llama voice"

    def __init__(
        self,
        **kwargs,
    ):
        if "llm_path" in kwargs:
            del kwargs["llm_path"]
        if "whisper_path" in kwargs:
            del kwargs["whisper_path"]
        if "wavlm_path" in kwargs:
            del kwargs["wavlm_path"]

        super().__init__(
            **kwargs
        )

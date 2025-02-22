"""LLaMA ASR model configuration"""
import os

from avater_infer.models.voice import AvaterVoiceConfig


class LlamaASRConfig(AvaterVoiceConfig):
    model_type = "llama asr"

    def __init__(
        self,
        asr_adapter_hidden_size=1024,
        asr_adapter_intermediate_size=2744,
        **kwargs,
    ):
        if "llm_path" in kwargs:
            del kwargs["llm_path"]
        if "whisper_path" in kwargs:
            del kwargs["whisper_path"]
        if "wavlm_path" in kwargs:
            del kwargs["wavlm_path"]

        super().__init__(
            asr_adapter_hidden_size=asr_adapter_hidden_size,
            asr_adapter_intermediate_size=asr_adapter_intermediate_size,
            **kwargs
        )

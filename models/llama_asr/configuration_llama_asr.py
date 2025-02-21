"""LLaMA ASR model configuration"""
import os

from avater_infer.models.voice import AvaterVoiceConfig


WHIPER_PATH = os.getenv("AVATER_WHIPER_PATH", None)
WAVLM_PATH = os.getenv("AVATER_WAVLM_PATH", None)
LLM_PATH = os.getenv("AVATER_LLM_PATH", None)


class LlamaASRConfig(AvaterVoiceConfig):
    model_type = "llama asr"

    def __init__(
        self,
        whisper_hidden_size=1280,
        wavlm_hidden_size=768,
        asr_adapter_hidden_size=1024,
        asr_adapter_intermediate_size=2744,
        **kwargs,
    ):
        if "whisper_path" in kwargs:
            del kwargs["whisper_path"]
        if "wavlm_path" in kwargs:
            del kwargs["wavlm_path"]

        super().__init__(
            whisper_hidden_size=whisper_hidden_size,
            wavlm_hidden_size=wavlm_hidden_size,
            asr_adapter_hidden_size=asr_adapter_hidden_size,
            asr_adapter_intermediate_size=asr_adapter_intermediate_size,
            llm_path=LLM_PATH,
            whisper_path=WHIPER_PATH,
            wavlm_path=WAVLM_PATH,
            **kwargs
        )

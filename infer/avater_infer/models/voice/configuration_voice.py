"""Avater Voice model configuration"""

import os
import json
from typing import Any, Dict

from transformers import LlamaConfig, WhisperConfig, WavLMConfig


WHIPER_PATH = os.getenv("AVATER_WHIPER_PATH", None)
WAVLM_PATH = os.getenv("AVATER_WAVLM_PATH", None)
LLM_PATH = os.getenv("AVATER_LLM_PATH", None)


class AvaterVoiceConfig(LlamaConfig):
    model_type = "avater voice"

    def __init__(
        self,
        audio_special_tokens=8,
        code_size=2048,
        code_layers=8,
        tts_adapter_hidden_layers=6,
        tts_adapter_hidden_size=1024,
        tts_adapter_intermediate_size=2744,
        tts_adapter_attention_heads=16,
        block_step=1,
        tts_adapter_dropout=0.0,
        tts_adapter_attention_dropout=0.0,
        boa_token_id=1,
        eoa_token_id=2,
        tie_audio_embeddings=False,
        asr_adapter_hidden_size=1024,
        asr_adapter_intermediate_size=2744,
        **kwargs,
    ):
        # tts adapter param
        self.audio_special_tokens = audio_special_tokens
        self.audio_vocab_size = (code_size + audio_special_tokens) * code_layers
        self.code_size = code_size
        self.code_layers = code_layers
        self.tts_adapter_hidden_layers = tts_adapter_hidden_layers
        self.tts_adapter_hidden_size = tts_adapter_hidden_size
        self.tts_adapter_intermediate_size = tts_adapter_intermediate_size
        self.tts_adapter_attention_heads = tts_adapter_attention_heads
        self.tts_adapter_dropout = tts_adapter_dropout
        self.tts_adapter_attention_dropout = tts_adapter_attention_dropout
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id
        self.tie_audio_embeddings = tie_audio_embeddings
        self.scale_embedding = 1.0
        self.block_step = block_step
        self.block_size = code_layers//self.block_step

        # llm param
        self.llm_path = LLM_PATH
        if LLM_PATH is not None:
            llm_config_path = os.path.join(LLM_PATH, "config.json")
            llm_config = json.load(open(llm_config_path, "r", encoding="utf-8"))
            del llm_config["architectures"]
            del llm_config["model_type"]
            kwargs.update(llm_config)

        # asr adapter param
        self.asr_adapter_hidden_size = asr_adapter_hidden_size
        self.asr_adapter_intermediate_size = asr_adapter_intermediate_size

        self.whisper_path = WHIPER_PATH
        if WHIPER_PATH is not None:
            whisper_config_path = os.path.join(WHIPER_PATH, "config.json")
            whisper_config = json.load(open(whisper_config_path, "r", encoding="utf-8"))
            self.whisper_config = WhisperConfig(**whisper_config)
        else:
            self.whisper_config = WhisperConfig(**kwargs["whisper_config"])
            del kwargs["whisper_config"]

        self.wavlm_path = WAVLM_PATH
        if WAVLM_PATH is not None:
            wavlm_config_path = os.path.join(WAVLM_PATH, "config.json")
            wavlm_config = json.load(open(wavlm_config_path, "r", encoding="utf-8"))
            self.wavlm_config = WavLMConfig(**wavlm_config)
        else:
            self.wavlm_config = WavLMConfig(**kwargs["wavlm_config"])
            del kwargs["wavlm_config"]

        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()

        if "llm_path" in output:
            del output["llm_path"]

        if "whisper_path" in output:
            del output["whisper_path"]

        if "wavlm_path" in output:
            del output["wavlm_path"]

        return output

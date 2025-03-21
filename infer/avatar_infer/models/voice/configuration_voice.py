"""Avatar Voice model configuration"""

import os
import json
from typing import Any, Dict

from transformers import LlamaConfig, WhisperConfig, WavLMConfig


WHISPER_PATH = os.getenv("AVATER_WHISPER_PATH", None)
WAVLM_PATH = os.getenv("AVATER_WAVLM_PATH", None)
LLM_PATH = os.getenv("AVATER_LLM_PATH", None)
TTS_PATH = os.getenv("AVATER_TTS_PATH", None)
ASR_PATH = os.getenv("AVATER_ASR_PATH", None)


class AvatarVoiceConfig(LlamaConfig):
    model_type = "avatar voice"

    def __init__(
        self,
        audio_special_tokens=None,
        code_size=None,
        code_layers=None,
        tts_adapter_hidden_layers=None,
        tts_adapter_hidden_size=None,
        tts_adapter_intermediate_size=None,
        tts_adapter_attention_heads=None,
        block_step=None,
        tts_adapter_dropout=None,
        tts_adapter_attention_dropout=None,
        boa_token_id=None,
        eoa_token_id=None,
        tie_audio_embeddings=None,
        asr_adapter_hidden_size=None,
        asr_adapter_intermediate_size=None,
        **kwargs,
    ):
        self.scale_embedding = 1.0
        self.llm_path = LLM_PATH
        self.asr_path = ASR_PATH
        self.tts_path = TTS_PATH
        self.whisper_path = WHISPER_PATH
        self.wavlm_path = WAVLM_PATH

        # asr adapter param
        if asr_adapter_hidden_size:
            self.asr_adapter_hidden_size = asr_adapter_hidden_size
            self.asr_adapter_intermediate_size = asr_adapter_intermediate_size
        elif self.asr_path is not None:
            asr_config_path = os.path.join(self.asr_path, "config.json")
            asr_config = json.load(open(asr_config_path, "r", encoding="utf-8"))
            self.asr_adapter_hidden_size = asr_config["asr_adapter_hidden_size"]
            self.asr_adapter_intermediate_size = asr_config["asr_adapter_intermediate_size"]
            kwargs["whisper_config"] = asr_config["whisper_config"]
            kwargs["wavlm_config"] = asr_config["wavlm_config"]

        # tts adapter param
        if audio_special_tokens:
            self.audio_special_tokens = audio_special_tokens
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
            self.block_step = block_step
            self.block_size = self.code_layers//self.block_step
            self.audio_vocab_size = (code_size + audio_special_tokens) * code_layers
        elif self.tts_path is not None:
            tts_config_path = os.path.join(self.tts_path, "config.json")
            tts_config = json.load(open(tts_config_path, "r", encoding="utf-8"))
            self.audio_special_tokens = tts_config["audio_special_tokens"]
            self.code_size = tts_config["code_size"]
            self.code_layers = tts_config["code_layers"]
            self.tts_adapter_hidden_layers = tts_config["tts_adapter_hidden_layers"]
            self.tts_adapter_hidden_size = tts_config["tts_adapter_hidden_size"]
            self.tts_adapter_intermediate_size = tts_config["tts_adapter_intermediate_size"]
            self.tts_adapter_attention_heads = tts_config["tts_adapter_attention_heads"]
            self.tts_adapter_dropout = tts_config["tts_adapter_dropout"]
            self.tts_adapter_attention_dropout = tts_config["tts_adapter_attention_dropout"]
            self.boa_token_id = tts_config["boa_token_id"]
            self.eoa_token_id = tts_config["eoa_token_id"]
            self.tie_audio_embeddings = tts_config["tie_audio_embeddings"]
            self.block_step = tts_config["block_step"]
            self.block_size = self.code_layers//self.block_step
            self.audio_vocab_size = (self.code_size + self.audio_special_tokens) * self.code_layers

        # Model config path
        if self.llm_path is not None:
            llm_config_path = os.path.join(self.llm_path, "config.json")
            llm_config = json.load(open(llm_config_path, "r", encoding="utf-8"))
            del llm_config["architectures"]
            del llm_config["model_type"]
            kwargs.update(llm_config)

        if self.whisper_path is not None:
            whisper_config_path = os.path.join(self.whisper_path, "config.json")
            whisper_config = json.load(open(whisper_config_path, "r", encoding="utf-8"))
            self.whisper_config = WhisperConfig(**whisper_config)
        elif "whisper_config" in kwargs:
            self.whisper_config = WhisperConfig(**kwargs["whisper_config"])
            del kwargs["whisper_config"]

        if self.wavlm_path is not None:
            wavlm_config_path = os.path.join(self.wavlm_path, "config.json")
            wavlm_config = json.load(open(wavlm_config_path, "r", encoding="utf-8"))
            self.wavlm_config = WavLMConfig(**wavlm_config)
        elif "wavlm_config" in kwargs:
            self.wavlm_config = WavLMConfig(**kwargs["wavlm_config"])
            del kwargs["wavlm_config"]

        super().__init__(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        output = super().to_dict()

        del output["llm_path"]
        del output["asr_path"]
        del output["tts_path"]
        del output["whisper_path"]
        del output["wavlm_path"]
        return output

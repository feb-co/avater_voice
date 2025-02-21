"""Avater Voice model configuration"""

import os
import json

from transformers import LlamaConfig


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
        llm_path=None,
        whisper_path=None,
        wavlm_path=None,
        whisper_hidden_size=1280,
        wavlm_hidden_size=768,
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
        self.llm_path = llm_path
        if llm_path is not None:
            llm_config_path = os.path.join(llm_path, "config.json")
            llm_config = json.load(open(llm_config_path, "r", encoding="utf-8"))
            del llm_config["architectures"]
            del llm_config["model_type"]
            kwargs.update(llm_config)

        # asr adapter param
        self.whisper_path = whisper_path
        self.whisper_hidden_size = whisper_hidden_size

        self.wavlm_path = wavlm_path
        self.wavlm_hidden_size = wavlm_hidden_size

        self.asr_adapter_hidden_size = asr_adapter_hidden_size
        self.asr_adapter_intermediate_size = asr_adapter_intermediate_size

        super().__init__(**kwargs)

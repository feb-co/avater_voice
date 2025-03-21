"""LLaMA TTS model configuration"""
import os

from avatar_infer.models.voice import AvatarVoiceConfig


class LlamaTTSConfig(AvatarVoiceConfig):
    model_type = "llama tts"

    def __init__(
        self,
        audio_special_tokens=8,
        block_step=1,
        code_size=2048,
        code_layers=8,
        tts_adapter_hidden_layers=6,
        tts_adapter_hidden_size=1024,
        tts_adapter_intermediate_size=2744,
        tts_adapter_attention_heads=16,
        tts_adapter_dropout=0.0,
        tts_adapter_attention_dropout=0.0,
        boa_token_id=1,
        eoa_token_id=2,
        tie_audio_embeddings=False,
        **kwargs,
    ):
        if "llm_path" in kwargs:
            del kwargs["llm_path"]
        if "whisper_path" in kwargs:
            del kwargs["whisper_path"]
        if "wavlm_path" in kwargs:
            del kwargs["wavlm_path"]

        super().__init__(
            audio_special_tokens=audio_special_tokens,
            block_step=block_step,
            code_size=code_size,
            code_layers=code_layers,
            tts_adapter_hidden_layers=tts_adapter_hidden_layers,
            tts_adapter_hidden_size=tts_adapter_hidden_size,
            tts_adapter_intermediate_size=tts_adapter_intermediate_size,
            tts_adapter_attention_heads=tts_adapter_attention_heads,
            tts_adapter_dropout=tts_adapter_dropout,
            tts_adapter_attention_dropout=tts_adapter_attention_dropout,
            boa_token_id=boa_token_id,
            eoa_token_id=eoa_token_id,
            tie_audio_embeddings=tie_audio_embeddings,
            **kwargs
        )

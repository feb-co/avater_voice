"""LLaMA TTS model configuration"""

from avater_infer.models.llama import LlamaVoiceConfig


class LlamaTTSConfig(LlamaVoiceConfig):
    model_type = "llama tts"

    def __init__(
        self,
        audio_special_tokens=8,
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
        llm_path=None,
        tie_audio_embeddings=False,
        **kwargs,
    ):
        super().__init__(
            audio_special_tokens=audio_special_tokens,
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
            llm_path=llm_path,
            tie_audio_embeddings=tie_audio_embeddings,
            **kwargs
        )

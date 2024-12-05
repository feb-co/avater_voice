"""LLaMA TTS model configuration"""

from transformers import LlamaConfig


class LlamaTTSConfig(LlamaConfig):
    model_type = "llama tts"

    def __init__(
        self,
        audio_vocab_size=32000,
        tts_adapter_hidden_layers=6,
        tts_adapter_attention_heads=16,
        tts_adapter_attn_implementation="eager",
        tts_adapter_dropout=0.0,
        boa_token_id=1,
        eoa_token_id=2,
        **kwargs,
    ):
        self.audio_vocab_size = audio_vocab_size
        self.tts_adapter_hidden_layers = tts_adapter_hidden_layers
        self.tts_adapter_attention_heads = tts_adapter_attention_heads
        self.tts_adapter_attn_implementation = tts_adapter_attn_implementation
        self.tts_adapter_dropout = tts_adapter_dropout
        self.scale_embedding = 1.0
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id

        super().__init__(**kwargs)

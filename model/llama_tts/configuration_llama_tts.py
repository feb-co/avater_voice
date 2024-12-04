"""LLaMA TTS model configuration"""

from transformers import LlamaConfig


class LlamaTTSConfig(LlamaConfig):
    model_type = "llama tts"

    def __init__(
        self,
        audio_vocab_size=32000,
        adapter_hidden_layers=6,
        adapter_attention_heads=16,
        dropout=0.0,
        boa_token_id=1,
        eoa_token_id=2,
        **kwargs,
    ):
        self.audio_vocab_size = audio_vocab_size
        self.adapter_hidden_layers = adapter_hidden_layers
        self.adapter_attention_heads = adapter_attention_heads
        self.dropout = dropout
        self.boa_token_id = boa_token_id
        self.eoa_token_id = eoa_token_id

        super().__init__(**kwargs)

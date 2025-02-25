import os
from typing import Any, Dict, Optional

from avater_infer.models.voice import AvaterVoiceTokenizer


class LlamaASRTokenizer(AvaterVoiceTokenizer):
    def __init__(
        self,
        audio_downsample_layer=2,
        audio_encoder_sample_rate=16000,
        audio_encoder_mel_size=128,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            audio_downsample_layer=audio_downsample_layer,
            audio_encoder_sample_rate=audio_encoder_sample_rate,
            audio_encoder_mel_size=audio_encoder_mel_size,
            device=device,
            **kwargs
        )

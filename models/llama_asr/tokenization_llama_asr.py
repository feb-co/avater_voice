import os
from typing import Any, Dict, Optional

from avater_infer.models.voice import AvaterVoiceTokenizer


TEXT_TOKENIZER_PATH = os.getenv("AVATER_TEXT_TOKENIZER_PATH", None)
AUDIO_TOKENIZER_PATH = os.getenv("AVATER_AUDIO_TOKENIZER_PATH", None)


class LlamaASRTokenizer(AvaterVoiceTokenizer):
    def __init__(
        self,
        audio_downsample_layer=2,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            audio_downsample_layer=audio_downsample_layer,
            device=device,
            **kwargs
        )

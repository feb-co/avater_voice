import os
from typing import Any, Dict, Optional

from avater_infer.models.voice import AvaterVoiceTokenizer



class LlamaTTSTokenizer(AvaterVoiceTokenizer):
    def __init__(
        self,
        audio_special_token: Optional[Dict[str, Any]] = None,
        short_wait_string="<|SHORT_WAIT|>",
        long_wait_string="<|LONG_WAIT|>",
        audio_tokenizer="moshi_mimi",
        text_duration_token=5,
        device="cpu",
        **kwargs
    ):
        super().__init__(
            audio_special_token=audio_special_token,
            short_wait_string=short_wait_string,
            long_wait_string=long_wait_string,
            audio_tokenizer=audio_tokenizer,
            text_duration_token=text_duration_token,
            device=device,
            **kwargs
        )

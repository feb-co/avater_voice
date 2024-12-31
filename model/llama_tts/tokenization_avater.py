import os
import re
from audiotools import AudioSignal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging
from transformers import AutoTokenizer


class AvaterTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        text_tokenizer_path,
        audio_special_token: Optional[Dict[str, Any]] = None,
        short_wait_string="<|SHORT_WAIT|>",
        long_wait_string="<|LONG_WAIT|>",
        audio_tokenizer="moshi_mimi",
        cpt_cache=".cache/",
        **kwargs
    ):
        if not os.path.isdir(text_tokenizer_path):
            raise ValueError(
                f"Can't find a text vocabulary file at path '{text_tokenizer_path}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        if not os.path.exists(cpt_cache):
            os.mkdir(cpt_cache)

        # var init
        self.audio_special_token = audio_special_token
        self.short_wait_string = short_wait_string
        self.long_wait_string = long_wait_string

        # tokenizer init
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)
        
        if audio_tokenizer == "moshi_mimi":
            from mimi import MimiTokenizer
            self.audio_tokenizer = MimiTokenizer.load_from_checkpoint(
                cpt_dir=cpt_cache,
                device="cpu"
            )
        else:
            raise NotImplementedError
    
    def encode(
        self,
        text: str,
        audio_signal: AudioSignal,
        add_special_tokens=True,
        **kwargs
    ):
        text_token_ids = self.text_tokenizer.encode(
            text=text,
            add_special_tokens=add_special_tokens,
            kwargs=kwargs
        )

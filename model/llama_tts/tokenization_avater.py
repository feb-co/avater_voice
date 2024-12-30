import os
import re
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
        **kwargs
    ):
        if not os.path.isdir(text_tokenizer_path):
            raise ValueError(
                f"Can't find a text vocabulary file at path '{text_tokenizer_path}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_tokenizer_path)

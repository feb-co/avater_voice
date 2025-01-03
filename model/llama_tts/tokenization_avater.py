import os
import json
import torch
import whisper
from audiotools import AudioSignal
from typing import Union, Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging
from transformers import AutoTokenizer


def load_whisper_audio(path):
    audio = whisper.load_audio(path)
    duration_ms = (len(audio) / 16000) * 1000
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio)
    return mel, int(duration_ms / 20) + 1


class AvaterTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        text_tokenizer_path,
        audio_special_token: Optional[Dict[str, Any]] = None,
        short_wait_string="<|SHORT_WAIT|>",
        long_wait_string="<|LONG_WAIT|>",
        audio_tokenizer="moshi_mimi",
        cpt_cache=".cache/",
        device="cpu",
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
        self.whisper_tokenizer = whisper.load_model("small").to(device)
        if audio_tokenizer == "moshi_mimi":
            from mimi import MimiTokenizer
            self.audio_tokenizer = MimiTokenizer.load_from_checkpoint(
                cpt_dir=cpt_cache,
                device=device
            )
        else:
            raise NotImplementedError

    def encode(
        self,
        text: str,
        audio_signal: Optional[Union[List[Dict], AudioSignal]]=None,
        add_special_tokens=True,
        **kwargs
    ) -> Union[Tuple[List, List], List]:
        text_token_ids = self.text_tokenizer.encode(
            text=text,
            add_special_tokens=add_special_tokens,
            kwargs=kwargs
        )

        if audio_signal:
            with torch.no_grad():
                if isinstance(audio_signal, list):
                    codes = []
                    for signal in audio_signal:
                        if signal["split"] == self.long_wait_string:
                            split_token = self.audio_special_token["long_wait_token"]
                        elif signal["split"] == self.short_wait_string:
                            split_token = self.audio_special_token["short_wait_token"]
                        else:
                            split_token = None

                        sub_codes = self.audio_tokenizer.encode(signal["signal"].audio_data if "signal" in signal else AudioSignal(signal["file"]).audio_data)
                        for idx, sub_code in enumerate(sub_codes):
                            code_list: list = sub_code.to_list()
                            
                            if split_token:
                                code_list = code_list.insert(0, split_token)

                            if len(codes) != sub_codes:
                                codes.append(code_list)
                            else:
                                codes[idx] += code_list
                else:
                    codes = self.audio_tokenizer.encode(audio_signal.audio_data)
                    for idx in range(len(codes)):
                        codes[idx] = codes[idx].to_list()

            return (text_token_ids, codes)
        else:
            return text_token_ids

    def encode_whisper_features(self, elem):
        if not isinstance(elem, dict):
            audio_signal = json.loads(elem)

        token_ids = []
        audio_features = None
        audio_pos = []
        for signal in audio_signal:
            if signal["split"]:
                token_ids += self.text_tokenizer.encode(signal["split"], add_special_tokens=False)

            mel, leng = load_whisper_audio(signal["file"])
            audio_pos.append([len(token_ids), leng])
            token_ids += [self.text_tokenizer.eos_token_id] * leng
            if not audio_features:
                audio_features = self.whisper_tokenizer.embed_audio(mel)[0][:leng]
            else:
                audio_features = torch.cat([audio_features, self.whisper_tokenizer.embed_audio(mel)[0][:leng]], dim=0)

        return token_ids, audio_features, audio_pos

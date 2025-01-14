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
            self.audio_duration_token = 13
            self.text_duration_token = 4
        else:
            raise NotImplementedError

    def __repr__(self) -> str:
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.text_tokenizer.added_tokens_decoder.items()])
        return (
            f"{self.__class__.__name__}(name_or_path='{self.text_tokenizer.name_or_path}',"
            f" vocab_size={self.text_tokenizer.vocab_size}, model_max_length={self.text_tokenizer.model_max_length}, is_fast={self.text_tokenizer.is_fast},"
            f" padding_side='{self.text_tokenizer.padding_side}', truncation_side='{self.text_tokenizer.truncation_side}',"
            f" special_tokens={self.text_tokenizer.special_tokens_map}, clean_up_tokenization_spaces={self.text_tokenizer.clean_up_tokenization_spaces},"
            " added_tokens_decoder={\n\t" + added_tokens_decoder_rep + "\n}\n)"
        )

    def encode(
        self,
        text: str,
        audio_signal: Optional[Union[List[Dict], AudioSignal]]=None,
        add_special_tokens=True,
        add_audio_special_tokens=True,
        acoustic_delay: int=1,
        **kwargs
    ) -> Union[Tuple[List, List], List]:
        text_token_ids = self.text_tokenizer.encode(
            text=text,
            add_special_tokens=add_special_tokens,
            **kwargs
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

                        sub_codes = self.audio_tokenizer.encode(signal["signal"].audio_data if "signal" in signal else AudioSignal(signal["file"]).audio_data)[0]
                        for idx, sub_code in enumerate(sub_codes):
                            code_list: list = sub_code.tolist()

                            if split_token:
                                code_list = code_list.insert(0, split_token)

                            if len(codes) != len(sub_codes):
                                codes.append(code_list)
                            else:
                                codes[idx] += code_list

                    for idx in range(len(codes)):
                        pad_tokens = [self.audio_special_token["pad_token"]] * acoustic_delay
                        if add_audio_special_tokens:
                            if idx == 0:
                                codes[idx] = [self.audio_special_token["bos_token"]] + codes[idx] + [self.audio_special_token["eos_token"]] + pad_tokens
                            else:
                                codes[idx] = pad_tokens + [self.audio_special_token["bos_token"]] + codes[idx] + [self.audio_special_token["eos_token"]]
                else:
                    codes = self.audio_tokenizer.encode(audio_signal.audio_data)[0]
                    for idx in range(len(codes)):
                        pad_tokens = [self.audio_special_token["pad_token"]] * acoustic_delay
                        codes[idx] = codes[idx].tolist()
                        if add_audio_special_tokens:
                            if idx == 0:
                                codes[idx] = [self.audio_special_token["bos_token"]] + codes[idx] + [self.audio_special_token["eos_token"]] + pad_tokens
                            else:
                                codes[idx] = pad_tokens + [self.audio_special_token["bos_token"]] + codes[idx] + [self.audio_special_token["eos_token"]]

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

    def convert_t2a_attention_mask(self, text_tokens: list[int], audio_tokens: list):
        audio_length = len(audio_tokens[0])
        text_length = len(text_tokens)

        attention_mask = [[0 for _ in range(text_length)] for _ in range(audio_length)]
        text_token_threshold = 0
        for audio_idx in range(audio_length):
            if audio_idx % self.audio_duration_token == 0:
                text_token_threshold += self.text_duration_token
            text_token_threshold = self.get_complete_phrase(text_tokens, text_token_threshold)
            print(audio_idx, text_token_threshold, "---", flush=True)

        assert text_token_threshold >= text_length, "audio_tokens need cover text_tokens!"
        return attention_mask

    def get_complete_phrase(self, text_tokens: list[int], text_token_threshold: int):
        pass

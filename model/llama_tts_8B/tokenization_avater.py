import os
import json
import copy
import torch
import whisper
from audiotools import AudioSignal
from collections.abc import Mapping, Sized
from typing import Union, Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer, EncodedInput, PaddingStrategy
from transformers.utils import TensorType
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
        acoustic_delay=1,
        text_duration_token=5,
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

        self.init_kwargs = copy.deepcopy(kwargs)

        # var init
        self.audio_special_token = audio_special_token
        self.short_wait_string = short_wait_string
        self.long_wait_string = long_wait_string
        self.acoustic_delay = acoustic_delay
        self.device = device
        self.text_duration_token = text_duration_token
        self.pad_token = None
        self.verbose = True

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
            self.code_size = 2048
        else:
            raise NotImplementedError

        self.phrase_stop_token = self.text_tokenizer.encode(f" {short_wait_string}", add_special_tokens=False)[0]

    def __repr__(self) -> str:
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.text_tokenizer.added_tokens_decoder.items()])
        return (
            f"{self.__class__.__name__}(name_or_path='{self.text_tokenizer.name_or_path}',"
            f" vocab_size={self.text_tokenizer.vocab_size}, model_max_length={self.text_tokenizer.model_max_length}, is_fast={self.text_tokenizer.is_fast},"
            f" padding_side='{self.text_tokenizer.padding_side}', truncation_side='{self.text_tokenizer.truncation_side}',"
            f" special_tokens={self.text_tokenizer.special_tokens_map}, clean_up_tokenization_spaces={self.text_tokenizer.clean_up_tokenization_spaces},"
            " added_tokens_decoder={\n\t" + added_tokens_decoder_rep + "\n}\n)"
        )

    def pad(
        self,
        encoded_inputs: Union[
            BatchEncoding,
            List[BatchEncoding],
            Dict[str, EncodedInput],
            Dict[str, List[EncodedInput]],
            List[Dict[str, EncodedInput]],
        ],
        padding: Union[bool, str, PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[str] = "right",
        return_attention_mask: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        verbose: bool = True,
    ) -> BatchEncoding:
        # If we have a list of dicts, let's convert it in a dict of lists
        # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
        if isinstance(encoded_inputs, (list, tuple)) and isinstance(encoded_inputs[0], Mapping):
            encoded_inputs = {key: [example[key] for example in encoded_inputs] for key in encoded_inputs[0].keys()}

        batch_outputs = {}
        for key, values in encoded_inputs.items():
            if key not in batch_outputs:
                batch_outputs[key] = []

            if "input_ids" == key:
                pad_token = self.text_tokenizer.pad_token_id
            elif "decoder_input_ids" == key:
                pad_token = self.audio_special_token["eoa_token"]
            elif "decoder_labels" == key:
                pad_token = -100
            else:
                pad_token = 0

            if key in ("input_ids", "attention_mask", "valid_tokens_pos", "decoder_attention_mask"):
                max_length = max([len(item) for item in values])
                for value in values:
                    difference = max_length - len(value)
                    if padding_side == "right":
                        outputs = value + [pad_token] * difference
                    elif padding_side == "left":
                        outputs = [pad_token] * difference + value
                    else:
                        raise ValueError(f"Invalid padding strategy: {padding_side}")

                    batch_outputs[key].append(outputs)
            elif key in ("decoder_input_ids",):
                max_length = max([len(item[0]) for item in values])
                for value in values:
                    outputs = []
                    difference = max_length - len(value[0])
                    for layer_idx, out in enumerate(value):
                        if padding_side == "right":
                            out = out + self.audio_code_shift([pad_token], layer_idx) * difference
                        elif padding_side == "left":
                            out = self.audio_code_shift([pad_token], layer_idx) * difference + out
                        else:
                            raise ValueError(f"Invalid padding strategy:{padding_side}")

                        outputs.append(out)
                    batch_outputs[key].append(outputs)
            elif key in ("decoder_labels", ):
                max_length = max([len(item[0]) for item in values])
                for layer_idx, value in enumerate(values):
                    outputs = []
                    difference = max_length - len(value[0])
                    for out in value:
                        if padding_side == "right":
                            out = out + [pad_token] * difference
                        elif padding_side == "left":
                            out = [pad_token] * difference + out
                        else:
                            raise ValueError(f"Invalid padding strategy:{padding_side}")

                        outputs.append(out)
                    batch_outputs[key].append(outputs)
            elif key in ("encoder_decoder_attention_mask",):
                max_length = [
                    max([len(item) for item in values]),
                    max([len(item[0]) for item in values]),
                ]
                for value in values:
                    outputs = torch.zeros(max_length, dtype=torch.long)
                    outputs[:len(value), :len(value[0])] = torch.LongTensor(value)
                    batch_outputs[key].append(outputs.tolist())
            else:
                raise ValueError(f"Invalid key padding strategy: {key}")

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def encode(
        self,
        text: str=None,
        audio_signal: Optional[Union[List[Dict], AudioSignal]]=None,
        add_special_tokens=True,
        add_audio_special_tokens=True,
        **kwargs
    ) -> Union[Tuple[List, List], List]:
        if text:
            text_token_ids = self.text_tokenizer.encode(
                text=text,
                add_special_tokens=add_special_tokens,
                **kwargs
            )
        else:
            text_token_ids = None
            
        boa_tokens = [self.audio_special_token["boa_token"]] * self.acoustic_delay
        eoa_tokens = [self.audio_special_token["eoa_token"]] * self.acoustic_delay

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

                        sub_codes = self.audio_tokenizer.encode(
                            signal["signal"].audio_data.to(self.device)
                            if "signal" in signal else
                            AudioSignal(signal["file"]).audio_data.to(self.device)
                        )[0]
                        for idx, sub_code in enumerate(sub_codes):
                            code_list: list = sub_code.tolist()

                            if split_token:
                                code_list.insert(0, split_token)

                            if len(codes) != len(sub_codes):
                                codes.append(code_list)
                            else:
                                codes[idx] += code_list

                    for idx in range(len(codes)):
                        if add_audio_special_tokens:
                            if idx == 0:
                                codes[idx] = [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]] + eoa_tokens
                            else:
                                codes[idx] = boa_tokens + [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]]
                else:
                    codes = self.audio_tokenizer.encode(audio_signal.audio_data)[0]
                    for idx in range(len(codes)):
                        codes[idx] = codes[idx].tolist()
                        if add_audio_special_tokens:
                            if idx == 0:
                                codes[idx] = [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]] + eoa_tokens
                            else:
                                codes[idx] = boa_tokens + [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]]

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

        attention_mask = torch.zeros([audio_length, text_length])
        text_token_threshold = 0
        for audio_idx in range(audio_length):
            if audio_idx % self.audio_duration_token == 0:
                text_token_threshold += self.text_duration_token
            text_token_threshold = self.get_complete_phrase(text_tokens, text_token_threshold)
            attention_mask[audio_idx][:text_token_threshold] = 1

        assert text_token_threshold >= text_length, "audio_tokens need cover text_tokens!"

        return attention_mask.tolist()

    def get_complete_phrase(self, text_tokens: list[int], text_token_threshold: int):
        subwords = self.text_tokenizer.convert_ids_to_tokens(text_tokens)
        current_phrase = subwords[:text_token_threshold]
        new_text_token_threshold = text_token_threshold
        for idx in range(text_token_threshold, len(subwords)):
            if subwords[idx].startswith("Ġ") and text_tokens[idx] != self.phrase_stop_token:
                break
            else:
                current_phrase.append(subwords[idx])
                new_text_token_threshold += 1

        return new_text_token_threshold

    def audio_code_shift(self, input_ids, layer_idx):
        return [input_id + layer_idx * (self.code_size+len(self.audio_special_token)) for input_id in input_ids]

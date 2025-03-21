import os
import json
import copy
import torch
import whisper
import numpy as np
from collections.abc import Mapping
from typing import Union, Any, Dict, List, Optional, Tuple

from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer, EncodedInput, PaddingStrategy, TOKENIZER_CONFIG_FILE
from transformers.utils import TensorType
from transformers.dynamic_module_utils import custom_object_save
from transformers import AutoTokenizer, AutoFeatureExtractor


TEXT_TOKENIZER_PATH = os.getenv("AVATER_TEXT_TOKENIZER_PATH", None)
AUDIO_TOKENIZER_PATH = os.getenv("AVATER_AUDIO_TOKENIZER_PATH", None)
WHISPER_TOKENIZER_PATH = os.getenv("AVATER_WHISPER_PATH", None)
WAVLM_TOKENIZER_PATH = os.getenv("AVATER_WAVLM_PATH", None)


def get_audio_encoder_out_seq_lens(array_len):
    duration_ms = (array_len / 16000) * 1000
    return int(duration_ms / 20) + 1


def get_1dconv_out_seq_lens(n_layers, in_seq_lens):
    if isinstance(in_seq_lens, int):
        out = in_seq_lens
        for _ in range(n_layers):
            out = int((out - 1) / 2 + 1)
        return out
    else:
        out = in_seq_lens.clone()
        for _ in range(n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out


def generate_chat_template(messages, custom_tools=None, builtin_tools=None, add_generation_prompt=False):
    # Initialize variables
    tools_in_user_message = True
    tools = custom_tools if custom_tools is not None else None
    system_message = ""

    # Extract system message
    if messages and messages[0]['role'] == 'system':
        system_message = messages[0]['content'].strip()
        messages = messages[1:]
    
    # Build template string
    template = "<|start_header_id|>system<|end_header_id|>\n\n"
    
    if builtin_tools or tools is not None:
        template += "Environment: ipython\n"
    
    if builtin_tools:
        tools_list = [tool for tool in builtin_tools if tool != 'code_interpreter']
        template += f"Tools: {', '.join(tools_list)}\n\n"
    
    if tools is not None and not tools_in_user_message:
        template += (
            "You have access to the following functions. To call a function, please respond with JSON for a function call.\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.\n"
            "Do not use variables.\n\n"
        )
        for tool in tools:
            template += f"{tool}\n\n"
    
    template += system_message + "<|eot_id|>\n\n"
    
    if tools_in_user_message and tools is not None:
        if messages:
            first_user_message = messages[0]['content'].strip()
            messages = messages[1:]
        else:
            raise Exception("Cannot put tools in the first user message when there's no first user message!")
        
        template += (
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "Given the following functions, please respond with a JSON for a function call "
            "with its proper arguments that best answers the given prompt.\n\n"
            "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.\n"
            "Do not use variables.\n\n"
        )
        for tool in tools:
            template += f"{tool}\n\n"
        
        template += first_user_message + "<|eot_id|>\n"
    
    for message in messages:
        if message['role'] not in ['ipython', 'tool'] and 'tool_calls' not in message:
            template += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>\n"
        elif 'tool_calls' in message:
            if len(message['tool_calls']) != 1:
                raise Exception("This model only supports single tool-calls at once!")
            
            tool_call = message['tool_calls'][0]['function']
            if builtin_tools and tool_call['name'] in builtin_tools:
                template += f"<|start_header_id|>assistant<|end_header_id|>\n\n<|python_tag|>{tool_call['name']}.call("
                template += ", ".join(f'{arg_name}="{arg_val}"' for arg_name, arg_val in tool_call['arguments'].items())
                template += ")"
            else:
                template += f"<|start_header_id|>assistant<|end_header_id|>\n\n{{\"name\": \"{tool_call['name']}\", \"parameters\": {tool_call['arguments']}}}"
            
            template += "<|eom_id|>" if builtin_tools else "<|eot_id|>"
        elif message['role'] in ["tool", "ipython"]:
            template += "<|start_header_id|>ipython<|end_header_id|>\n\n"
            content = message['content']
            if isinstance(content, (dict, list)):
                template += json.dumps(content)
            else:
                template += content
            template += "<|eot_id|>\n"
    
    if add_generation_prompt:
        template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    
    return template


def get_jinja_template():
    return """{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message + builtin tools #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if builtin_tools is defined or tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{%- if builtin_tools is defined %}
    {{- "Tools: " + builtin_tools | reject('equalto', 'code_interpreter') | join(", ") + "\n\n"}}
{%- endif %}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- "<|python_tag|>" + tool_call.name + ".call(" }}
            {%- for arg_name, arg_val in tool_call.arguments | items %}
                {{- arg_name + '="' + arg_val + '"' }}
                {%- if not loop.last %}
                    {{- ", " }}
                {%- endif %}
                {%- endfor %}
            {{- ")" }}
        {%- else  %}
            {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
            {{- '{"name": "' + tool_call.name + '", ' }}
            {{- '"parameters": ' }}
            {{- tool_call.arguments | tojson }}
            {{- "}" }}
        {%- endif %}
        {%- if builtin_tools is defined %}
            {#- This means we're in ipython mode #}
            {{- "<|eom_id|>" }}
        {%- else %}
            {{- "<|eot_id|>" }}
        {%- endif %}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""


class AvatarVoiceTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        audio_special_token: Optional[Dict[str, Any]] = None,
        short_wait_string=None,
        long_wait_string=None,
        audio_tokenizer=None,
        text_duration_token=None,
        audio_downsample_layer=None,
        audio_encoder_sample_rate=None,
        audio_encoder_mel_size=None,
        device="cpu",
        chat_template=None,
        **kwargs
    ):
        if not os.path.isdir(TEXT_TOKENIZER_PATH):
            raise ValueError(
                f"Can't find a text tokenizer file at path '{TEXT_TOKENIZER_PATH}'. To load the vocabulary from a Google pretrained"
                " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )

        self.init_kwargs = copy.deepcopy(kwargs)

        # var init
        self.device = device
        self.verbose = True
        self.chat_template = chat_template if chat_template else get_jinja_template()
        self.init_inputs = ()
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, attr, None)

        # text tokenizer init
        self.text_tokenizer = AutoTokenizer.from_pretrained(TEXT_TOKENIZER_PATH)
        self.bos_token = self.text_tokenizer.bos_token
        self.total_vocab_size = len(self.get_vocab())

        # audio encoder init
        if audio_downsample_layer:
            self.audio_downsample_layer = audio_downsample_layer
            self.audio_encoder_sample_rate = audio_encoder_sample_rate
            self.audio_encoder_mel_size = audio_encoder_mel_size
            self.whisper_processor = AutoFeatureExtractor.from_pretrained(WHISPER_TOKENIZER_PATH)
            self.wavlm_processor = AutoFeatureExtractor.from_pretrained(WAVLM_TOKENIZER_PATH)

        # audio tokenizer init
        if audio_special_token:
            self.audio_special_token = audio_special_token
            self.short_wait_string = short_wait_string
            self.long_wait_string = long_wait_string
            self.text_duration_token = text_duration_token
            self.phrase_stop_token = self.text_tokenizer.encode(f" {short_wait_string}", add_special_tokens=False)[0]
            if audio_tokenizer == "moshi_mimi":
                from mimi import MimiTokenizer
                self.audio_tokenizer_type = audio_tokenizer
                self.audio_tokenizer = MimiTokenizer.load_from_checkpoint(
                    cpt_dir=AUDIO_TOKENIZER_PATH,
                    device=device
                )
                self.audio_duration_token = 13
                self.code_size = 2048
                self.code_layer = 8
                self.audio_tokenizer_sample_rate = 24000
            else:
                raise NotImplementedError

    @property
    def all_special_tokens_extended(self):
        """
        `List[Union[str, tokenizers.AddedToken]]`: All the special tokens (`'<unk>'`, `'<cls>'`, etc.), the order has
        nothing to do with the index of each tokens. If you want to know the correct indices, check
        `self.added_tokens_encoder`. We can't create an order anymore as the keys are `AddedTokens` and not `Strings`.

        Don't convert tokens of `tokenizers.AddedToken` type to string so they can be used to control more finely how
        special tokens are tokenized.
        """
        all_tokens = []
        seen = set()
        for value in self.text_tokenizer.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def all_special_ids(self) -> List[int]:
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        all_toks = self.text_tokenizer.all_special_tokens
        all_ids = self.text_tokenizer.convert_tokens_to_ids(all_toks)
        return all_ids

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.text_tokenizer.sp_model.get_piece_size()

    def __repr__(self) -> str:
        added_tokens_decoder_rep = "\n\t".join([f"{k}: {v.__repr__()}," for k, v in self.text_tokenizer.added_tokens_decoder.items()])
        return (
            f"{self.__class__.__name__}(name_or_path='{self.text_tokenizer.name_or_path}',"
            f" vocab_size={self.text_tokenizer.vocab_size}, model_max_length={self.text_tokenizer.model_max_length}, is_fast={self.text_tokenizer.is_fast},"
            f" padding_side='{self.text_tokenizer.padding_side}', truncation_side='{self.text_tokenizer.truncation_side}',"
            f" special_tokens={self.special_tokens_map}, clean_up_tokenization_spaces={self.text_tokenizer.clean_up_tokenization_spaces},"
            " added_tokens_decoder={\n\t" + added_tokens_decoder_rep + "\n}\n)"
        )

    def _encode_whisper_feature(self, signal_array, type="tensor"):
        audio = whisper.pad_or_trim(signal_array)

        # Create mel spectrogram
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.audio_encoder_mel_size, device=self.device)

        # Calculate mask based on actual audio length
        audio_length = len(signal_array)
        n_frames = audio_length // whisper.audio.HOP_LENGTH
        mel_mask = torch.zeros(whisper.audio.N_FRAMES, dtype=torch.int32)
        mel_mask[:n_frames] = 1

        if type=="list":
            return {"input_features": mel.tolist(), "attention_mask": mel_mask.tolist()}
        else:
            return {"input_features": mel, "attention_mask": mel_mask}

    def _encode_wavlm_feature(self, signal_array, type="tensor"):
        wavlm_input = self.wavlm_processor(signal_array, sampling_rate=self.audio_encoder_sample_rate, return_attention_mask=True)

        if type=="list":
            return {"input_values": wavlm_input.input_values[0].tolist(), "attention_mask": wavlm_input.attention_mask[0].tolist()}
        else:
            return {"input_values": wavlm_input.input_values[0], "attention_mask": wavlm_input.attention_mask[0]}

    def _encode_mimi_codes(self, audio_array):
        signal_data = torch.FloatTensor(audio_array).view(1, 1, -1).to(self.device)
        hop = self.audio_tokenizer_sample_rate * 18
        step = 0
        codes_tensor = []
        with torch.inference_mode():
            for offset in range(0, signal_data.shape[-1], hop):
                offset_l = max(offset-step, 0)
                offset_r = max(offset+hop+step, 0)
                codes = self.audio_tokenizer.encode(signal_data[:, :, offset_l: offset_r])[0]
                codes_tensor.append(codes)
            codes_tensor = torch.cat(codes_tensor, dim=-1)
        return codes_tensor

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        legacy_format: Optional[bool] = None,
        filename_prefix: Optional[str] = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> Tuple[str]:
        if os.path.isfile(save_directory):
            return

        os.makedirs(save_directory, exist_ok=True)

        tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)

        # Let's save the init kwargs
        target_keys = set(self.init_kwargs.keys())
        # Let's save the special tokens map (only the strings)
        target_keys.update(["model_max_length", "clean_up_tokenization_spaces"])

        for k in target_keys:
            if hasattr(self, k):
                tokenizer_config[k] = getattr(self, k)

        tokenizer_config.update({
            "name_or_path": "avatar-tokenizer",
            "add_prefix_space": False,
            "use_fast": False,
            "device": self.device,
            "chat_template": self.chat_template,
            "bos_token": self.bos_token
        })

        if getattr(self, "audio_special_token", None):
            tokenizer_config.update({
                "short_wait_string": self.short_wait_string,
                "long_wait_string": self.long_wait_string,
                "audio_tokenizer": self.audio_tokenizer_type,
                "text_duration_token": self.text_duration_token,
                "audio_special_token": self.audio_special_token,
            })

        if getattr(self, "audio_downsample_layer", None):
            tokenizer_config.update({
                "audio_downsample_layer": self.audio_downsample_layer,
                "audio_encoder_sample_rate": self.audio_encoder_sample_rate,
                "audio_encoder_mel_size": self.audio_encoder_mel_size,
            })

        tokenizer_class = self.__class__.__name__
        # Remove the Fast at the end unless we have a special `PreTrainedTokenizerFast`
        if tokenizer_class.endswith("Fast") and tokenizer_class != "PreTrainedTokenizerFast":
            tokenizer_class = tokenizer_class[:-4]
        tokenizer_config["tokenizer_class"] = tokenizer_class

        if getattr(self, "_auto_map", None) is not None:
            tokenizer_config["auto_map"] = self._auto_map

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=tokenizer_config)

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)

        file_names = (tokenizer_config_file, )

        return file_names

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

        batch_outputs = {
            "input_ids": [], "attention_mask": [], "text_labels": [],
            "audio_features": [], "audio_attention_mask": [],
            "wavlm_features": [], "wavlm_attention_mask": [],
            "audio_positions": [],
            "valid_tokens_pos": [], "encoder_decoder_attention_mask": [],
            "decoder_input_ids": [], "decoder_attention_mask": [], "decoder_labels": [],
        }

        # text
        for key in ["input_ids", "attention_mask", "text_labels"]:
            values = encoded_inputs[key]
            if "input_ids" == key:
                pad_token = self.text_tokenizer.pad_token_id
            elif "attention_mask" == key:
                pad_token = 0
            elif "text_labels" == key:
                pad_token = -100
            else:
                raise ValueError(f"Invalid key: {key}")

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

        # audio input
        if getattr(self, "audio_downsample_layer", None):
            for batch_idx, batch_values in enumerate(encoded_inputs["audio_features"]):
                if batch_values is None:
                    continue

                for value, pos in zip(batch_values, encoded_inputs["audio_positions"][batch_idx]):
                    # pos
                    batch_outputs["audio_positions"].append([batch_idx]+pos)

                    # audio
                    batch_outputs["audio_features"].append(value["whisper_input"]["input_features"])
                    batch_outputs["audio_attention_mask"].append(value["whisper_input"]["attention_mask"])

                    # wavlm
                    pad_len = self.audio_encoder_sample_rate*30-len(value["wavlm_input"]["input_values"])
                    batch_outputs["wavlm_features"].append(value["wavlm_input"]["input_values"]+[0.0]*pad_len)
                    batch_outputs["wavlm_attention_mask"].append(value["wavlm_input"]["attention_mask"]+[0]*pad_len)

        # audio output
        if getattr(self, "audio_special_token", None):
            for key in ["valid_tokens_pos", "encoder_decoder_attention_mask", "decoder_input_ids", "decoder_attention_mask", "decoder_labels"]:
                values = encoded_inputs[key]

                if "decoder_input_ids" == key:
                    pad_token = self.audio_special_token["eoa_token"]
                elif "decoder_labels" == key:
                    pad_token = -100
                else:
                    pad_token = 0

                if key in ("valid_tokens_pos", "decoder_attention_mask"):
                    try:
                        max_length = max([len(item) for item in values if item])
                    except:
                        continue
                    for value in values:
                        if value is None and key == "decoder_attention_mask":
                            value = [1 for _ in range(max_length)]
                        elif value is None:
                            value = []

                        difference = max_length - len(value)
                        if padding_side == "right":
                            outputs = value + [pad_token] * difference
                        elif padding_side == "left":
                            outputs = [pad_token] * difference + value
                        else:
                            raise ValueError(f"Invalid padding strategy: {padding_side}")

                        batch_outputs[key].append(outputs)
                elif key in ("decoder_input_ids", "decoder_labels",):
                    try:
                        max_length = max([len(item[0]) for item in values if item])
                    except:
                        continue
                    for layer_idx, value in enumerate(values):
                        if value is None:
                            value = [[] for _ in range(self.code_layer)]

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
                    try:
                        max_length = [
                            max([len(item) for item in values if item]),
                            max([len(item[0]) for item in values if item]),
                        ]
                    except:
                        continue
                    for value in values:
                        if value is not None:
                            outputs = torch.zeros(max_length, dtype=torch.long)
                            outputs[:len(value), :len(value[0])] = torch.LongTensor(value)
                        else:
                            outputs = torch.ones(max_length, dtype=torch.long)
                        batch_outputs[key].append(outputs.tolist())
                else:
                    raise ValueError(f"Invalid key padding strategy: {key}")

        # remove empty field
        for key in list(batch_outputs.keys()):
            if batch_outputs[key]:
                continue
            
            del batch_outputs[key]

        return BatchEncoding(batch_outputs, tensor_type=return_tensors)

    def encode(
        self,
        text: str=None,
        audio_signal=None,
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

        if audio_signal:
            if isinstance(audio_signal, list):
                codes = []
                for signal in audio_signal:
                    if signal["split"] == self.long_wait_string:
                        split_token = self.audio_special_token["long_wait_token"]
                    elif signal["split"] == self.short_wait_string:
                        split_token = self.audio_special_token["short_wait_token"]
                    else:
                        split_token = None

                    sub_codes = self._encode_mimi_codes(signal["array"])
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
                            codes[idx] = [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]]
                        else:
                            codes[idx] = [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]]
            else:
                codes = self._encode_mimi_codes(audio_signal["array"])
                codes = codes.tolist()
                for idx in range(len(codes)):
                    if add_audio_special_tokens:
                        if idx == 0:
                            codes[idx] = [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]]
                        else:
                            codes[idx] = [self.audio_special_token["boa_token"]] + codes[idx] + [self.audio_special_token["eoa_token"]]

            return (text_token_ids, codes)
        else:
            return text_token_ids

    def encode_audio_feature(self, elem: dict):
        assert elem["type"] == "audio", f"The type of input element ({elem}) must be `audio`."
        signal_array = np.array(elem["array"], dtype=np.float32)

        whisper_input = self._encode_whisper_feature(signal_array, type="list")
        wavlm_input = self._encode_wavlm_feature(signal_array, type="list")

        audio_length = get_1dconv_out_seq_lens(
            self.audio_downsample_layer,
            get_audio_encoder_out_seq_lens(len(signal_array))
        )

        return audio_length, {"whisper_input": whisper_input, "wavlm_input": wavlm_input}

    def decode(
        self,
        audio_codes: Optional[Union[List, torch.LongTensor]],
    ):
        with torch.no_grad():
            if isinstance(audio_codes, list):
                audio_codes = torch.LongTensor(audio_codes).to(self.device)
                audio_codes = audio_codes.view(1, audio_codes.size(0), audio_codes.size(-1))
            audio = self.audio_tokenizer.decode(audio_codes)
        return audio

    def convert_t2a_attention_mask(self, text_length: int, audio_length: int, remove_assert=False):
        attention_mask = torch.zeros([audio_length, text_length])
        text_token_threshold = self.text_duration_token
        for audio_idx in range(audio_length):
            attention_mask[audio_idx][:text_token_threshold] = 1
            text_token_threshold += 1

        if not remove_assert:
            assert text_token_threshold >= text_length, "audio_tokens need cover text_tokens!"

        return attention_mask.tolist()

    def get_text_token_requirement(self, audio_length: int, offset=1):
        text_token_threshold = 0
        for audio_idx in range(audio_length):
            if audio_idx % self.audio_duration_token == 0:
                text_token_threshold += (self.text_duration_token+offset)
        return text_token_threshold

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.text_tokenizer.convert_ids_to_tokens(i): i for i in range(self.text_tokenizer.vocab_size)}
        vocab.update(self.text_tokenizer.added_tokens_encoder)
        return vocab

    def _get_complete_phrase(self, text_tokens: list[int], text_token_threshold: int):
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

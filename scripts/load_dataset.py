import os
import sys
import logging
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, Tuple

from datasets import Dataset, load_dataset, IterableDataset
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dataset import DataArguments, DatasetAttr, Role, TemplateFeb, get_template_and_fix_tokenizer

IGNORE_INDEX = 0

logger = logging.getLogger(__name__)


def convert_avater_audio(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
        dataset_attr.mask_tag: Role.MASK.value,
    }
    even_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    odd_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag, dataset_attr.mask_tag)
    accept_tags = (even_tags, odd_tags)
    messages = example[dataset_attr.messages]
    if (
        dataset_attr.system_tag
        and len(messages) != 0
        and messages[0][dataset_attr.role_tag].lower() == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = dataset_attr.system if dataset_attr.system else ""

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag].lower() not in accept_tags[turn_idx % 2]:
            print(f"Invalid role tag in {messages}.")
            broken_data = True

        aligned_messages.append(
            {
                "role": tag_mapping[message[dataset_attr.role_tag].lower()],
                "content": message[dataset_attr.content_tag],
                "audios": [
                    {
                        "id": item["id"],
                        "file": data_args.dataset_dir + "/" + item["file"],
                        "split": item["split"],
                    }
                    if "file" in item else
                    item
                    for item in message[dataset_attr.audio_tag]
                ]
            }
        )

        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        print("Skipping this abnormal example.")
        prompt, response = [], []

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": None,
        "_videos": None,
    }
    return output


def _encode_avater_audio_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    train_on_prompt: bool,
    mask_history: bool,
) -> Tuple[List[int], List[int]]:
    if processor is not None and not hasattr(
        processor, "image_seq_length"
    ):  # llava-like models
        prompt[0]["content"] = template.image_token + prompt[0]["content"]

    messages = prompt + response

    text_input_ids, text_labels, valid_tokens_pos = [], [], []
    audio_features, audio_pos = [], []
    audio_codes_ids, audio_codes_labels = [], []
    t2a_attention_mask = []

    prefix_ids = template.encode_system(tokenizer=tokenizer, system=system, tools=tools)
    encoded_pairs = template.encode_avater_audio(tokenizer=tokenizer, prompt_messages=prompt, response_message=response[-1])
    text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
    for turn_idx, (source_dict, target_dict) in enumerate(encoded_pairs):
        # text
        source_token_ids = source_dict["token_ids"]
        target_token_ids = target_dict["token_ids"]

        source_text_len = len(source_token_ids)
        target_text_len = len(target_token_ids)

        source_text_label = [IGNORE_INDEX] * source_text_len

        if turn_idx != len(encoded_pairs) - 1:
            target_text_label = [IGNORE_INDEX] * target_text_len
        elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
            target_text_label = [IGNORE_INDEX] * target_text_len
        else:
            target_text_label = target_token_ids
            valid_tokens_pos = [idx for idx in range(len(text_labels), len(text_labels)+len(target_token_ids))]

        text_input_ids += source_token_ids + target_token_ids
        text_labels += source_text_label + target_text_label

        # audio input
        if "audio_features" in source_dict:
            audio_features += source_dict["audio_features"]
            audio_pos += source_dict["audio_pos"]

        # audio output
        if turn_idx == len(encoded_pairs) - 1 and "audio_codes" in target_dict:
            audio_codes_ids = target_dict["audio_codes"]
            audio_codes_labels = target_dict["audio_codes"]
            for idx in range(len(audio_codes_labels)):
                if idx == 0:
                    audio_codes_labels[idx] = audio_codes_labels[0][:-tokenizer.acoustic_delay] + [IGNORE_INDEX] * tokenizer.acoustic_delay
                else:
                    audio_codes_labels[idx] = [IGNORE_INDEX] * tokenizer.acoustic_delay + audio_codes_labels[0][tokenizer.acoustic_delay:]
            t2a_attention_mask = tokenizer.convert_t2a_attention_mask(target_token_ids, target_dict["audio_codes"])

    assert len(text_input_ids) == len(text_labels), "The length of text_input_ids should equal with labels' length!"
    assert (len(audio_codes_ids) == len(audio_codes_labels) and len(audio_codes_ids[0]) == len(audio_codes_labels[0])), "The length of audio_codes_ids should equal with labels' length!"
    return {
        "prefix_ids": prefix_ids,
        "text_input_ids": text_input_ids, "text_labels": text_labels, "valid_tokens_pos": valid_tokens_pos,
        "audio_features": audio_features, "audio_pos": audio_pos,
        "audio_codes_ids": audio_codes_ids, "audio_codes_labels": audio_codes_labels,
        "t2a_attention_mask": t2a_attention_mask,
    }


def print_avater_audio_dataset_example(
    example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer"
) -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["text_labels"]))
    print("input_ids:\n{}".format(example["input_ids"]), flush=True)
    print(
        "inputs:\n{}".format(
            tokenizer.text_tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        ),
        flush=True,
    )
    print("label_ids:\n{}".format(example["text_labels"]), flush=True)
    print(
        "text_labels:\n{}".format(tokenizer.text_tokenizer.decode(valid_labels, skip_special_tokens=False)),
        flush=True,
    )


def preprocess_avater_audio_dataset(
    examples: Dict[str, List[Any]],
    template: "TemplateFeb",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {
        "input_ids": [], "attention_mask": [], "text_labels": [], "valid_tokens_pos": [],
        "decoder_input_ids": [], "decoder_attention_mask": [], "encoder_decoder_attention_mask": [], "labels": [],
        "images": [], "videos": []
    }
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        enocde_outputs: dict = _encode_avater_audio_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            train_on_prompt=data_args.train_on_prompt,
            mask_history=data_args.mask_history,
        )

        # text encoder
        model_inputs["input_ids"].append(enocde_outputs["prefix_ids"] + enocde_outputs["text_input_ids"])
        model_inputs["attention_mask"].append([1] * (len(enocde_outputs["prefix_ids"]) + len(enocde_outputs["text_input_ids"])))
        model_inputs["text_labels"].append([IGNORE_INDEX] * len(enocde_outputs["prefix_ids"]) + enocde_outputs["text_labels"])
        model_inputs["valid_tokens_pos"].append(enocde_outputs["valid_tokens_pos"])

        # audio encoder

        # tts adapter
        model_inputs["decoder_input_ids"].append(enocde_outputs["audio_codes_ids"])
        model_inputs["decoder_attention_mask"].append([1] * len(enocde_outputs["audio_codes_ids"]))
        model_inputs["encoder_decoder_attention_mask"].append(enocde_outputs["t2a_attention_mask"])
        model_inputs["labels"].append([1] * len(enocde_outputs["audio_codes_ids"]))

        # other
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    stage,
    template: "TemplateFeb",
    tokenizer: Optional[Union["PreTrainedTokenizer", "MimiTokenizer"]],
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func = partial(
        preprocess_avater_audio_dataset,
        template=template,
        tokenizer=tokenizer,
        processor=None,
        data_args=data_args,
    )
    print_function = print_avater_audio_dataset_example

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=1,
            load_from_cache_file=(False),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        preprocess_func,
        batched=True,
        remove_columns=column_names,
        **kwargs,
    )

    if True:
        try:
            print(f"{stage} eval example:" if is_eval else f"{stage} training example:", flush=True)
            print_function(next(iter(dataset)), tokenizer=tokenizer)
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def align_dataset(
    dataset,
    dataset_attr: "DatasetAttr",
    data_args: "DataArguments",
):
    r"""
    Aligned dataset:
        _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
        _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
        _system: "..."
        _tools: "...",
        _images: [],
        _videos: [],
    """
    convert_func = partial(convert_avater_audio, dataset_attr=dataset_attr, data_args=data_args)

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not False:
        kwargs = dict(
            num_proc=16,
            load_from_cache_file=(not False),
            desc="Converting format of dataset",
        )

    return dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )


def load_single_dataset(data_dir, data_files, tokenizer_dir):
    dataset_attr = DatasetAttr(
        load_from="file",
        dataset_name="tts_adapter",
        stage="audio_tts",
        formatting="avater_audio",
    )
    data_args = DataArguments(
        template="llama3",
        streaming=False,
        dataset_dir=data_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=False,
        split_special_tokens=False,
        padding_side="right",
        trust_remote_code=True
    )
    
    template = get_template_and_fix_tokenizer(tokenizer, data_args, is_avater_tokenizer=True)

    ##################################
    dataset = load_dataset(
        path="json",
        name=None,
        data_dir=None,
        data_files=data_files,
        split="train",
        cache_dir=None,
        token=None,
        streaming=False,
        num_proc=16,
        trust_remote_code=False,
    )

    dataset = align_dataset(dataset, dataset_attr, data_args)

    dataset = _get_preprocessed_dataset(
        dataset,
        data_args=data_args,
        stage=dataset_attr.stage,
        template=template,
        tokenizer=tokenizer
    )

    return dataset


if __name__ == "__main__":
    audio_dir = sys.argv[1]
    audio_file = sys.argv[2]
    tokenizer_dir = sys.argv[3]
    dataset = load_single_dataset(audio_dir, [audio_file], tokenizer_dir)

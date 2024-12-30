import os
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union, Tuple

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.tts_adapter.schema import DatasetAttr
from scripts.tts_adapter.schema import Role


def convert_avater_audio(
    example: Dict[str, Any],
    dataset_attr: "DatasetAttr",
    data_args=None,
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
                "audio": message[dataset_attr.audio_tag]
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


def _encode_conversation_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    template: "Template",
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
    input_ids, labels = [], []

    if processor is not None and hasattr(
        processor, "image_seq_length"
    ):  # paligemma models
        image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
        input_ids += [image_token_id] * getattr(processor, "image_seq_length")
        labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

    prefix_ids = template.encode_system(tokenizer=tokenizer, system=system, tools=tools)
    encoded_pairs = template.encode_multiturn(tokenizer=tokenizer, messages=messages, system=None, tools=None)
    text_pairs = [(messages[i], messages[i + 1]) for i in range(0, len(messages), 2)]
    for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
        source_len = len(source_ids)
        target_len = len(target_ids)

        if train_on_prompt:
            source_label = source_ids
        elif turn_idx != 0 and template.efficient_eos:
            source_label = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
        else:
            source_label = [IGNORE_INDEX] * source_len

        if mask_history and turn_idx != len(encoded_pairs) - 1:
            target_label = [IGNORE_INDEX] * target_len
        elif text_pairs[turn_idx][1]["role"] == Role.MASK.value:
            target_label = [IGNORE_INDEX] * target_len
        else:
            target_label = target_ids

        input_ids += source_ids + target_ids
        labels += source_label + target_label

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    assert len(input_ids) == len(labels), "The length of input_ids should equal with labels' length!"
    return prefix_ids, input_ids, labels


def preprocess_conversation_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        system_ids, input_ids, labels = _encode_conversation_example(
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
        model_inputs["input_ids"].append(system_ids + input_ids)
        model_inputs["attention_mask"].append([1] * (len(system_ids) + len(input_ids)))
        model_inputs["labels"].append([IGNORE_INDEX] * len(system_ids) + labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    stage,
    template: "Template",
    tokenizer: Optional[Union["PreTrainedTokenizer", "MimiTokenizer"]],
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""
    Preprocesses the dataset, including format checking and tokenization.
    """
    if dataset is None:
        return None

    preprocess_func = None
    print_function = None

    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not False:
        kwargs = dict(
            num_proc=16,
            load_from_cache_file=(not False),
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
            print_function(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def align_dataset(
    dataset,
    dataset_attr: "DatasetAttr"
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
    convert_func = partial(convert_avater_audio, dataset_attr=dataset_attr, data_args=None)

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


def load_single_dataset(data_files, tokenizer_dir):
    dataset_attr = DatasetAttr(
        load_from="file",
        dataset_name="tts_adapter",
        stage="audio_tts",
        formatting="avater_audio"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=False,
        split_special_tokens=False,
        padding_side="right",
        trust_remote_code=True
    )

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

    dataset = align_dataset(dataset, dataset_attr)

    dataset = _get_preprocessed_dataset(
        dataset, stage=dataset_attr.stage, template=None, tokenizer=tokenizer
    )

    return dataset


if __name__ == "__main__":
    audio_file = sys.argv[1]
    tokenizer_dir = sys.argv[2]
    dataset = load_single_dataset([audio_file], tokenizer_dir)

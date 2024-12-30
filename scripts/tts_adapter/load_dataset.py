import os
import sys
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments
from mimi import MimiTokenizer

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
    
    tokenizer = MimiTokenizer.load_from_checkpoint(cpt_dir=tokenizer_dir)

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

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence


@dataclass
class DatasetAttr:
    r"""
    Dataset attributes.
    """

    # basic configs
    load_from: Literal["hf_hub", "ms_hub", "om_hub", "script", "file"]
    dataset_name: str
    stage: Literal["pretrain", "conversation", "instruction", "audio_tts"] = "conversation"
    formatting: Literal["alpaca", "sharegpt", "document", "avater_audio"] = "sharegpt"
    ranking: bool = False

    # extra configs
    subset: Optional[str] = None
    split: str = "train"
    folder: Optional[str] = None
    samples_ratio: Optional[float] = None

    # common columns
    system: Optional[str] = None
    tools: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None

    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    kto_tag: Optional[str] = None

    # alpaca columns
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None

    # sharegpt columns
    messages: Optional[str] = "conversations"

    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "user"
    assistant_tag: Optional[str] = "assistant"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    mask_tag: Optional[str] = "mask"

    # document columns
    prefix: Optional[str] = "prefix_text"
    document: Optional[str] = "document"

    def __repr__(self) -> str:
        return self.dataset_name

    def set_attr(self, key: str, obj: Dict[str, Any], default: Optional[Any] = None) -> None:
        setattr(self, key, obj.get(key, default))

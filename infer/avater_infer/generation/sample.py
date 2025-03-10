import torch
from typing import TYPE_CHECKING, Optional, Union

from transformers.generation.utils import GenerateNonBeamOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.utils import logging

from .sample_llm import sample_llm_tokens


if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer


logger = logging.get_logger(__name__)



def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    return sample_llm_tokens(
        self, input_ids,
        logits_processor, stopping_criteria,
        generation_config, synced_gpus, streamer,
        **model_kwargs,
    )

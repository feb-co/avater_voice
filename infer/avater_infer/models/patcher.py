from functools import partial

from transformers.generation.logits_process import TopPLogitsWarper
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerationMixin

from ..generation.sample import sample


def patch_model(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    # param
    model.config.is_encoder_decoder = True

    # function
    return model, tokenizer


def patch_init(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    GenerationMixin._sample = sample

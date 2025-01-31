from functools import partial

from transformers.generation.logits_process import TopPLogitsWarper
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation import GenerationMixin

from ..generation.samples import sample


def patch_model(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    # param
    model.config.is_encoder_decoder = True
    model.generation_config.decoder_start_token_id = model.config.boa_token_id
    model.generation_config.output_hidden_states = True

    # function
    return model, tokenizer


def patch_init(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer"):
    GenerationMixin._sample = sample

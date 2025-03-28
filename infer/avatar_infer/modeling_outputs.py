
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch


from transformers.utils import ModelOutput


@dataclass
class AdapterModelOutputWithPastAndCrossAttentions(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class Seq2SeqCausalLMOutputWithCrossAttentions(ModelOutput):
    """
    Base class for sequence-to-sequence causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss (for next-token prediction) when labels are provided.
        encoder_logits (`torch.FloatTensor`):
            Prediction scores from the encoder language modeling head (scores for each vocabulary token before SoftMax).
        encoder_past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Cached key and value states in the encoder.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states output from each layer of the encoder.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights from each layer of the encoder.
        decoder_logits (`List[torch.FloatTensor]`):
            Prediction scores from the decoder language modeling head.
        decoder_past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
            Cached key and value states in the decoder.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states output from each layer of the decoder.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Self-attention weights from each layer of the decoder.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*):
            Cross-attention weights from each layer of the decoder.
    """
    loss: Optional[torch.FloatTensor] = None
    encoder_logits: torch.FloatTensor = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_logits: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
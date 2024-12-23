"""PyTorch LLaMA TTS model."""

import os
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging,
    ModelOutput
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_utils import (
    _add_variant,
    get_checkpoint_shard_files,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    PreTrainedModel
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
    ACT2FN,
    LlamaForCausalLM,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaConfig
)
from transformers.models.bart.modeling_bart import (
    _prepare_4d_causal_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_attention_mask
)

from .configuration_llama_tts import LlamaTTSConfig


logger = logging.get_logger(__name__)


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
        decoder_logits (`torch.FloatTensor`):
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
    encoder_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_logits: torch.FloatTensor = None
    decoder_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def get_archive_file(
    pretrained_model_name_or_path,
    subfolder="",
    use_safetensors=True,
    variant=None,
):
    if use_safetensors is not False and os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
    ):
        # Load from a safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
        )
    elif use_safetensors is not False and os.path.isfile(
        os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
        )
    ):
        # Load from a sharded safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
        )
        is_sharded = True
    elif not use_safetensors and os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
    ):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
        )
    elif not use_safetensors and os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
    ):
        # Load from a sharded PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
        )
        is_sharded = True
    else:
        raise ValueError(
            "we don't support other type 'archive_file' now!"
        )
    
    return archive_file, is_sharded


def apply_cross_attn_rotary_pos_emb(q, k, cos, sin, e_cos, e_sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    e_cos = e_cos.unsqueeze(unsqueeze_dim)
    e_sin = e_sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * e_cos) + (rotate_half(k) * e_sin)
    return q_embed, k_embed


class AdapterAudioEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class TTSAdapterMLP(nn.Module):
    def __init__(self, config: LlamaTTSConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.tts_adapter_hidden_size
        self.intermediate_size = config.tts_adapter_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class TTSAdapterAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaTTSConfig, layer_idx: Optional[int] = None, encoder_attn=False, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.encoder_attn = encoder_attn
        self.is_causal = is_causal
        
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.tts_adapter_hidden_size
        self.attention_dropout = config.tts_adapter_attention_dropout
        self.num_heads = config.tts_adapter_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = self.num_heads // 4
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            config.hidden_size if self.encoder_attn else self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size if self.encoder_attn else self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int, num_heads: int):
        return tensor.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, tgt_len, _ = hidden_states.size()

        # Proj Q,K,V based on past_key_value
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz, self.num_heads)
        if self.encoder_attn:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz, self.num_key_value_heads)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.num_key_value_heads)

        # Rotary Embedding
        if self.encoder_attn:
            cos, sin = position_embeddings
            e_cos, e_sin = encoder_position_embeddings
            query_states, key_states = apply_cross_attn_rotary_pos_emb(query_states, key_states, cos, sin, e_cos, e_sin)
            cache_kwargs = {"sin": sin, "cos": cos, "e_sin": e_sin, "e_cos": e_cos, "cache_position": cache_position}
        else:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Past Key Value
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Attn
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        src_len = key_states.size(2)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class TTSAdapterFlashAttention2(TTSAdapterAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False
        bsz, q_len, _ = hidden_states.size()

        # Proj Q,K,V based on past_key_value
        query_states = self._shape(self.q_proj(hidden_states), q_len, bsz, self.num_heads)
        if self.encoder_attn:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz, self.num_key_value_heads)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.num_key_value_heads)

        # Rotary Embedding
        if self.encoder_attn:
            cos, sin = position_embeddings
            e_cos, e_sin = encoder_position_embeddings
            query_states, key_states = apply_cross_attn_rotary_pos_emb(query_states, key_states, cos, sin, e_cos, e_sin)
            cache_kwargs = {"sin": sin, "cos": cos, "e_sin": e_sin, "e_cos": e_cos, "cache_position": cache_position}
        else:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Past Key Value
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class TTSAdapterSdpaAttention(TTSAdapterAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "TTSModel is using TTSAdapterFlashAttention2, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` not None. Falling back to the manual attention"
                ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                key_value_states=key_value_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                encoder_position_embeddings=encoder_position_embeddings,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position
            )
        
        bsz, tgt_len, _ = hidden_states.size()

        # Proj Q,K,V based on past_key_value
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz, self.num_heads)
        if self.encoder_attn:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz, self.num_key_value_heads)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.num_key_value_heads)

        # Rotary Embedding
        if self.encoder_attn:
            cos, sin = position_embeddings
            e_cos, e_sin = encoder_position_embeddings
            query_states, key_states = apply_cross_attn_rotary_pos_emb(query_states, key_states, cos, sin, e_cos, e_sin)
            cache_kwargs = {"sin": sin, "cos": cos, "e_sin": e_sin, "e_cos": e_cos, "cache_position": cache_position}
        else:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Past Key Value
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Attn
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
        is_causal = True if self.is_causal and attention_mask is None and tgt_len > 1 else False

        # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
        # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value


TTS_ADAPTER_ATTENTION_CLASSES = {
    "eager": TTSAdapterAttention,
    "flash_attention_2": TTSAdapterFlashAttention2,
    "sdpa": TTSAdapterSdpaAttention,
}


class TTSAdapterLayer(nn.Module):
    def __init__(self, config: LlamaTTSConfig, layer_idx: int):
        super().__init__()
        self.dropout = config.tts_adapter_dropout
        self.embed_dim = config.tts_adapter_hidden_size

        self.self_attn = TTS_ADAPTER_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.self_attn_layer_norm = LlamaRMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        self.encoder_attn = TTS_ADAPTER_ATTENTION_CLASSES[config._attn_implementation if config._attn_implementation != "flash_attention_2" else "eager"](
            config=config,
            layer_idx=layer_idx,
            encoder_attn=True,
        )
        self.encoder_attn_layer_norm = LlamaRMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        self.mlp = TTSAdapterMLP(config)
        self.final_layer_norm = LlamaRMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        position_embeddings: Optional[torch.Tensor] = None,
        encoder_position_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        # Self Attention
        residual = hidden_states

        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states

        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            encoder_position_embeddings=encoder_position_embeddings,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value, cross_attn_present_key_value)

        return outputs


class LlamaTTSPreTrainedModel(PreTrainedModel):
    config_class = LlamaTTSConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "TTSAdapterLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class TTSAdapter(LlamaTTSPreTrainedModel):
    _tied_weights_keys = ["embed_tokens.weight", "adapter_head.weight"]

    def __init__(self, config: LlamaTTSConfig):
        super().__init__(config)
        self.config = config
        self.dropout = config.tts_adapter_dropout
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"
        embed_scale = math.sqrt(config.tts_adapter_hidden_size) if config.scale_embedding else 1.0

        self.embed_tokens = AdapterAudioEmbedding(
            config.audio_vocab_size, config.tts_adapter_hidden_size, self.padding_idx, embed_scale=embed_scale
        )
        self.layers = nn.ModuleList(
            [TTSAdapterLayer(config, layer_idx) for layer_idx in range(config.tts_adapter_hidden_layers)]
        )
        self.adapter_head = nn.Linear(config.tts_adapter_hidden_size, config.audio_vocab_size, bias=False)

        self.norm = LlamaRMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        rope_embedding_config = LlamaConfig(
            rope_theta=config.rope_theta,
            partial_rotary_factor=config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0,
            hidden_size=config.tts_adapter_hidden_size,
            num_attention_heads=config.tts_adapter_attention_heads,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=rope_embedding_config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def _tie_weights(self):
        if self.config.tie_audio_embeddings:
            self._tie_or_clone_weights(self.adapter_head, self.embed_tokens)

    def _update_adapter_attention_mask(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
    ):
        input_shape = inputs_embeds.size()[:-1]

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                input_shape,
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_flash_attention_2:
                encoder_attention_mask = encoder_attention_mask if 0 in encoder_attention_mask else None
            elif self._use_sdpa and not output_attentions and len(encoder_attention_mask.size())==2:
                # output_attentions=True can not be supported when using SDPA, and we fall back on
                # the manual implementation that requires a 4D causal mask in all cases.
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask,
                    inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif len(encoder_attention_mask.size())==2:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                encoder_attention_mask = _prepare_4d_attention_mask(
                    encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            else:
                encoder_attention_mask = 1.0 - encoder_attention_mask
                encoder_attention_mask = encoder_attention_mask.masked_fill(encoder_attention_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)

        return attention_mask, encoder_attention_mask

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, AdapterModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # expand cache
        return_legacy_cache = False
        if (
            use_cache and not isinstance(past_key_values, Cache) and not self.training
        ):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # expand posistion
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)
        encoder_position_ids = torch.arange(0, encoder_hidden_states.shape[1], device=encoder_hidden_states.device).unsqueeze(0)

        # expand attention mask
        attention_mask, encoder_attention_mask = self._update_adapter_attention_mask(
            inputs_embeds, past_seen_tokens, encoder_hidden_states, attention_mask, encoder_attention_mask, output_attentions
        )

        # dropout
        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        encoder_position_embeddings = self.rotary_emb(encoder_hidden_states, encoder_position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    output_attentions,
                    use_cache,
                    position_embeddings,
                    encoder_position_embeddings
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    encoder_position_embeddings=encoder_position_embeddings
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()
        
        # Logits
        logits = self.adapter_head(hidden_states)
        logits = logits.float()

        if not return_dict:
            return tuple(
                v
                for v in [logits, hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        return AdapterModelOutputWithPastAndCrossAttentions(
            logits=logits,
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class LlamaTTS(LlamaTTSPreTrainedModel):
    def __init__(self, config: LlamaTTSConfig):
        super().__init__(config)
        
        self.config = config
        self.audio_vocab_size = config.audio_vocab_size

        self.llm = LlamaForCausalLM(config)
        self.tts_adapter = TTSAdapter(config)

        # Initialize weights and apply final processing
        self.post_init()
        
        # frozen llm
        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

    def __repr__(self):
        return self.tts_adapter.__repr__()
    
    def load_llm_state_dict(self, llm_pretrained_model_name_or_path):
        archive_file, is_sharded = get_archive_file(llm_pretrained_model_name_or_path)
        if is_sharded:
            archive_file, sharded_metadata = get_checkpoint_shard_files(
                llm_pretrained_model_name_or_path,
                archive_file,
            )

        state_dict = None

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_state_dict_keys = list(state_dict.keys())

        (
            self.llm,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = self._load_pretrained_model(
            self.llm,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            archive_file,
            llm_pretrained_model_name_or_path
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        encoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Seq2SeqCausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs: CausalLMOutputWithPast = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=encoder_past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )

        decoder_outputs: AdapterModelOutputWithPastAndCrossAttentions = self.tts_adapter(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.hidden_states[-1],
            encoder_attention_mask=encoder_decoder_attention_mask,
            past_key_values=decoder_past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Loss
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = decoder_outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.audio_vocab_size)
            shift_labels = shift_labels.view(-1)

            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return Seq2SeqCausalLMOutputWithCrossAttentions(
            loss=loss,
            encoder_logits=encoder_outputs.logits,
            encoder_past_key_values=encoder_outputs.past_key_values,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_logits=decoder_outputs.logits,
            decoder_past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

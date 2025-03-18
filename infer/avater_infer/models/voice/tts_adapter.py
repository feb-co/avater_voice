"""PyTorch Avater TTS adapter."""

import math
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    ACT2FN,
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

from ...modeling_outputs import AdapterModelOutputWithPastAndCrossAttentions
from .configuration_voice import AvaterVoiceConfig
from ...cache_utils import AvaterCache


logger = logging.get_logger(__name__)



def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class AdapterAudioEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class AdapterHead(nn.Module):
    def __init__(self, config: AvaterVoiceConfig):
        super().__init__()
        self.head_block = nn.ModuleList([])
        for _ in range(config.block_size):
            self.head_block.append(nn.Linear(config.tts_adapter_hidden_size, config.audio_vocab_size//config.block_size, bias=False))

    def forward(self, block_idx, hidden_states: torch.Tensor):
        return self.head_block[block_idx](hidden_states)


class TTSAdapterMLP(nn.Module):
    def __init__(self, config: AvaterVoiceConfig):
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

    def __init__(self, config: AvaterVoiceConfig, layer_idx: Optional[int] = None, block_idx: Optional[int] = None, encoder_attn=False, is_causal: bool = False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.encoder_attn = encoder_attn
        self.is_causal = is_causal
        self.num_key_value_groups = 4

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
        self.num_key_value_heads = self.num_heads // self.num_key_value_groups
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
        past_key_values: Optional[AvaterCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        attn_idx = self.layer_idx + (self.block_idx * self.config.tts_adapter_hidden_layers)
        bsz, tgt_len, _ = hidden_states.size()

        # Proj Q,K,V based on past_key_values
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz, self.num_heads)
        if self.encoder_attn:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz, self.num_key_value_heads)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.num_key_value_heads)

        # Rotary Embedding
        if not self.encoder_attn:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Past Key Value
        if past_key_values is not None:
            if self.encoder_attn:
                key_states, value_states = past_key_values.cross_attention_cache.update(key_states, value_states, attn_idx, {})
            elif not self.encoder_attn:
                key_states, value_states = past_key_values.self_attention_cache.update(key_states, value_states, attn_idx, cache_kwargs)

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

        return attn_output, attn_weights, past_key_values


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
        past_key_values: Optional[AvaterCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False
        attn_idx = self.layer_idx + (self.block_idx * self.config.tts_adapter_hidden_layers)
        bsz, tgt_len, _ = hidden_states.size()

        # Proj Q,K,V based on past_key_values
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz, self.num_heads)
        if self.encoder_attn:
            key_states = self._shape(self.k_proj(key_value_states), tgt_len, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(key_value_states), tgt_len, bsz, self.num_key_value_heads)
        else:
            key_states = self._shape(self.k_proj(hidden_states), tgt_len, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(hidden_states), tgt_len, bsz, self.num_key_value_heads)

        # Rotary Embedding
        if not self.encoder_attn:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Past Key Value
        if past_key_values is not None:
            if self.encoder_attn:
                key_states, value_states = past_key_values.cross_attention_cache.update(key_states, value_states, attn_idx, {})
            elif not self.encoder_attn:
                key_states, value_states = past_key_values.self_attention_cache.update(key_states, value_states, attn_idx, cache_kwargs)

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
            tgt_len,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, tgt_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class TTSAdapterSdpaAttention(TTSAdapterAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[AvaterCache] = None,
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
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position
            )

        attn_idx = self.layer_idx + (self.block_idx * self.config.tts_adapter_hidden_layers)
        bsz, tgt_len, _ = hidden_states.size()

        # Proj Q,K,V based on past_key_values
        query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz, self.num_heads)
        if self.encoder_attn:
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz, self.num_key_value_heads)
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz, self.num_key_value_heads)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz, self.num_key_value_heads)

        # Rotary Embedding
        if not self.encoder_attn:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Past Key Value
        if past_key_values is not None:
            if self.encoder_attn:
                key_states, value_states = past_key_values.cross_attention_cache.update(key_states, value_states, attn_idx, {})
            elif not self.encoder_attn:
                key_states, value_states = past_key_values.self_attention_cache.update(key_states, value_states, attn_idx, cache_kwargs)

        # Attn
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

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
        
        return attn_output, None, past_key_values


TTS_ADAPTER_ATTENTION_CLASSES = {
    "eager": TTSAdapterAttention,
    "flash_attention_2": TTSAdapterFlashAttention2,
    "sdpa": TTSAdapterSdpaAttention,
}


class TTSAdapterBlock(nn.Module):
    def __init__(self, config: AvaterVoiceConfig, block_idx: int, layer_idx: int):
        super().__init__()
        self.block_idx = block_idx
        self.dropout = config.tts_adapter_dropout
        self.embed_dim = config.tts_adapter_hidden_size

        self.self_attn = TTS_ADAPTER_ATTENTION_CLASSES[config._attn_implementation](
            config=config,
            layer_idx=layer_idx,
            block_idx=block_idx,
            is_causal=True,
        )
        self.self_attn_layer_norm = LlamaRMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        if block_idx == 0:
            self.encoder_attn = TTS_ADAPTER_ATTENTION_CLASSES[config._attn_implementation if config._attn_implementation != "flash_attention_2" else "sdpa"](
                config=config,
                layer_idx=layer_idx,
                block_idx=block_idx,
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
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        position_embeddings: Optional[torch.Tensor] = None,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        if self.block_idx == 0:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                past_key_values=present_key_value,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
        else:
            cross_attn_weights = None

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, self_attn_weights, cross_attn_weights, present_key_value


class TTSAdapterLayer(nn.Module):
    def __init__(self, config: AvaterVoiceConfig, layer_idx: int):
        super().__init__()
        self.block = nn.ModuleList([])
        for block_idx in range(config.block_size):
            self.block.append(TTSAdapterBlock(config, block_idx, layer_idx))

    def forward(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        position_embeddings: Optional[torch.Tensor] = None,
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
        hidden_states, self_attn_weights, cross_attn_weights, present_key_value = self.block[block_idx](
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            output_attentions,
            use_cache,
            position_embeddings
        )

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class AvaterTTSPreTrainedModel(PreTrainedModel):
    config_class = AvaterVoiceConfig
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


class TTSAdapter(AvaterTTSPreTrainedModel):
    _tied_weights_keys = ["embed_tokens.weight", "adapter_head.weight"]

    def __init__(self, config: AvaterVoiceConfig):
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
        # self.adapter_head = nn.Linear(config.tts_adapter_hidden_size, config.audio_vocab_size, bias=False)
        self.adapter_head = AdapterHead(config)

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

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the attention_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(inputs_embeds.dtype).min
            attention_mask = AttentionMaskConverter._unmask_unattended(attention_mask, min_dtype)

        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            if self._use_sdpa and not output_attentions and len(encoder_attention_mask.size())==2:
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
            elif len(encoder_attention_mask.size())==3:
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1).to(inputs_embeds)
                encoder_attention_mask = 1.0 - encoder_attention_mask
                encoder_attention_mask = encoder_attention_mask.masked_fill(encoder_attention_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
            else:
                encoder_attention_mask = encoder_attention_mask.to(inputs_embeds)
                encoder_attention_mask = 1.0 - encoder_attention_mask
                encoder_attention_mask = encoder_attention_mask.masked_fill(encoder_attention_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)

        if (
            self.config._attn_implementation in ("sdpa", "flash_attention_2")
            and encoder_attention_mask is not None
            and encoder_attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the encoder_attention_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(inputs_embeds.dtype).min
            encoder_attention_mask = AttentionMaskConverter._unmask_unattended(encoder_attention_mask, min_dtype)

        return attention_mask, encoder_attention_mask

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward_block(
        self,
        block_idx,
        position_embeddings,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        past_key_values,
        all_hidden_states,
        output_hidden_states,
        all_self_attns,
        all_cross_attentions,
        output_attentions,
        use_cache
    ):
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    block_idx,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                    output_attentions,
                    use_cache,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    block_idx,
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[3 if output_attentions else 1]
            else:
                next_decoder_cache = None

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return hidden_states, next_decoder_cache

    def forward(
        self,
        input_ids: List[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[AvaterCache, List[torch.FloatTensor]]] = None,
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
            pos_bias = (
                torch.arange(0, self.config.code_layers)
                *
                (self.config.code_size+self.config.audio_special_tokens)
            ).view(1, self.config.code_layers, 1).to(input_ids)
            inputs_embeds = self.embed_tokens(input_ids+pos_bias)
            inputs_embeds = inputs_embeds.sum(dim=1)

        # expand cache
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        # expand posistion
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        # pos embedding
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # dropout
        inputs_embeds = nn.functional.dropout(inputs_embeds, p=self.dropout, training=self.training)

        # expand attention mask
        attention_mask, encoder_attention_mask = self._update_adapter_attention_mask(
            inputs_embeds, past_seen_tokens, encoder_hidden_states, attention_mask, encoder_attention_mask, output_attentions
        )

        # decoder layers
        logits = []
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        hidden_states = inputs_embeds
        for block_idx in range(self.config.block_size):
            residual = hidden_states

            hidden_states, next_decoder_cache = self.forward_block(
                block_idx=block_idx,
                position_embeddings=position_embeddings,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                all_hidden_states=all_hidden_states,
                output_hidden_states=output_hidden_states,
                all_self_attns=all_self_attns,
                all_cross_attentions=all_cross_attentions,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

            # Logits
            logits.append(self.adapter_head(block_idx, hidden_states))

            # residual
            hidden_states = residual.to(hidden_states) + hidden_states

        next_cache = next_decoder_cache if use_cache else None

        logits = torch.cat(logits, dim=-1)
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

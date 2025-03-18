"""Inference-only TTS Adapter model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import torch
from torch import nn

from vllm.attention import Attention, AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear, 
    RowParallelLinear,
    ColumnParallelLinear
)
from vllm.model_executor.models import SupportsLoRA, SupportsPP
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


from avater_infer.models.voice.configuration_voice import AvaterVoiceConfig


class TTSAdapterHead(nn.Module):
    def __init__(
        self,
        config: AvaterVoiceConfig,
        lora_config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vocab_size = config.audio_vocab_size//config.block_size
        self.unpadded_vocab_size = self.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        self.head_block = nn.ModuleList([])
        for idx in range(config.block_size):
            self.head_block.append(
                ParallelLMHead(
                    self.unpadded_vocab_size,
                    config.tts_adapter_hidden_size,
                    org_num_embeddings=self.vocab_size,
                    bias=False,
                    padding_size=(
                        DEFAULT_VOCAB_PADDING_SIZE
                        # We need bigger padding if using lora for kernel
                        # compatibility
                        if not lora_config else
                        lora_config.lora_vocab_padding_size),
                    quant_config=quant_config,
                    prefix=f"{prefix}.head_block.{idx}",
                )
            )

    def forward(self, block_idx, hidden_states: torch.Tensor):
        return self.head_block[block_idx](hidden_states)


class TTSAdapterMLP(nn.Module):
    def __init__(
        self,
        config: AvaterVoiceConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.tts_adapter_hidden_size
        self.intermediate_size = config.tts_adapter_intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            bias=config.mlp_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class TTSAdapterSelfAttention(nn.Module):
    def __init__(
        self,
        config: AvaterVoiceConfig,
        layer_idx: Optional[int] = None,
        block_idx: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = ""
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.config = config
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.num_key_value_groups = 4

        self.hidden_size = config.tts_adapter_hidden_size
        self.total_num_heads = config.tts_adapter_attention_heads
        self.total_num_kv_heads = self.total_num_heads // self.num_key_value_groups
        self.head_dim = self.hidden_size // self.total_num_heads

        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # Phi models introduced a partial_rotary_factor parameter in the config
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type == "llama":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=config.rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            interleaved_sliding_window = config.interleaved_sliding_window
            if isinstance(interleaved_sliding_window, int):
                sliding_window = interleaved_sliding_window
            elif isinstance(interleaved_sliding_window, list):
                sw_idx = layer_idx % len(interleaved_sliding_window)
                sliding_window = interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(
                    f"{type(interleaved_sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class TTSAdapterCrossAttention(nn.Module):
    def __init__(
        self,
        config: AvaterVoiceConfig,
        layer_idx: Optional[int] = None,
        block_idx: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.config = config
        self.layer_idx = layer_idx
        self.block_idx = block_idx
        self.num_key_value_groups = 4

        self.hidden_size = config.tts_adapter_hidden_size
        self.total_num_heads = config.tts_adapter_attention_heads
        self.total_num_kv_heads = self.total_num_heads // self.num_key_value_groups
        self.head_dim = self.hidden_size // self.total_num_heads

        assert self.total_num_heads % tp_size == 0
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.kv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        # (afeldman-nm 2024/07/22) TODO:
        # Need a more efficient solution for q/k/v
        qkv_dec, _ = self.qkv_proj(decoder_hidden_states)
        q, _, _ = qkv_dec.split([self.q_size, self.kv_size, self.kv_size],
                                dim=-1)
        if encoder_hidden_states is None:
            k = None
            v = None
        else:
            qkv_enc, _ = self.qkv_proj(encoder_hidden_states)
            _, k, v = qkv_enc.split([self.q_size, self.kv_size, self.kv_size],
                                    dim=-1)

        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        output, _ = self.out_proj(attn_output)
        return output


class TTSAdapterBlock(nn.Module):
    def __init__(
        self,
        config: AvaterVoiceConfig,
        block_idx: int, layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ):
        super().__init__()
        self.block_idx = block_idx
        self.dropout = config.tts_adapter_dropout
        self.embed_dim = config.tts_adapter_hidden_size

        self.self_attn = TTSAdapterSelfAttention(
            config=config,
            layer_idx=layer_idx,
            block_idx=block_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        if block_idx == 0:
            self.encoder_attn = TTSAdapterCrossAttention(
                config=config,
                layer_idx=layer_idx,
                block_idx=block_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.encoder_attn",
            )
            self.encoder_attn_layer_norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        self.mlp = TTSAdapterMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.final_layer_norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)


class TTSAdapterLayer(nn.Module):
    def __init__(
        self,
        config: AvaterVoiceConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.mlp_block = nn.ModuleList([])
        self.layer_idx=int(prefix.split(".")[-1])
        for block_idx in range(config.block_size):
            self.mlp_block.append(TTSAdapterBlock(
                config,
                block_idx,
                self.layer_idx,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.mlp_block.{block_idx}"
            ))


# @support_torch_compile
class TTSAdapter(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "kv_proj": ["k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: AvaterVoiceConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.config.audio_vocab_size,
                self.config.tts_adapter_hidden_size,
                org_num_embeddings=self.config.audio_vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.tts_adapter_hidden_layers,
            lambda prefix: TTSAdapterLayer(config=config,
                                      cache_config=cache_config,
                                      quant_config=quant_config,
                                      prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.adapter_head = TTSAdapterHead(
                config,
                lora_config,
                quant_config,
                prefix=f"{prefix}.adapter_head"
            )

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.adapter_head.unpadded_vocab_size,
                self.adapter_head.vocab_size,
                logit_scale
            )
        else:
            self.adapter_head = PPMissingLayer()
        
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.tts_adapter_hidden_size
        )

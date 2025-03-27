"""Inference-only TTS Adapter model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import torch
from torch import nn

from vllm.attention import AttentionMetadata, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
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
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
)


from avatar_infer.models.voice.configuration_voice import AvatarVoiceConfig
from avatar_infer.models_vllm.layers import Attention


class TTSAdapterHead(nn.Module):
    def __init__(
        self,
        config: AvatarVoiceConfig,
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
                        8
                        # We need bigger padding if using lora for kernel
                        # compatibility
                        if not lora_config else
                        lora_config.lora_vocab_padding_size),
                    quant_config=quant_config,
                    prefix=f"{prefix}.head_block.{idx}",
                )
            )


class TTSAdapterMLP(nn.Module):
    def __init__(
        self,
        config: AvatarVoiceConfig,
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
        config: AvatarVoiceConfig,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = ""
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.config = config
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

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
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
        q_state, k_state, v_state = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_state, k_state = self.rotary_emb(positions, q_state, k_state)
        attn_output = self.attn(q_state, k_state, v_state, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class TTSAdapterCrossAttention(nn.Module):
    def __init__(
        self,
        config: AvatarVoiceConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()

        self.config = config
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
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)

        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
        )

        output, _ = self.o_proj(attn_output)

        return output


class TTSAdapterBlock(nn.Module):
    def __init__(
        self,
        config: AvatarVoiceConfig,
        block_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = ""
    ):
        super().__init__()
        prefix_list = prefix.split(".")
        self.attn_idx = int(prefix_list[-1])
        self.block_idx = block_idx
        self.dropout = config.tts_adapter_dropout
        self.embed_dim = config.tts_adapter_hidden_size

        self.self_attn = TTSAdapterSelfAttention(
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=".".join(prefix_list[:-1]+[str(self.attn_idx)]) + ".self_attn",
        )
        self.self_attn_layer_norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        if block_idx == 0:
            self.encoder_attn = TTSAdapterCrossAttention(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=".".join(prefix_list[:-1]+[str(self.attn_idx+1)]) + ".encoder_attn",
            )
            self.encoder_attn_layer_norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

        self.mlp = TTSAdapterMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        self.final_layer_norm = RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_caches[self.attn_idx],
            attn_metadata=attn_metadata
        )
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        if self.block_idx == 0:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states = self.encoder_attn(
                hidden_states=hidden_states,
                kv_cache=kv_caches[self.attn_idx+1],
                attn_metadata=attn_metadata,
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TTSAdapterLayer(nn.Module):
    def __init__(
        self,
        config: AvatarVoiceConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList([])
        prefix_list = prefix.split(".")
        layer_idx = int(prefix_list[-1])
        for block_idx in range(config.block_size):
            if block_idx == 0:
                attn_idx = layer_idx*2
            else:
                attn_idx = layer_idx + (block_idx * config.tts_adapter_hidden_layers) + config.tts_adapter_hidden_layers

            laryer_prefix = ".".join(prefix_list[:-1]+[str(attn_idx)])
            self.block.append(TTSAdapterBlock(
                config,
                block_idx=block_idx,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=laryer_prefix
            ))

    def forward(
        self,
        block_idx: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        hidden_states = self.block[block_idx](
            positions,
            hidden_states,
            encoder_hidden_states,
            kv_caches,
            attn_metadata
        )
        return hidden_states


@support_torch_compile
class TTSAdapter(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "kv_proj": ["k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: AvatarVoiceConfig = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id

        self.embed_tokens = VocabParallelEmbedding(
            self.config.audio_vocab_size,
            self.config.tts_adapter_hidden_size,
            org_num_embeddings=self.config.audio_vocab_size,
            quant_config=quant_config,
        )

        self.layers = nn.ModuleList([
            TTSAdapterLayer(
                config,
                cache_config,
                quant_config,
                prefix=f"{prefix}.layers.{layer_idx}"
            ) for layer_idx in range(self.config.tts_adapter_hidden_layers)
        ])

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

        self.norm = nn.ModuleList([])
        for _ in range(config.block_size):
            self.norm.append(RMSNorm(config.tts_adapter_hidden_size, eps=config.rms_norm_eps))

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.tts_adapter_hidden_size
        )

    def forward_block(
        self,
        block_idx: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ):
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                block_idx,
                positions,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata
            )

        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        pos_bias = (
            torch.arange(0, self.config.code_layers, device=input_ids.device)
            *
            (self.config.code_size+self.config.audio_special_tokens)
        ).view(1, self.config.code_layers)
        inputs_embeds = self.embed_tokens(input_ids+pos_bias)
        inputs_embeds = inputs_embeds.sum(dim=1)

        logits_hidden_states = []
        hidden_states = inputs_embeds
        for block_idx in range(self.config.block_size):
            residual = hidden_states

            hidden_states = self.forward_block(
                block_idx=block_idx,
                positions=positions,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata
            )

            # hidden_states for Logits
            logits_hidden_states.append(hidden_states)

            # residual
            hidden_states = residual.to(hidden_states) + hidden_states
        
        logits_hidden_states = torch.cat(logits_hidden_states, dim=-1)
        return logits_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
            (".encoder_attn.kv_proj", ".encoder_attn.k_proj", "k"),
            (".encoder_attn.kv_proj", ".encoder_attn.v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

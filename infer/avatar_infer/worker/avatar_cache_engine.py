"""AvatarCacheEngine class for managing the KV cache."""
from typing import List, Dict, Any

import numpy as np
import torch

from vllm import envs
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    align_to_256bytes,
    get_dtype_size,
    is_pin_memory_available
)

logger = init_logger(__name__)


def bind_kv_cache(
    ctx: Dict[str, Any],
    kv_cache: List[List[torch.Tensor]],  # [virtual_engine][layer_index]
) -> None:
    """Bind KV cache tensors to attention modules.
    
    This function maps KV cache tensors to corresponding attention modules:
    ctx[layer_name].kv_cache[ve] = kv_cache[ve][extract_layer_index(layer_name)]
    
    Special cases handled:
    1. Models with non-attention layers (e.g. Jamba)
    2. Pipeline parallelism with subset of layers per rank
    3. Encoder attention without KV cache
    4. Encoder-decoder models where encoder-decoder attention and decoder-only
       attention of same layer share KV cache tensor
    """
    from vllm.attention import AttentionType
    from vllm.model_executor.models.utils import extract_layer_index

    used_kv_cache_idx = 0
    for model_type in ["llm", "tts_adapter"]:
        # Get layers that need KV cache (decoder and encoder-decoder attention)
        layer_need_kv_cache = [
            layer_name for layer_name in ctx
            if ctx[layer_name].attn_type in (AttentionType.DECODER, AttentionType.ENCODER_DECODER) 
            and layer_name.startswith(model_type)
        ]

        # Get sorted unique layer indices
        layer_index_sorted = sorted(
            set(
                extract_layer_index(layer_name)
                for layer_name in layer_need_kv_cache
            )
        )

        # Bind KV cache tensors to attention modules
        for layer_name in layer_need_kv_cache:
            kv_cache_idx = layer_index_sorted.index(extract_layer_index(layer_name))
            forward_ctx = ctx[layer_name]
            assert len(forward_ctx.kv_cache) == len(kv_cache)

            for ve, ve_kv_cache in enumerate(kv_cache):
                forward_ctx.kv_cache[ve] = ve_kv_cache[used_kv_cache_idx:][kv_cache_idx]

        used_kv_cache_idx += len(layer_index_sorted)


def get_avatar_param(model_config: ModelConfig):
    llm_head_size = model_config.hf_config.hidden_size // model_config.hf_config.num_attention_heads
    llm_num_attention_layers = model_config.hf_config.num_hidden_layers
    llm_num_kv_heads = model_config.hf_config.num_key_value_heads

    tts_adapter_head_size = model_config.hf_config.tts_adapter_hidden_size // model_config.hf_config.tts_adapter_attention_heads
    tts_adapter_num_attention_layers = (
        model_config.hf_config.tts_adapter_hidden_layers
        *
        model_config.hf_config.block_size
    ) + model_config.hf_config.tts_adapter_hidden_layers
    tts_adapter_num_kv_heads = model_config.hf_config.tts_adapter_attention_heads//4
    
    return (
        llm_head_size, llm_num_attention_layers, llm_num_kv_heads,
        tts_adapter_head_size, tts_adapter_num_attention_layers, tts_adapter_num_kv_heads
    )


class AvatarCacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        (
            self.llm_head_size, self.llm_num_attention_layers, self.llm_num_kv_heads,
            self.tts_adapter_head_size, self.tts_adapter_num_attention_layers, self.tts_adapter_num_kv_heads
        ) = get_avatar_param(model_config)

        self.align_cache = self._align_cache(model_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.llm_attn_backend = get_attn_backend(
            self.llm_head_size,
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
            model_config.is_attention_free,
            use_mla=model_config.use_mla
        )
        self.tts_adapter_attn_backend = get_attn_backend(
            self.tts_adapter_head_size,
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
            model_config.is_attention_free,
            use_mla=model_config.use_mla
        )

        # Calculate memory requirements for LLM and TTS adapter
        llm_block_size = self._calculate_model_block_size(
            self.llm_head_size,
            self.llm_num_kv_heads,
            self.llm_num_attention_layers,
            model_config
        )

        tts_block_size = self._calculate_model_block_size(
            self.tts_adapter_head_size,
            self.tts_adapter_num_kv_heads,
            self.tts_adapter_num_attention_layers,
            model_config
        )

        # Calculate the ratio based on memory requirements
        total_size = llm_block_size + tts_block_size
        self.llm_ratio = llm_block_size / total_size if total_size > 0 else 0.5

        logger.info(
            f"Dynamically allocated memory ratio: LLM {self.llm_ratio:.2f}, "
            f"TTS Adapter {1 - self.llm_ratio:.2f}"
        )

        # Calculate blocks for each model
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
            self.llm_num_gpu_blocks = max(1, int(self.num_gpu_blocks * self.llm_ratio))
            self.tts_num_gpu_blocks = max(1, self.num_gpu_blocks - self.llm_num_gpu_blocks)

        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size
            self.llm_num_cpu_blocks = max(1, int(self.num_cpu_blocks * self.llm_ratio))
            self.tts_num_cpu_blocks = max(1, self.num_cpu_blocks - self.llm_num_cpu_blocks)

        # Initialize the separate caches
        self.llm_gpu_cache = self._allocate_llm_kv_cache(self.llm_num_gpu_blocks, self.device_config.device_type)
        self.llm_cpu_cache = self._allocate_llm_kv_cache(self.llm_num_cpu_blocks, "cpu")

        self.tts_gpu_cache = self._allocate_tts_kv_cache(self.tts_num_gpu_blocks, self.device_config.device_type)
        self.tts_cpu_cache = self._allocate_tts_kv_cache(self.tts_num_cpu_blocks, "cpu")

        # Combine them for the main interface
        self.gpu_cache = self.llm_gpu_cache + self.tts_gpu_cache
        self.cpu_cache = self.llm_cpu_cache + self.tts_cpu_cache

    def _calculate_model_block_size(self, head_size, num_kv_heads, num_layers, model_config):
        """Calculate memory requirements for a model component."""
        key_cache_entry = num_kv_heads * head_size
        if self._align_cache(model_config):
            key_cache_entry = align_to_256bytes(key_cache_entry, model_config.dtype)

        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_layers * self.block_size * (key_cache_entry + value_cache_entry)

        if self.cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[self.cache_config.cache_dtype]

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    def _allocate_llm_kv_cache(self, num_blocks: int, device: str) -> List[torch.Tensor]:
        """Allocates KV cache for LLM on the specified device."""
        if num_blocks == 0:
            return []

        llm_kv_cache_shape = self.llm_attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.llm_num_kv_heads, self.llm_head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        if self.align_cache:
            llm_entry_size = np.prod(llm_kv_cache_shape[2:])
            llm_alloc_entry_size = align_to_256bytes(llm_entry_size, self.dtype)
            llm_alloc_shape = (*llm_kv_cache_shape[:2], llm_alloc_entry_size)
        else:
            llm_alloc_shape = llm_kv_cache_shape

        for _ in range(self.llm_num_attention_layers):
            layer_kv_cache = torch.zeros(
                llm_alloc_shape,
                dtype=self.dtype,
                pin_memory=pin_memory,
                device=device
            )
            
            if self.align_cache:
                layer_kv_cache = layer_kv_cache[..., :llm_entry_size]
                
            kv_cache.append(layer_kv_cache.view(llm_kv_cache_shape))
            
        return kv_cache

    def _allocate_tts_kv_cache(self, num_blocks: int, device: str) -> List[torch.Tensor]:
        """Allocates KV cache for TTS adapter on the specified device."""
        if num_blocks == 0:
            return []

        tts_kv_cache_shape = self.tts_adapter_attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.tts_adapter_num_kv_heads, self.tts_adapter_head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        if self.align_cache:
            tts_entry_size = np.prod(tts_kv_cache_shape[2:])
            tts_alloc_entry_size = align_to_256bytes(tts_entry_size, self.dtype)
            tts_alloc_shape = (*tts_kv_cache_shape[:2], tts_alloc_entry_size)
        else:
            tts_alloc_shape = tts_kv_cache_shape

        for _ in range(self.tts_adapter_num_attention_layers):
            layer_kv_cache = torch.zeros(
                tts_alloc_shape,
                dtype=self.dtype,
                pin_memory=pin_memory,
                device=device
            )

            if self.align_cache:
                layer_kv_cache = layer_kv_cache[..., :tts_entry_size]

            kv_cache.append(layer_kv_cache.view(tts_kv_cache_shape))

        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        """Swap blocks from CPU to GPU."""
        # Extract the block indices relevant to each model
        # You'll need to adapt this based on how you track which blocks belong to which model
        llm_indices = src_to_dst[src_to_dst[:, 0] < self.llm_num_gpu_blocks]
        tts_indices = src_to_dst[src_to_dst[:, 0] >= self.llm_num_gpu_blocks]

        # Adjust TTS indices to account for offset
        if tts_indices.shape[0] > 0:
            tts_indices[:, 0] -= self.llm_num_gpu_blocks
            tts_indices[:, 1] -= self.llm_num_cpu_blocks

        # Perform swaps for each model
        for i in range(self.llm_num_attention_layers):
            if llm_indices.shape[0] > 0:
                self.llm_attn_backend.swap_blocks(
                    self.llm_cpu_cache[i], 
                    self.llm_gpu_cache[i],
                    llm_indices
                )

        for i in range(self.tts_adapter_num_attention_layers):
            if tts_indices.shape[0] > 0:
                self.tts_adapter_attn_backend.swap_blocks(
                    self.tts_cpu_cache[i], 
                    self.tts_gpu_cache[i],
                    tts_indices
                )

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        """Swap blocks from GPU to CPU."""
        # Similar logic to swap_in but in reverse direction
        llm_indices = src_to_dst[src_to_dst[:, 0] < self.llm_num_gpu_blocks]
        tts_indices = src_to_dst[src_to_dst[:, 0] >= self.llm_num_gpu_blocks]

        if tts_indices.shape[0] > 0:
            tts_indices[:, 0] -= self.llm_num_gpu_blocks
            tts_indices[:, 1] -= self.llm_num_cpu_blocks

        for i in range(self.llm_num_attention_layers):
            if llm_indices.shape[0] > 0:
                self.llm_attn_backend.swap_blocks(
                    self.llm_gpu_cache[i], 
                    self.llm_cpu_cache[i],
                    llm_indices
                )

        for i in range(self.tts_adapter_num_attention_layers):
            if tts_indices.shape[0] > 0:
                self.tts_adapter_attn_backend.swap_blocks(
                    self.tts_gpu_cache[i], 
                    self.tts_cpu_cache[i],
                    tts_indices
                )

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        """Copy blocks within GPU cache."""
        # Split copy operations by model
        llm_indices = src_to_dsts[src_to_dsts[:, 0] < self.llm_num_gpu_blocks]
        tts_indices = src_to_dsts[src_to_dsts[:, 0] >= self.llm_num_gpu_blocks]

        if tts_indices.shape[0] > 0:
            tts_indices[:, 0] -= self.llm_num_gpu_blocks
            tts_indices[:, 1] -= self.llm_num_gpu_blocks

        if llm_indices.shape[0] > 0:
            self.llm_attn_backend.copy_blocks(self.llm_gpu_cache, llm_indices)

        if tts_indices.shape[0] > 0:
            self.tts_adapter_attn_backend.copy_blocks(self.tts_gpu_cache, tts_indices)

    @staticmethod
    def _align_cache(model_config: ModelConfig):
        # Currently align_cache only applies to MLA models since the other
        # cache kernels haven't been updated yet to support non-continguous
        # tensors
        return model_config.use_mla and current_platform.is_cuda() \
            and envs.VLLM_CUDA_MEM_ALIGN_KV_CACHE

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        """Calculate the combined cache block size for both LLM and TTS adapter."""
        # Need to account for both model components
        (
            llm_head_size, llm_num_layers, llm_num_kv_heads,
            tts_head_size, tts_num_layers, tts_num_kv_heads
        ) = get_avatar_param(model_config)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Calculate LLM cache size
        llm_key_cache_entry = llm_num_kv_heads * llm_head_size
        if AvatarCacheEngine._align_cache(model_config):
            llm_key_cache_entry = align_to_256bytes(llm_key_cache_entry, model_config.dtype)

        llm_value_cache_entry = llm_key_cache_entry if not model_config.use_mla else 0
        llm_total = llm_num_layers * cache_config.block_size * (llm_key_cache_entry + llm_value_cache_entry)

        # Calculate TTS adapter cache size
        tts_key_cache_entry = tts_num_kv_heads * tts_head_size
        if AvatarCacheEngine._align_cache(model_config):
            tts_key_cache_entry = align_to_256bytes(tts_key_cache_entry, model_config.dtype)

        tts_value_cache_entry = tts_key_cache_entry if not model_config.use_mla else 0
        tts_total = tts_num_layers * cache_config.block_size * (tts_key_cache_entry + tts_value_cache_entry)

        # Combined size
        dtype_size = get_dtype_size(dtype)
        return dtype_size * (llm_total + tts_total)

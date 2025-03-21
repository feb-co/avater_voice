from typing import Dict, List, Optional, Set, Tuple, Type, Union

from vllm.config import VllmConfig
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.worker import Worker


from .avatar_model_runner import AvatarModelRunner
from .avatar_cache_engine import AvatarCacheEngine, bind_kv_cache


# Save original init method
original_worker_init = Worker.__init__
original_init_cache_engine = Worker._init_cache_engine


def custom_worker_init(
    self,
    vllm_config: VllmConfig,
    local_rank: int,
    rank: int,
    distributed_init_method: str,
    is_driver_worker: bool = False,
    model_runner_cls: Optional[Type[GPUModelRunnerBase]] = None,
) -> None:
    """Custom Worker initialization method
    
    Args:
        vllm_config: vLLM configuration
        local_rank: Local rank
        rank: Global rank
        distributed_init_method: Distributed initialization method
        is_driver_worker: Whether this is a driver worker
        model_runner_cls: Model runner class
    """
    original_worker_init(
        self,
        vllm_config,
        local_rank,
        rank,
        distributed_init_method,
        is_driver_worker,
        model_runner_cls=AvatarModelRunner
    )


def custom_init_cache_engine(self):
    """Custom implementation of _init_cache_engine using AvatarCacheEngine"""
    assert self.cache_config.num_gpu_blocks is not None

    # Initialize cache engines for each pipeline parallel rank
    self.cache_engine = [
        AvatarCacheEngine(
            self.cache_config,
            self.model_config,
            self.parallel_config,
            self.device_config
        ) for _ in range(self.parallel_config.pipeline_parallel_size)
    ]

    # Get GPU caches from each cache engine
    self.gpu_cache = [
        self.cache_engine[ve].gpu_cache
        for ve in range(self.parallel_config.pipeline_parallel_size)
    ]

    # Bind KV cache to static forward context
    bind_kv_cache(self.compilation_config.static_forward_context, self.gpu_cache)


def apply_patch() -> None:
    """Apply patch to replace Worker's init method"""
    Worker.__init__ = custom_worker_init
    Worker._init_cache_engine = custom_init_cache_engine

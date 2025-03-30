from typing import Callable, List, Optional, Type

from vllm.config import VllmConfig, SchedulerConfig
from vllm.entrypoints.llm import LLM
from vllm.worker.model_runner import GPUModelRunnerBase
from vllm.worker.worker import Worker
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.core.scheduler import Scheduler
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.sequence import Sequence
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import Counter

from ..worker.avatar_llm_engine import AvatarLLMEngine
from ..worker.avatar_model_runner import AvatarModelRunner
from ..worker.avatar_cache_engine import AvatarCacheEngine, bind_kv_cache


# Save original init method
original_worker_init = Worker.__init__
original_init_cache_engine = Worker._init_cache_engine
original_output_processor = SequenceGroupOutputProcessor.create_output_processor


def custom_engine_class(self):
    return AvatarLLMEngine


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


def custom_get_cache_block_size_bytes(self) -> int:
    """
    Get the size of the KV cache block size in bytes.
    """
    return AvatarCacheEngine.get_cache_block_size(
        self.cache_config,
        self.model_config,
        self.parallel_config
    )


def custom_create_output_processor(
    scheduler_config: SchedulerConfig,
    detokenizer: Detokenizer,
    scheduler: List[Scheduler],
    seq_counter: Counter,
    get_tokenizer_for_seq: Callable[[Sequence], AnyTokenizer],
    stop_checker: "StopChecker",
):
    """Create an output processor.
    This returns a single-step output processor if num_lookahead_slots is
    zero, else returns a multi-step output processor.
    """
    if scheduler_config.num_lookahead_slots == 0:
        # Importing here to avoid cycle.
        from ..processor.avatar_step import AvatarStepOutputProcessor
        return AvatarStepOutputProcessor(
            scheduler_config,
            detokenizer,
            scheduler,
            seq_counter,
            stop_checker
        )
    else:
        # Importing here to avoid cycle.
        from vllm.engine.output_processor.multi_step import (
            MultiStepOutputProcessor)
        return MultiStepOutputProcessor(
            detokenizer,
            scheduler,
            seq_counter,
            get_tokenizer_for_seq,
            stop_checker,
        )



def apply_patch() -> None:
    # Engine patch
    LLM.get_engine_class = custom_engine_class

    # Worker patch
    Worker.__init__ = custom_worker_init
    Worker._init_cache_engine = custom_init_cache_engine
    Worker.get_cache_block_size_bytes = custom_get_cache_block_size_bytes

    # Processor
    SequenceGroupOutputProcessor.create_output_processor = custom_create_output_processor

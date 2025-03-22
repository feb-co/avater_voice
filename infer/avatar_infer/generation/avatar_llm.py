import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, Union

from vllm import LLM, SamplingParams
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner

from transformers.cache_utils import DynamicCache

from avatar_infer.worker.patch import apply_patch
from avatar_infer.cache_utils import AvatarCache, AvatarTokenCache
from .text_generator import LLMGenerator 
from .voice_generator import VoiceGenerator


apply_patch()


class AvatarLLM(LLM):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        **kwargs
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            **kwargs
        )

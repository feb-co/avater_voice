import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

from vllm import LLM, SamplingParams

from transformers.cache_utils import DynamicCache

from avater_infer.cache_utils import AvaterCache, AvaterTokenCache
from avater_infer.modeling_utils import load_model_tokenizer
from .text_generator import LLMGenerator
from .voice_generator import VoiceGenerator


class AvaterForGeneration:
    def __init__(
        self,
        args
    ) -> None:
        # Initialize VLLM model - this will be used for both LLM and TTS
        self.vllm_model = LLM(
            model=args.model, 
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
            dtype=getattr(args, 'dtype', 'float16'),
            max_model_len=getattr(args, 'max_model_len', 8192),
        )
        import pdb; pdb.set_trace()

        # Get the tokenizer from VLLM model
        self.tokenizer = self.vllm_model.get_tokenizer()

        # VLLM handles its own device placement
        # Set multi_gpu flag based on available GPU count
        self.multi_gpu = torch.cuda.device_count() > 1

        # Initialize shared cache
        self.cache = AvaterCache(
            llm_attention_cache=DynamicCache(),
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
            avater_token_cache=AvaterTokenCache()
        )

        # Create VLLM sampling parameters
        self.sampling_params = SamplingParams(
            temperature=getattr(args, 'temperature', 0.4),
            top_p=getattr(args, 'top_p', 0.01),
            max_tokens=getattr(args, 'max_tokens', 512),
        )

        # Create generators - both using VLLM
        self.llm_generator = LLMGenerator(
            model=None,  # We're fully using VLLM, no need for original model
            tokenizer=self.tokenizer, 
            cache=self.cache,
            vllm_model=self.vllm_model,
            sampling_params=self.sampling_params
        )
        
        # Create TTS generator with VLLM
        tts_sampling_params = SamplingParams(
            temperature=getattr(args, 'tts_temperature', 0.1),
            top_p=getattr(args, 'tts_top_p', 0.01),
            max_tokens=getattr(args, 'tts_max_tokens', 512),
        )
        
        self.voice_generator = VoiceGenerator(
            model=None,  # No original model
            tokenizer=self.tokenizer, 
            cache=self.cache,
            vllm_model=self.vllm_model,
            sampling_params=tts_sampling_params
        )

        # Generation options
        self.progressive_generation = True
        self.progressive_delay = 0.5

    async def chat(
        self,
        conversation,
    ):
        if self.progressive_generation:
            return await self.progressive_chat(conversation)
        else:
            return await self.parallel_chat(conversation)

    async def progressive_chat(self, conversation):
        """Incremental generation mode that starts TTS after a short delay"""
        executor = ThreadPoolExecutor(max_workers=2)
        loop = asyncio.get_running_loop()
        
        # Start LLM generation
        llm_future = loop.run_in_executor(executor, self.llm_generator.run, conversation)
        
        # Wait briefly to let LLM generate initial tokens
        await asyncio.sleep(self.progressive_delay)
        
        # Start voice generation
        voice_future = loop.run_in_executor(executor, self.voice_generator.run, conversation)
        
        # Wait for both tasks to complete
        llm_outputs, voice_outputs = await asyncio.gather(llm_future, voice_future)
        
        # Ensure all CUDA operations have completed
        torch.cuda.synchronize()
        
        # Performance logging
        print("LLM generator time:", self.llm_generator.test_time)
        print("Voice generator time:", self.voice_generator.test_time)
        
        return llm_outputs, voice_outputs

    async def parallel_chat(self, conversation):
        """Original parallel generation mode"""
        executor = ThreadPoolExecutor(max_workers=2)
        loop = asyncio.get_running_loop()
        
        # Schedule both tasks to run concurrently using the thread pool
        llm_future = loop.run_in_executor(executor, self.llm_generator.run, conversation)
        voice_future = loop.run_in_executor(executor, self.voice_generator.run, conversation)

        # Wait for both tasks to complete
        llm_outputs, voice_outputs = await asyncio.gather(llm_future, voice_future)

        # Ensure all CUDA operations have completed
        torch.cuda.synchronize()
        
        # Performance logging
        print("LLM generator time:", self.llm_generator.test_time)
        print("Voice generator time:", self.voice_generator.test_time)
        
        return llm_outputs, voice_outputs

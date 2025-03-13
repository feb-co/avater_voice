import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

from transformers.cache_utils import DynamicCache

from avater_infer.cache_utils import AvaterCache, AvaterTokenCache
from avater_infer.modeling_utils import load_model_tokenizer
from .text_generator import LLMGenerator
from .voice_generator import VoiceGenerator


class AvaterForGeneration:
    def __init__(
        self,
        model: str,
    ) -> None:
        # Initialize model and tokenizer
        self.tokenizer, self.model = load_model_tokenizer(model)

        # Try to place models on separate devices if available
        if torch.cuda.device_count() > 1:
            try:
                self.model.llm.to('cuda:0')
                self.model.tts_adapter.to('cuda:1')
                self.multi_gpu = True
            except:
                self.multi_gpu = False
        else:
            self.multi_gpu = False

        # Initialize shared cache
        self.cache = AvaterCache(
            llm_attention_cache=DynamicCache(),
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
            avater_token_cache=AvaterTokenCache()
        )

        # Create generators
        self.llm_generator = LLMGenerator(self.model.llm, self.tokenizer, self.cache)
        self.voice_generator = VoiceGenerator(self.model.tts_adapter, self.tokenizer, self.cache)

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

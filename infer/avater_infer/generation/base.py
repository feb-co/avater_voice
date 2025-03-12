# AvaterForGeneration class
import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor

from transformers.cache_utils import DynamicCache
from transformers.generation import GenerationMixin

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

        # Initialize shared cache
        self.cache_values = AvaterCache(
            llm_attention_cache=DynamicCache(),
            self_attention_cache=DynamicCache(),
            cross_attention_cache=DynamicCache(),
            avater_token_cache=AvaterTokenCache()
        )

        # Pre-initialize generators to avoid creating them for each chat
        self.llm_generator = LLMGenerator(self.model.llm, self.tokenizer, self.cache_values)
        self.voice_generator = VoiceGenerator(self.model.tts_adapter, self.tokenizer, self.cache_values)

        # Create dedicated CUDA streams for parallel execution
        self.llm_stream = torch.cuda.Stream()
        self.voice_stream = torch.cuda.Stream()

        # Create thread pool executor with fixed size
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def chat(
        self,
        conversation,
    ):
        # Define LLM generation task that runs in its own CUDA stream
        def run_llm_task():
            with torch.cuda.stream(self.llm_stream):
                # All operations in this context will be queued in this stream
                outputs = GenerationMixin.generate(
                    self.llm_generator, 
                    **self.llm_generator._prepare_inputs(conversation),
                    generation_config=self.llm_generator.generation_config
                )

                # Update token cache after generation
                self.cache_values.avater_token_cache.update(
                    layer_idx=0,
                    token_ids=self.llm_generator.generation_config.eos_token_id
                )

                return outputs

        # Define voice generation task that runs in its own CUDA stream
        def run_voice_task():
            with torch.cuda.stream(self.voice_stream):
                # All operations in this context will be queued in this stream
                return GenerationMixin.generate(
                    self.voice_generator, 
                    **self.voice_generator._prepare_inputs(conversation),
                    generation_config=self.voice_generator.generation_config
                )

        # Schedule both tasks to run concurrently using the thread pool
        loop = asyncio.get_running_loop()
        llm_future = loop.run_in_executor(self.executor, run_llm_task)
        voice_future = loop.run_in_executor(self.executor, run_voice_task)

        # Wait for both tasks to complete
        llm_outputs, voice_outputs = await asyncio.gather(llm_future, voice_future)

        # Ensure all CUDA operations have completed
        torch.cuda.synchronize()
        return llm_outputs, voice_outputs

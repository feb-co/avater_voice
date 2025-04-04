import time
import inspect
import dataclasses
import weakref
import msgspec
import copy
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Type, cast, Set

import torch
import torch.distributed

from vllm.sequence import SequenceData
from vllm.attention import get_attn_backend
from vllm.attention.backends.abstract import (AttentionBackend, AttentionMetadata)
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank, graph_capture)
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalKwargs
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceGroupMetadata)
from vllm.worker.model_runner import (
    GPUModelRunnerBase,
    ModelInputForGPUBuilder,
    ModelInputForGPUWithSamplingMetadata,
    CUDAGraphRunner
)
from vllm.worker.model_runner_base import (_add_attn_metadata_broadcastable_dict, _add_sampling_metadata_broadcastable_dict)
from vllm.utils import GiB_bytes

from .avatar_cache_engine import get_avatar_param
from .forward_avatar_context import set_tts_forward_context
from avatar_infer.models.voice.configuration_voice import AvatarVoiceConfig
from avatar_infer.dataclass.sequence import (
    TTSSequenceData,
    AvatarSequenceGroupMetadata,
)
from avatar_infer.utils import token_to_tts_codes


logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class AvatarModelInput(ModelInputForGPUWithSamplingMetadata):
    """Used by the AvatarModelInput."""
    encoder_input_tokens: Optional[torch.Tensor] = None
    encoder_input_positions: Optional[torch.Tensor] = None
    encoder_attn_metadata: Optional["AttentionMetadata"] = None
    encoder_sampling_metadata: Optional["SamplingMetadata"] = None    

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "encoder_input_tokens": self.encoder_input_tokens,
            "encoder_input_positions": self.encoder_input_positions,
            "virtual_engine": self.virtual_engine,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
            "multi_modal_kwargs": self.multi_modal_kwargs,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.encoder_attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict, self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "AvatarModelInput":
        return cast(AvatarModelInput, super().from_broadcasted_tensor_dict(tensor_dict, attn_backend))


class AvatarModelInputForGPUBuilder(ModelInputForGPUBuilder):
    """Build AvatarModelInputForGPU from SequenceGroupMetadata."""
    def __init__(
        self,
        runner: "GPUModelRunnerBase",
        finished_requests_ids: Optional[List[str]] = None
    ):
        super().__init__(runner)
        # Compute functions for each sequence in a sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_compute_fns = [
            self._compute_lens,
            self._compute_for_prefix_cache_hit,
            self._compute_for_sliding_window,
            self._compute_lora_input,
        ]
        # Compute functions for each sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_group_compute_fns = [
            self._compute_prompt_adapter_input,
            self._compute_multi_modal_input,
        ]

        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.scheduler_config = self.runner.scheduler_config
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.enable_lora = self.runner.lora_config is not None
        self.enable_prompt_adapter = (self.runner.prompt_adapter_config
                                      is not None)
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper

        # Attention metadata inputs.
        if self.attn_backend is not None:
            # spec decode (e.g. Medusa) does not have atten backend
            self.attn_metadata_builder = self.attn_backend.get_builder_cls()(weakref.proxy(self))

        # Engine/Model configurations.
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None
            and self.scheduler_config.chunked_prefill_enabled)
        if self.sliding_window is not None:
            self.sliding_window_blocks = (self.sliding_window + self.block_size - 1) // self.block_size
            self.block_aligned_sliding_window = self.sliding_window_blocks * self.block_size
    
    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        inter_data = self.init_cached_inter_data(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums,
            reinit=True,
            reinit_use_defaults=True,
            encoder_seq_len=0)

        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            for per_seq_fn in self.per_seq_compute_fns:
                per_seq_fn(inter_data, seq_idx, seq_group_metadata)
        for per_seq_group_fn in self.per_seq_group_compute_fns:
            per_seq_group_fn(inter_data, seq_group_metadata)


class AvatarModelRunner(GPUModelRunnerBase[AvatarModelInput]):
    _builder_cls: Type[AvatarModelInputForGPUBuilder] = AvatarModelInputForGPUBuilder

    def __init__(self, original_model_runner):
        # Store the original model runner
        self.original_model_runner = original_model_runner

        # Forward all attributes from the original model runner
        for attr_name in dir(original_model_runner):
            if not attr_name.startswith('__') and not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(original_model_runner, attr_name))

        # add config for override
        self.avatar_config: AvatarVoiceConfig = original_model_runner.vllm_config.model_config.hf_config
        self._model_input_cls: Type[AvatarModelInput] = AvatarModelInput
        self.builder = self._builder_cls(weakref.proxy(self))
        self.tts_head_size = self.model_config.hf_config.tts_adapter_hidden_size//self.model_config.hf_config.tts_adapter_attention_heads
        self.tts_attn_backend = get_attn_backend(
            self.tts_head_size,
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        if self.tts_attn_backend:
            self.tts_attn_state = self.tts_attn_backend.get_state_cls()(weakref.proxy(self))

    @torch.inference_mode()
    def capture_model(self, kv_caches: List[List[torch.Tensor]]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        # Check model configuration
        assert not self.model_config.enforce_eager
        logger.info("Capturing cudagraphs for decoding. This may lead to "
                   "unexpected consequences if the model is not static. To "
                   "run the model in eager mode, set 'enforce_eager=True' or "
                   "use '--enforce-eager' in the CLI. "
                   "If out-of-memory error occurs during cudagraph capture,"
                   " consider decreasing `gpu_memory_utilization` or "
                   "switching to eager mode. You can also reduce the "
                   "`max_num_seqs` as needed to decrease memory usage.")

        # Record start time and GPU memory
        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Prepare input tensors
        max_batch_size = self.max_batchsize_to_capture
        input_tokens = torch.zeros([max_batch_size, self.avatar_config.code_layers], dtype=torch.long, device=self.device)
        input_positions = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)
        encoder_input_ids = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)
        encoder_positions = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)

        if self.model_config.uses_mrope:
            input_positions = torch.tile(input_positions, (3, 1)).cuda(device=self.device)
            encoder_positions = torch.tile(encoder_positions, (3, 1)).cuda(device=self.device)

        # Prepare hidden states
        previous_hidden_states = None
        if "previous_hidden_states" in inspect.signature(self.model.forward).parameters:
            previous_hidden_states = torch.empty(
                [max_batch_size, self.model_config.get_hidden_size()],
                dtype=self.model_config.dtype,
                device=self.device)

        # Prepare intermediate inputs
        intermediate_inputs = None
        if not get_pp_group().is_first_rank:
            intermediate_inputs = self.model.make_empty_intermediate_tensors(
                batch_size=max_batch_size,
                dtype=self.model_config.dtype,
                device=self.device)

        # Execute graph capture
        with self.attn_state.graph_capture(max_batch_size), self.tts_attn_state.graph_capture(max_batch_size), graph_capture(self.device) as graph_capture_context:
            for virtual_engine in range(self.parallel_config.pipeline_parallel_size):
                # Only print progress bar on rank 0
                cudagraph_capture_sizes = (
                    tqdm(
                        self.vllm_config.compilation_config.cudagraph_capture_sizes,
                        desc="Capturing CUDA graph shapes"
                    ) if get_tensor_model_parallel_rank() == 0 else
                    self.vllm_config.compilation_config.cudagraph_capture_sizes
                )

                for batch_size in cudagraph_capture_sizes:
                    # Prepare attention metadata
                    encoder_attn_metadata = self.attn_state.graph_capture_get_metadata_for_batch(
                        batch_size,
                        is_encoder_decoder_model=False
                    )
                    encoder_attn_metadata.enable_kv_scales_calculation = False
                    attn_metadata = self.tts_attn_state.graph_capture_get_metadata_for_batch(
                        batch_size,
                        is_encoder_decoder_model=False
                    )
                    attn_metadata.enable_kv_scales_calculation = False

                    # Setup LoRA
                    if self.lora_config:
                        lora_mapping = LoRAMapping(
                            **dict(index_mapping=[0] * batch_size,
                                  prompt_mapping=[0] * batch_size,
                                  is_prefill=False))
                        self.set_active_loras(set(), lora_mapping)

                    # Setup prompt adapter
                    if self.prompt_adapter_config:
                        prompt_adapter_mapping = PromptAdapterMapping(
                            [-1] * batch_size,
                            [-1] * batch_size)
                        self.set_active_prompt_adapters(set(), prompt_adapter_mapping)

                    # Create graph runner
                    graph_runner = CUDAGraphRunner(
                        self.model,
                        self.attn_backend.get_name(),
                        self.attn_state.graph_clone(batch_size),
                        self.model_config.is_encoder_decoder)

                    # Prepare capture inputs
                    capture_inputs = {
                        "input_ids": input_tokens[:batch_size],
                        "positions": input_positions[..., :batch_size],
                        "encoder_input_ids": encoder_input_ids[:batch_size],
                        "encoder_positions": encoder_positions[..., :batch_size],
                        "intermediate_inputs": intermediate_inputs[:batch_size] if intermediate_inputs is not None else None,
                        "kv_caches": kv_caches[virtual_engine],
                        "attn_metadata": attn_metadata,
                        "encoder_attn_metadata": encoder_attn_metadata,
                        "memory_pool": self.graph_memory_pool,
                        "stream": graph_capture_context.stream
                    }

                    if previous_hidden_states is not None:
                        capture_inputs["previous_hidden_states"] = previous_hidden_states[:batch_size]

                    if self.has_inner_state:
                        capture_inputs.update({
                            "seqlen_agnostic_capture_inputs":
                            self.model.get_seqlen_agnostic_capture_inputs(batch_size)
                        })

                    # Execute capture
                    with set_forward_context(encoder_attn_metadata, self.vllm_config, virtual_engine), set_tts_forward_context(attn_metadata, self.vllm_config, virtual_engine):
                        graph_runner.capture(**capture_inputs)

                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][batch_size] = graph_runner

        # Record end time and memory usage
        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory

        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB", elapsed_time, cuda_graph_size / GiB_bytes)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: AvatarModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[PoolerOutput]]:
        if num_steps > 1:
            raise ValueError("num_steps > 1 is not supported in AvatarModelInput")

        # if (model_input.attn_metadata is not None and 
        #     model_input.attn_metadata.prefill_metadata is None and
        #     model_input.attn_metadata.decode_metadata.use_cuda_graph):
        #     assert model_input.input_tokens is not None
        #     graph_batch_size = model_input.input_tokens.shape[0]
        #     model_executable = self.graph_runners[model_input.virtual_engine][graph_batch_size]
        # else:
        model_executable = self.model

        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        with set_forward_context(model_input.encoder_attn_metadata, self.vllm_config, model_input.virtual_engine), set_tts_forward_context(model_input.attn_metadata, self.vllm_config, model_input.virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                encoder_input_ids=model_input.encoder_input_tokens,
                encoder_positions=model_input.encoder_input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                encoder_attn_metadata=model_input.encoder_attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(multi_modal_kwargs, device=self.device),
                **seqlen_agnostic_kwargs
            )

        logits = self.model.compute_logits(
            hidden_or_intermediate_states,
            model_input.sampling_metadata,
            model_input.encoder_sampling_metadata
        )

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
            encoder_sampling_metadata=model_input.encoder_sampling_metadata
        )

        return [output]

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []

        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(self.model_config)
        if max_mm_tokens > 0:
            logger.info("Starting profile run for multi-modal models.")

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs + (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            llm_dummy_data = self.input_registry.dummy_data_for_profiling(
                self.model_config,
                seq_len,
                self.mm_registry,
                is_encoder_data=True
            )
            tts_dummy_data = self.input_registry.dummy_data_for_profiling(
                self.model_config,
                seq_len,
                self.mm_registry,
                is_encoder_data=False
            )

            llm_seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: llm_dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                multi_modal_data=llm_dummy_data.multi_modal_data,
                multi_modal_placeholders=llm_dummy_data.multi_modal_placeholders
            )
            tts_seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: tts_dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(AvatarSequenceGroupMetadata(
                llm_seq_group_metadata=llm_seq,
                tts_seq_group_metadata=tts_seq
            ))

        # Run the model with the dummy inputs.
        (
            llm_head_size, llm_num_layers, llm_num_kv_heads,
            tts_head_size, tts_num_layers, tts_num_kv_heads
        ) = get_avatar_param(self.model_config)
        num_layers = llm_num_layers + tts_num_layers

        # use an empty tensor instead of `None`` to force Dynamo to pass it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as a placeholder (it has wide hardware support).
        kv_caches = [torch.tensor([], dtype=torch.float32, device=self.device) for _ in range(num_layers)]
        finished_requests_ids = [seq.llm_seq_group_metadata.request_id for seq in seqs]
        model_input = self.prepare_model_input(seqs, finished_requests_ids=finished_requests_ids, is_prompt=True)
        intermediate_tensors = None
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def make_model_input_from_broadcasted_tensor_dict(self, tensor_dict: Dict[str, Any]) -> AvatarModelInput:
        return AvatarModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[AvatarSequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None,
        is_prompt=None,
    ) -> AvatarModelInput:
        """
        Prepare the model input based on a given sequence group, including metadata for the sampling step.

        Since chunked prefill is not supported for encoder/decoder models, `input_tokens` is assumed to be either entirely 
        prefill tokens or entirely decode tokens.
        """
        generators = self.get_generators(finished_requests_ids)

        llm_seqs = []
        tts_seqs = []
        for seq_group_metadata in seq_group_metadata_list:
            llm_seq_group_metadata = seq_group_metadata.llm_seq_group_metadata
            llm_seqs.append(llm_seq_group_metadata)

            tts_seq_group_metadata = copy.deepcopy(seq_group_metadata.tts_seq_group_metadata)
            if len(seq_group_metadata_list)<32:
                print(llm_seq_group_metadata)
            for seq_id in tts_seq_group_metadata.seq_data:
                tts_seq_group_metadata.seq_data[seq_id].prompt_token_ids = [
                    token_to_tts_codes(token_id, self.avatar_config.code_layers)
                    for token_id in tts_seq_group_metadata.seq_data[seq_id]._prompt_token_ids
                ]
                tts_seq_group_metadata.seq_data[seq_id].output_token_ids = [
                    token_to_tts_codes(token_id, self.avatar_config.code_layers)
                    for token_id in tts_seq_group_metadata.seq_data[seq_id]._output_token_ids
                ]
            tts_seqs.append(tts_seq_group_metadata)

        llm_model_input = self._prepare_model_input_tensors(llm_seqs, finished_requests_ids)
        llm_sampling_metadata = SamplingMetadata.prepare(
            llm_seqs,
            llm_model_input.seq_lens,
            llm_model_input.query_lens,
            self.device,
            self.pin_memory,
            generators=generators
        )

        tts_model_input = self._prepare_model_input_tensors(tts_seqs, finished_requests_ids)
        tts_sampling_metadata = SamplingMetadata.prepare(
            tts_seqs,
            tts_model_input.seq_lens,
            tts_model_input.query_lens,
            self.device,
            self.pin_memory,
            generators=generators
        )

        model_input = dataclasses.replace(
            tts_model_input,
            encoder_input_tokens=llm_model_input.input_tokens,
            encoder_input_positions=llm_model_input.input_positions,
            encoder_attn_metadata=llm_model_input.attn_metadata,
            sampling_metadata=tts_sampling_metadata,
            encoder_sampling_metadata=llm_sampling_metadata,
            is_prompt=is_prompt,
            virtual_engine=virtual_engine
        )

        return model_input

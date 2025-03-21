import time
import inspect
import dataclasses
import itertools
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
import torch.distributed

from vllm.sequence import SequenceData
from vllm.attention.backends.abstract import (AttentionBackend, AttentionMetadata)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.attention.selector import (get_env_variable_attn_backend, get_global_forced_attn_backend)
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import (get_tensor_model_parallel_rank, graph_capture)
from vllm.distributed import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.multimodal import MultiModalKwargs
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.sampling_params import SamplingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceGroupMetadata)
from vllm.utils import GiB_bytes, make_tensor_with_pad
from vllm.worker.model_runner import (GPUModelRunnerBase, ModelInputForGPUBuilder, ModelInputForGPUWithSamplingMetadata, CUDAGraphRunner)
from vllm.worker.enc_dec_model_runner import EncoderDecoderModelRunner
from vllm.worker.model_runner_base import (_add_attn_metadata_broadcastable_dict, _add_sampling_metadata_broadcastable_dict)


from avatar_infer.models.voice.configuration_voice import AvatarVoiceConfig


logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class AvatarModelInput(ModelInputForGPUWithSamplingMetadata):
    """Used by the AvatarModelInput."""
    encoder_input_tokens: Optional[torch.Tensor] = None
    encoder_input_positions: Optional[torch.Tensor] = None

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
        _add_sampling_metadata_broadcastable_dict(tensor_dict, self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "AvatarModelInput":
        return cast(AvatarModelInput, super().from_broadcasted_tensor_dict(tensor_dict, attn_backend))


class AvatarModelRunner(GPUModelRunnerBase[AvatarModelInput]):
    _model_input_cls: Type[AvatarModelInput] = AvatarModelInput
    _builder_cls: Type[ModelInputForGPUBuilder] = ModelInputForGPUBuilder

    def __init__(self, original_model_runner: EncoderDecoderModelRunner):
        # Store the original model runner
        self.original_model_runner = original_model_runner

        # Forward all attributes from the original model runner
        for attr_name in dir(original_model_runner):
            if not attr_name.startswith('__') and not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(original_model_runner, attr_name))
                
        # add config
        self.avater_config: AvatarVoiceConfig = original_model_runner.vllm_config.model_config.hf_config

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
        input_tokens = torch.zeros([max_batch_size, self.avater_config.code_layers], dtype=torch.long, device=self.device)
        input_positions = torch.zeros(max_batch_size, dtype=torch.long, device=self.device)

        if self.model_config.uses_mrope:
            input_positions = torch.tile(input_positions, (3, 1)).cuda(device=self.device)

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
        with self.attn_state.graph_capture(max_batch_size), graph_capture(self.device) as graph_capture_context:
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
                    attn_metadata = self.attn_state.graph_capture_get_metadata_for_batch(
                        batch_size,
                        is_encoder_decoder_model=self.model_config.is_encoder_decoder)
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
                        "intermediate_inputs": intermediate_inputs[:batch_size] if intermediate_inputs is not None else None,
                        "kv_caches": kv_caches[virtual_engine],
                        "attn_metadata": attn_metadata,
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

                    if self.model_config.is_encoder_decoder:
                        self._update_inputs_to_capture_for_enc_dec_model(capture_inputs)

                    # Execute capture
                    with set_forward_context(attn_metadata, self.vllm_config, virtual_engine):
                        graph_runner.capture(**capture_inputs)

                    self.graph_memory_pool = graph_runner.graph.pool()
                    self.graph_runners[virtual_engine][batch_size] = graph_runner

        # Record end time and memory usage
        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory

        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                   elapsed_time, cuda_graph_size / GiB_bytes)

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

        if (model_input.attn_metadata is not None and 
            model_input.attn_metadata.prefill_metadata is None and
            model_input.attn_metadata.decode_metadata.use_cuda_graph):
            assert model_input.input_tokens is not None
            graph_batch_size = model_input.input_tokens.shape[0]
            model_executable = self.graph_runners[model_input.virtual_engine][graph_batch_size]
        else:
            model_executable = self.model

        seqlen_agnostic_kwargs = {
            "finished_requests_ids": model_input.finished_requests_ids,
            "request_ids_to_seq_ids": model_input.request_ids_to_seq_ids,
        } if self.has_inner_state else {}

        multi_modal_kwargs = model_input.multi_modal_kwargs or {}
        with set_forward_context(model_input.attn_metadata, self.vllm_config, model_input.virtual_engine):
            hidden_or_intermediate_states = model_executable(
                input_ids=model_input.input_tokens,
                positions=model_input.input_positions,
                encoder_input_ids=model_input.encoder_input_tokens,
                encoder_positions=model_input.encoder_input_positions,
                kv_caches=kv_caches,
                attn_metadata=model_input.attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **MultiModalKwargs.as_kwargs(multi_modal_kwargs, device=self.device),
                **seqlen_agnostic_kwargs)

        logits = self.model.compute_logits(hidden_or_intermediate_states, model_input.sampling_metadata)

        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        return [output]

    def make_model_input_from_broadcasted_tensor_dict(self, tensor_dict: Dict[str, Any]) -> AvatarModelInput:
        return AvatarModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> AvatarModelInput:
        """
        Prepare the model input based on a given sequence group, including metadata for the sampling step.

        Since chunked prefill is not supported for encoder/decoder models, `input_tokens` is assumed to be either entirely 
        prefill tokens or entirely decode tokens.
        """
        model_input = self._prepare_model_input_tensors(seq_group_metadata_list, finished_requests_ids)
        input_tokens, input_positions, attn_metadata = self._prepare_voice_input_tensors(model_input)
        model_input = dataclasses.replace(
            model_input,
            input_tokens=input_tokens,
            input_positions=input_positions,
            attn_metadata=attn_metadata,
        )

        attn_metadata, encoder_input_tokens_tensor, encoder_input_positions_tensor = (
            self._prepare_encoder_model_input_tensors(seq_group_metadata_list, model_input)
        )

        # Inject attn_metadata encoder/cross-attention fields & encoder input tokens/positions into model_input.
        # Frozen dataclass fields cannot be modified, so use dataclasses.replace to construct a new model input instance.
        model_input = dataclasses.replace(
            model_input,
            attn_metadata=attn_metadata,
            encoder_input_tokens=encoder_input_tokens_tensor,
            encoder_input_positions=encoder_input_positions_tensor,
        )

        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            model_input.seq_lens,
            model_input.query_lens,
            self.device,
            self.pin_memory,
            generators=generators
        )

        is_prompt = seq_group_metadata_list[0].is_prompt if seq_group_metadata_list else None

        return dataclasses.replace(
            model_input,
            sampling_metadata=sampling_metadata,
            is_prompt=is_prompt,
            virtual_engine=virtual_engine
        )

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

            decoder_dummy_data = self.input_registry.dummy_data_for_profiling(
                self.model_config,
                seq_len,
                self.mm_registry,
                is_encoder_data=False
            )

            encoder_dummy_data = self.input_registry.dummy_data_for_profiling(
                self.model_config,
                seq_len,
                self.mm_registry,
                is_encoder_data=True
            )

            # Having more tokens is over-conservative but otherwise fine
            assert len(decoder_dummy_data.seq_data.prompt_token_ids) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(decoder_dummy_data.seq_data.prompt_token_ids)}"
            )

            assert decoder_dummy_data.multi_modal_data is None or encoder_dummy_data.multi_modal_data is None, (
                "Multi-modal data can't be provided in both encoder and decoder"
            )

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: decoder_dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                encoder_seq_data=encoder_dummy_data.seq_data,
                cross_block_table=None,
                multi_modal_data=decoder_dummy_data.multi_modal_data or encoder_dummy_data.multi_modal_data,
                multi_modal_placeholders=decoder_dummy_data.multi_modal_placeholders or encoder_dummy_data.multi_modal_placeholders
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.hf_config.num_hidden_layers + (
            self.model_config.hf_config.tts_adapter_hidden_layers
            *
            self.model_config.hf_config.code_layers
        )

        # use an empty tensor instead of `None`` to force Dynamo to pass it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as a placeholder (it has wide hardware support).
        kv_caches = [torch.tensor([], dtype=torch.float32, device=self.device) for _ in range(num_layers)]
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.cuda.synchronize()
        return

    def _prepare_voice_input_tensors(self, model_input: AvatarModelInput) -> AttentionMetadata:
        """Prepare input tensors for voice model by reshaping tokens and adjusting attention metadata.
        
        Args:
            model_input: Input data containing tokens and attention metadata
            
        Returns:
            Tuple of (input_tokens, input_positions, attn_metadata)
        """
        code_layers = self.avater_config.code_layers

        # Reshape input tokens to (batch_size * code_layers)
        input_tokens = model_input.input_tokens.view(-1, code_layers)

        # Update attention metadata dimensions
        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None
        
        # Adjust all sequence length related fields by dividing by code_layers
        attn_metadata.num_prefill_tokens //= code_layers
        attn_metadata.seq_lens = [item // code_layers for item in attn_metadata.seq_lens]
        attn_metadata.seq_lens_tensor //= code_layers
        attn_metadata.max_prefill_seq_len //= code_layers
        attn_metadata.max_query_len //= code_layers
        attn_metadata.query_start_loc //= code_layers
        attn_metadata.seq_start_loc //= code_layers

        # Reshape input positions tensor
        input_positions = (model_input.input_positions
                         .view(len(attn_metadata.seq_lens), -1)
                         [:, :attn_metadata.seq_lens[0]]
                         .contiguous()
                         .view(-1))

        return input_tokens, input_positions, attn_metadata

    def _prepare_encoder_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        model_input: AvatarModelInput,
    ) -> Tuple[AttentionMetadata, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Helper method to prepare the encoder- and cross-attn-related model inputs based on a given sequence group. 
        These additional inputs are used to augment an already-computed `AvatarModelInput` data structure which already 
        has decoder-related model inputs populated.

        Sets the following attn_metadata fields:
        * `num_encoder_tokens`
        * `encoder_seq_lens`
        * `encoder_seq_lens_tensor`
        * `max_encoder_seq_len`
        * `cross_slot_mapping`
        * `cross_block_tables`

        Constructs a new model inputs data structure, based on:
        1) the existing fields in the `model_inputs` argument
        Constructs a new model inputs data structure, based on
        (1) the existing fields in the `model_inputs` argument,
        and (2) the following additional fields which are
        computed (or in the case of `attn_metadata`, updated) 
        by this function:
        * attn_metadata
        * encoder_input_tokens
        * encoder_input_positions

        Arguments:

        * seq_group_metadata_list: list of sequence groups for which to
                                   compute inputs
        * model_inputs: model inputs data structure with decoder-oriented
                        fields already computed.

        Return:

        * Updated model inputs data structure
        """

        if len(seq_group_metadata_list) == 0:
            return (model_input.attn_metadata, None, None)

        # Since we are not supporting chunked prefill either the entire
        # batch is prefill or it is decode
        is_prompt = seq_group_metadata_list[0].is_prompt

        # Build encoder inputs
        encoder_seq_lens: List[int] = []
        if is_prompt:
            # Prefill phase.
            cross_block_tables = self._empty_int32_tensor().view(
                len(seq_group_metadata_list), -1)

            # Extract input tokens/positions, cross-attention slot-mapping,
            # & seq len from each sequence group metadata
            (
                encoder_input_tokens,
                encoder_input_positions,
                cross_slot_mapping,
            ) = (
                [],
                [],
                [],
            )
            for seq_group_metadata in seq_group_metadata_list:
                # Build seq lens
                seq_len = seq_group_metadata.encoder_seq_data.get_len()
                token_ids = seq_group_metadata.encoder_seq_data.get_token_ids()
                encoder_seq_lens.append(seq_len)

                # Build slot mapping
                is_profile_run = (seq_group_metadata.block_tables is None)
                if is_profile_run:
                    # During memory profiling, the block tables are not
                    # initialized yet. In this case, we just use a dummy
                    # slot mapping.
                    # In embeddings, the block tables are {seq_id: None}.
                    cross_slot_mapping.extend([PAD_SLOT_ID] * seq_len)
                else:
                    for i in range(0, seq_len):
                        block_number = seq_group_metadata.cross_block_table[
                            i // self.block_size]
                        block_offset = i % self.block_size
                        slot = block_number * self.block_size + block_offset
                        cross_slot_mapping.append(slot)

                # Build encoder input tokens
                encoder_input_tokens.extend(token_ids)
                encoder_input_positions.extend(list(range(0, seq_len)))

            # Convert tokens/positions & cross-attention
            # slot-mapping to encoder input tensors
            encoder_input_tokens_tensor = self._list_to_long_tensor(
                encoder_input_tokens)
            encoder_input_positions_tensor = self._list_to_long_tensor(
                encoder_input_positions)
            cross_slot_mapping_tensor = self._list_to_long_tensor(
                cross_slot_mapping)

        else:
            # Decode phase.
            encoder_input_tokens_tensor = self._empty_long_tensor()
            encoder_input_positions_tensor = self._empty_long_tensor()
            cross_slot_mapping_tensor = self._empty_long_tensor()
            # Extract cross-attention block tables &
            # seq len from each sequence group metadata.
            # Cross-attention block tables are empty
            # during vLLM memory profiling.
            cross_block_tables = []
            for seq_group_metadata in seq_group_metadata_list:
                for _ in range(len(seq_group_metadata.seq_data)):
                    encoder_seq_lens.append(
                        seq_group_metadata.encoder_seq_data.get_len())
                    cross_block_table = seq_group_metadata.cross_block_table
                    cross_block_tables.append([] if (
                        cross_block_table is None) else cross_block_table)

            if (model_input.attn_metadata is not None
                    and model_input.attn_metadata.use_cuda_graph):
                # We will be using CUDA graph replay for this decode.
                max_len_of_block_table = self.get_max_block_per_batch()
                batch_size = len(encoder_seq_lens)
                graph_batch_size = self.vllm_config.pad_for_cudagraph(
                    batch_size)
                assert graph_batch_size >= batch_size
                cuda_graph_pad_size = graph_batch_size - batch_size
                # extend the cross_block_tables and encoder_seq_lens to match
                # the graph_batch_size.
                cross_block_tables.extend([[]
                                           for _ in range(cuda_graph_pad_size)
                                           ])
                encoder_seq_lens.extend(
                    itertools.repeat(1, cuda_graph_pad_size))

            else:
                max_len_of_block_table = max(
                    len(block_table) for block_table in cross_block_tables)

            cross_block_tables = make_tensor_with_pad(
                cross_block_tables,
                max_len=max_len_of_block_table,
                pad=0,
                dtype=torch.int32,
                device=self.device,
            )

        # Compute encoder sequence lengths & encoder
        # sequence starting offset tensors
        max_encoder_seq_len = max(encoder_seq_lens, default=0)
        encoder_seq_lens_tensor = self._list_to_int32_tensor(encoder_seq_lens)
        encoder_seq_start_loc = torch.zeros(encoder_seq_lens_tensor.shape[0] +
                                            1,
                                            dtype=torch.int32,
                                            device=self.device)
        torch.cumsum(encoder_seq_lens_tensor,
                     dim=0,
                     dtype=encoder_seq_start_loc.dtype,
                     out=encoder_seq_start_loc[1:])

        # Update attention metadata with encoder-oriented attributes
        attn_metadata = model_input.attn_metadata
        assert attn_metadata is not None
        (
            attn_metadata.num_encoder_tokens,
            attn_metadata.encoder_seq_lens,
            attn_metadata.encoder_seq_lens_tensor,
            attn_metadata.max_encoder_seq_len,
            attn_metadata.encoder_seq_start_loc,
            attn_metadata.cross_slot_mapping,
            attn_metadata.cross_block_tables,
        ) = (
            sum(encoder_seq_lens),
            encoder_seq_lens,
            encoder_seq_lens_tensor,
            max_encoder_seq_len,
            encoder_seq_start_loc,
            cross_slot_mapping_tensor,
            cross_block_tables,
        )

        return (attn_metadata, encoder_input_tokens_tensor, encoder_input_positions_tensor)

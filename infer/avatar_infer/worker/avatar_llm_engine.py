import time
from typing import (Union, List, Mapping, Optional)

from vllm.engine.llm_engine import LLMEngine, SchedulerContext, SchedulerOutputState
from vllm.core.scheduler import ScheduledSequenceGroup
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.inputs import ProcessorInputs
from vllm.sequence import (
    ExecuteModelRequest,
    Sequence,
    SequenceGroupMetadata,
    SequenceGroup,
    SequenceOutput,
    SequenceGroupOutput,
    ParallelSampleSequenceGroup
)


from avatar_infer.dataclass.sequence import (
    TTSSequence,
    AvatarSequenceGroup,
    AvatarSamplerOutput,
    AvatarSequenceGroupMetadata,
)
from avatar_infer.dataclass.outputs import (
    PoolingRequestOutput,
    AvatarRequestOutput,
    AvatarRequestOutputFactory
)


logger = init_logger(__name__)


class AvatarLLMEngine(LLMEngine):
    """
    Extended LLMEngine class specifically for Avatar generation, 
    handling both LLM and TTS (Text-to-Speech) outputs.
    """

    def _add_processed_request(
        self,
        request_id: str,
        processed_inputs: ProcessorInputs,
        params: Union[SamplingParams],
        arrival_time: float,
        lora_request: Optional[LoRARequest],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        trace_headers: Optional[Mapping[str, str]] = None,
        priority: int = 0,
    ) -> Optional[SequenceGroup]:
        """Add a processed request to the engine's request pool.
        return the created sequence group.
        """
        if isinstance(params, SamplingParams) and params.n > 1:
            ParallelSampleSequenceGroup.add_request(
                request_id,
                self,
                params,
                processed_inputs=processed_inputs,
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
            )
            return None

        self._validate_model_inputs(processed_inputs, lora_request)
        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        eos_token_id = self.input_preprocessor.get_eos_token_id(lora_request)
        eoa_token_id = self.model_config.hf_config.eoa_token_id

        llm_inputs = processed_inputs["encoder"]
        tts_inputs = processed_inputs["decoder"]
        llm_seq = Sequence(seq_id, llm_inputs, block_size, eos_token_id, lora_request, prompt_adapter_request)
        tts_seq = TTSSequence(seq_id, tts_inputs, block_size, eoa_token_id, lora_request, prompt_adapter_request)

        # Create a SequenceGroup based on SamplingParams or PoolingParams
        llm_seq_group = self._create_sequence_group_with_sampling(
            request_id,
            llm_seq,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=None,
            priority=priority
        )
        tts_seq_group = self._create_sequence_group_with_sampling(
            request_id,
            tts_seq,
            params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            prompt_adapter_request=prompt_adapter_request,
            encoder_seq=None,
            priority=priority
        )

        avatar_seq_group = AvatarSequenceGroup(
            llm_seq_group=llm_seq_group,
            tts_seq_group=tts_seq_group,
        )

        # Add the sequence group to the scheduler with least unfinished seqs.
        costs = [
            scheduler.get_num_unfinished_seq_groups()
            for scheduler in self.scheduler
        ]
        min_cost_scheduler = self.scheduler[costs.index(min(costs))]
        min_cost_scheduler.add_seq_group(avatar_seq_group)

        return avatar_seq_group

    def _process_model_outputs(
        self,
        ctx: SchedulerContext,
        request_id: Optional[str] = None
    ) -> None:
        """
        Process model outputs for scheduled sequence groups and generate responses.
        
        Args:
            ctx: The virtual engine context containing output queue and other state
            request_id: If provided, only this specific request will be processed
        """
        now = time.time()

        # Nothing to process if output queue is empty
        if len(ctx.output_queue) == 0:
            return None

        # Extract the pending output data
        if request_id:
            # When processing just one request, keep it in the queue for now
            (
                outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                is_last_step, is_first_step_output, skip
            ) = ctx.output_queue[0]
        else:
            # Otherwise, pop the first item from the output queue
            (
                outputs, seq_group_metadata_list, scheduler_outputs, is_async,
                is_last_step, is_first_step_output, skip
            ) = ctx.output_queue.popleft()

        # Ensure metadata and scheduled groups match
        assert len(seq_group_metadata_list) == len(scheduler_outputs.scheduled_seq_groups)

        # Handle outputs (speculative decoding may produce multiple outputs)
        outputs_by_sequence_group: List[List[SequenceGroupOutput]] = outputs

        # Determine which sequence groups to process
        if request_id:
            # Find the specific sequence group with the requested ID
            indices = []
            for i, seq_group_meta in enumerate(seq_group_metadata_list):
                if seq_group_meta.request_id == request_id:
                    assert i not in skip  # Ensure we're not processing it twice
                    indices.append(i)
                    break

            # If request_id not found, nothing to process
            if not indices:
                return
        else:
            # Process all sequence groups
            indices = range(len(seq_group_metadata_list))

        # Track which sequence groups were already finished or finish now
        finished_before: List[int] = []
        finished_now: List[int] = []

        # Process each selected sequence group
        for i in indices:
            if i in skip:
                continue  # Skip sequence groups marked to be skipped

            seq_group_meta: AvatarSequenceGroupMetadata = seq_group_metadata_list[i]
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]
            seq_group: AvatarSequenceGroup = scheduled_seq_group.seq_group

            # Skip already finished sequence groups
            if seq_group.is_finished():
                finished_before.append(i)
                continue

            # Get the output for this sequence group
            output: List[SequenceGroupOutput] = [outputs_by_sequence_group[0][i]]

            # Update token counts for proper tracking
            if not is_async:
                # Standard token count update
                seq_group.update_num_computed_tokens(seq_group_meta.token_chunk_size or 0)

            # Update sequence metrics from model output timing data
            if outputs:
                for o in outputs:
                    if (
                        isinstance(o, AvatarSamplerOutput)
                        and seq_group.llm_seq_group.metrics is not None
                        and seq_group.tts_seq_group.metrics is not None
                    ):
                        # Add model forward and execution times to metrics
                        if seq_group.llm_seq_group.metrics.model_forward_time is not None:
                            seq_group.llm_seq_group.metrics.model_forward_time += (o.model_forward_time or 0)
                        else:
                            seq_group.llm_seq_group.metrics.model_forward_time = (o.model_forward_time)
                            
                        if seq_group.llm_seq_group.metrics.model_execute_time is not None:
                            seq_group.llm_seq_group.metrics.model_execute_time += (o.model_execute_time or 0)
                        else:
                            seq_group.llm_seq_group.metrics.model_execute_time = (o.model_execute_time)

                        # Add model forward and execution times to metrics
                        if seq_group.tts_seq_group.metrics.model_forward_time is not None:
                            seq_group.tts_seq_group.metrics.model_forward_time += (o.model_forward_time or 0)
                        else:
                            seq_group.tts_seq_group.metrics.model_forward_time = (o.model_forward_time)
                            
                        if seq_group.tts_seq_group.metrics.model_execute_time is not None:
                            seq_group.tts_seq_group.metrics.model_execute_time += (o.model_execute_time or 0)
                        else:
                            seq_group.tts_seq_group.metrics.model_execute_time = (o.model_execute_time)

            # Process outputs based on runner type
            if self.model_config.runner_type == "pooling":
                self._process_sequence_group_outputs(seq_group, output)
            else:
                # Process probability logs and sampling results
                self.output_processor.process_prompt_logprob(seq_group, output)
                if seq_group_meta.llm_seq_group_metadata.do_sample:
                    self.output_processor.process_outputs(seq_group, output, is_async)

            # Check if the sequence group has finished after processing
            if seq_group.is_finished():
                finished_now.append(i)

        # Generate final outputs for sequence groups that finished in this iteration
        for i in finished_now:
            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]
            seq_group = scheduled_seq_group.seq_group

            # Set timing information
            seq_group.llm_seq_group.maybe_set_first_token_time(now)
            seq_group.tts_seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.llm_seq_group.set_last_token_time(now)
                seq_group.tts_seq_group.set_last_token_time(now)

            # Create request output
            request_output = AvatarRequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs
            )

            if request_output:
                ctx.request_outputs.append(request_output)

        # Special handling for single request processing
        if request_id:
            assert len(indices) == 1
            skip.append(indices[0])  # Mark this request as processed

            # Call the output callback if needed
            if (finished_now and self.process_request_outputs_callback is not None):
                self.process_request_outputs_callback(ctx.request_outputs)
                ctx.request_outputs.clear()
            return

        # Free resources for finished sequence groups
        if finished_now:
            for scheduler in self.scheduler:
                scheduler.free_finished_seq_groups()

        # Create outputs for active (not finished) sequence groups
        for i in indices:
            if i in skip or i in finished_before or i in finished_now:
                continue  # Skip already processed groups

            scheduled_seq_group = scheduler_outputs.scheduled_seq_groups[i]
            seq_group = scheduled_seq_group.seq_group

            # Set timing information
            seq_group.llm_seq_group.maybe_set_first_token_time(now)
            seq_group.tts_seq_group.maybe_set_first_token_time(now)
            if not seq_group.is_prefill():
                seq_group.llm_seq_group.set_last_token_time(now)
                seq_group.tts_seq_group.set_last_token_time(now)

            # Create request output
            request_output = AvatarRequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs)

            if request_output:
                ctx.request_outputs.append(request_output)

        # Process ignored sequence groups (if they need output)
        for seq_group in scheduler_outputs.ignored_seq_groups:
            params = seq_group.sampling_params
            # Skip delta outputs for unfinished sequences
            if params is not None and params.output_kind == (RequestOutputKind.DELTA) and not seq_group.is_finished():
                continue

            request_output = AvatarRequestOutputFactory.create(
                seq_group,
                self.seq_id_to_seq_group,
                use_cache=self.use_cached_outputs,
            )
            if request_output:
                ctx.request_outputs.append(request_output)

        # Process all collected outputs
        if (ctx.request_outputs and self.process_request_outputs_callback is not None):
            self.process_request_outputs_callback(ctx.request_outputs)
            ctx.request_outputs.clear()

        # Record statistics for async processing
        if is_async:
            # Log statistics
            self.do_log_stats(scheduler_outputs, outputs, finished_before, skip)
            # Record tracing information
            self.do_tracing(scheduler_outputs, finished_before)

        return None

    def _advance_to_next_step(
        self,
        output: AvatarSamplerOutput,
        seq_group_metadata_list: List[AvatarSequenceGroupMetadata],
        scheduled_seq_groups: List[ScheduledSequenceGroup]
    ) -> None:
        """
        Advance to the next generation step by appending tokens to sequences.
        This is needed for async forward pass to the next step.
        
        Args:
            output: Model output from the current step
            seq_group_metadata_list: Metadata for sequence groups
            scheduled_seq_groups: The scheduled sequence groups
        """
        # Process each sequence group
        for seq_group_metadata, sequence_group_outputs, scheduled_seq_group in zip(
            seq_group_metadata_list, output, scheduled_seq_groups
        ):
            seq_group: AvatarSequenceGroup = scheduled_seq_group.seq_group

            # Skip finished sequences
            if seq_group.is_finished():
                continue

            # If sampling is enabled (not greedy decoding)
            if seq_group_metadata.llm_seq_group_metadata.do_sample:
                # Avatar model has both LLM and TTS sequences
                assert len(seq_group.llm_seq_group.seqs) == 1
                assert len(seq_group.tts_seq_group.seqs) == 1
                assert len(sequence_group_outputs.samples) == 1, (
                    "Async output processor expects a single sample (sampling_params.n == 1)"
                )

                # Get the LLM and TTS samples
                llm_sample: SequenceOutput = sequence_group_outputs.llm_samples[0]
                tts_sample: SequenceOutput = sequence_group_outputs.tts_samples[0]

                # Get the corresponding sequences
                llm_seq = seq_group.llm_seq_group.seqs[0]
                tts_seq = seq_group.tts_seq_group.seqs[0]

                # Update token counts
                llm_seq.data.update_num_computed_tokens(
                    llm_seq.data.get_len()
                    if seq_group_metadata.llm_seq_group_metadata.is_prompt
                    else seq_group_metadata.llm_seq_group_metadata.token_chunk_size
                )
                tts_seq.data.update_num_computed_tokens(
                    tts_seq.data.get_len()
                    if seq_group_metadata.llm_seq_group_metadata.is_prompt
                    else 1
                )

                # Append new tokens to sequences
                # First append LLM token with its logprobs
                llm_seq.append_token_id(llm_sample.output_token, llm_sample.logprobs)
                tts_seq.append_token_id(tts_sample.output_token, {tts_sample.output_token: llm_sample.logprobs[llm_sample.output_token]})

    def step(self) -> List[Union[AvatarRequestOutput, PoolingRequestOutput]]:
        """Performs one decoding iteration and returns newly generated results.

        .. figure:: https://i.imgur.com/sv2HssD.png
            :alt: Overview of the step function
            :align: center

            Overview of the step function.

        Details:
            - Step 1: Schedules the sequences to be executed in the next
              iteration and the token blocks to be swapped in/out/copy.

                - Depending on the scheduling policy,
                  sequences may be `preempted/reordered`.
                - A Sequence Group (SG) refer to a group of sequences
                  that are generated from the same prompt.

            - Step 2: Calls the distributed executor to execute the model.
            - Step 3: Processes the model output. This mainly includes:

                - Decodes the relevant outputs.
                - Updates the scheduled sequence groups with model outputs
                  based on its `sampling parameters` (`use_beam_search` or not).
                - Frees the finished sequence groups.

            - Finally, it creates and returns the newly generated results.

        Example:
            >>> # Please see the example/ folder for more detailed examples.
            >>>
            >>> # initialize engine and request arguments
            >>> engine = LLMEngine.from_engine_args(engine_args)
            >>> example_inputs = [(0, "What is LLM?",
            >>>    SamplingParams(temperature=0.0))]
            >>>
            >>> # Start the engine with an event loop
            >>> while True:
            >>>     if example_inputs:
            >>>         req_id, prompt, sampling_params = example_inputs.pop(0)
            >>>         engine.add_request(str(req_id),prompt,sampling_params)
            >>>
            >>>     # continue the request processing
            >>>     request_outputs = engine.step()
            >>>     for request_output in request_outputs:
            >>>         if request_output.finished:
            >>>             # return or show the request output
            >>>
            >>>     if not (engine.has_unfinished_requests() or example_inputs):
            >>>         break
        """
        if self.parallel_config.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "Pipeline parallelism is only supported through AsyncLLMEngine "
                "as performance will be severely degraded otherwise.")

        # For llm_engine, there is no pipeline parallel support, so the engine
        # used is always 0.
        virtual_engine = 0

        # These are cached outputs from previous iterations. None if on first
        # iteration
        cached_outputs = self.cached_scheduler_outputs[virtual_engine]
        seq_group_metadata_list = cached_outputs.seq_group_metadata_list
        scheduler_outputs = cached_outputs.scheduler_outputs
        allow_async_output_proc = cached_outputs.allow_async_output_proc

        ctx = self.scheduler_contexts[virtual_engine]

        # Clear outputs for each new scheduler iteration
        ctx.request_outputs.clear()

        # Skip the scheduler if there are any remaining steps in the seq groups.
        # This ensures that the scheduler is only called again when the current
        # batch has completed.
        if not self._has_remaining_steps(seq_group_metadata_list):
            # Schedule iteration
            (
                seq_group_metadata_list, scheduler_outputs, allow_async_output_proc
            ) = self.scheduler[virtual_engine].schedule()

            ctx.seq_group_metadata_list = seq_group_metadata_list
            ctx.scheduler_outputs = scheduler_outputs

            finished_requests_ids = self.scheduler[virtual_engine].get_and_reset_finished_requests_ids()

            # Maybe switch from async mode to sync mode
            if not allow_async_output_proc and len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)

            if (self.scheduler_config.is_multi_step and scheduler_outputs.num_lookahead_slots > 0):
                # cache the scheduler outputs for the next iteration if we have
                # lookahead slots
                self._cache_scheduler_outputs_for_multi_step(
                    virtual_engine,
                    seq_group_metadata_list,
                    scheduler_outputs,
                    allow_async_output_proc
                )
        else:
            finished_requests_ids = list()

        assert seq_group_metadata_list is not None
        assert scheduler_outputs is not None

        if not scheduler_outputs.is_empty():
            # Check if we have a cached last_output from the previous iteration.
            # For supporting PP this is probably the best way to pass the
            # sampled_token_ids, as a separate broadcast over all the PP stages
            # will cause one virtual engine's microbatch to block the pipeline.
            last_sampled_token_ids = self._get_last_sampled_token_ids(virtual_engine)

            execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                num_lookahead_slots=scheduler_outputs.num_lookahead_slots,
                running_queue_size=scheduler_outputs.running_queue_size,
                finished_requests_ids=finished_requests_ids,
                # We use ExecuteModelRequest to pass the last sampled_token_ids
                # to each of the non-last PP stages for in-place prepare_input.
                last_sampled_token_ids=last_sampled_token_ids)

            if allow_async_output_proc:
                execute_model_req.async_callback = self.async_callbacks[
                    virtual_engine]

            outputs = self.model_executor.execute_model(
                execute_model_req=execute_model_req)

            # We need to do this here so that last step's sampled_token_ids can
            # be passed to the next iteration for PP.
            if self.scheduler_config.is_multi_step:
                self._update_cached_scheduler_output(virtual_engine, outputs)
        else:
            # Nothing scheduled => If there is pending async postprocessor,
            # then finish it here.
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            # No outputs in this case
            outputs = []

        # Finish the current step for all the sequence groups.
        if self.scheduler_config.is_multi_step:
            for seq_group in seq_group_metadata_list:
                seq_group.finish_step()

        if not self._has_remaining_steps(seq_group_metadata_list):
            # clear the cache if we have finished all the steps.
            if self.scheduler_config.is_multi_step:
                self.cached_scheduler_outputs[0] = SchedulerOutputState()

            # is_first_step_output is True only when the num_steps of all
            # the sequences are 1. When the num_steps > 1,
            # multi_step_model_runner does the first-step output append.
            is_first_step_output: bool = (
                False if not seq_group_metadata_list 
                else seq_group_metadata_list[0].llm_seq_group_metadata.state.num_steps == 1
            )

            # Add results to the output_queue
            ctx.append_output(
                outputs=outputs,
                seq_group_metadata_list=seq_group_metadata_list,
                scheduler_outputs=scheduler_outputs,
                is_async=allow_async_output_proc,
                is_last_step=True,
                is_first_step_output=is_first_step_output
            )

            if outputs and allow_async_output_proc:
                assert len(outputs) == 1, (
                    "Async postprocessor expects only a single output set")

                self._advance_to_next_step(
                    outputs[0], seq_group_metadata_list,
                    scheduler_outputs.scheduled_seq_groups)

            # Check if need to run the usual non-async path
            if not allow_async_output_proc:
                self._process_model_outputs(ctx=ctx)

                # Log stats.
                self.do_log_stats(scheduler_outputs, outputs)

                # Tracing
                self.do_tracing(scheduler_outputs)
        else:
            # Multi-step case
            return ctx.request_outputs

        if not self.has_unfinished_requests():
            # Drain async postprocessor (if exists)
            if len(ctx.output_queue) > 0:
                self._process_model_outputs(ctx=ctx)
            assert len(ctx.output_queue) == 0

            # Stop the execute model loop in parallel workers until there are
            # more requests to process. This avoids waiting indefinitely in
            # torch.distributed ops which may otherwise timeout, and unblocks
            # the RPC thread in the workers so that they can process any other
            # queued control plane messages, such as add/remove lora adapters.
            logger.debug("Stopping remote worker execution loop.")
            self.model_executor.stop_remote_worker_execution_loop()

        return ctx.request_outputs

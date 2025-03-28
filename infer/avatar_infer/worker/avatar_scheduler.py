import enum
import os
import random
import time
from collections import deque
from typing import Set, Tuple, Deque, Dict, List, Optional, Callable

from vllm.core.scheduler import (
    SchedulerConfig,
    Scheduler,
    SchedulerOutputs,
    SchedulerPrefillOutputs,
    SchedulingBudget,
    PartialPrefillMetadata,
    ScheduledSequenceGroup,
    SchedulerRunningOutputs,
    SchedulerSwappedInOutputs,
    PreemptionMode,
    seq_group_metadata_builder,
    scheduled_seq_group_builder,
)
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta,
                           SequenceStage, SequenceStatus)
from vllm.utils import Device, PyObjectCache


from avatar_infer.dataclass.sequence import (
    AvatarSequenceGroup,
    AvatarSequenceGroupMetadata,
)


logger = init_logger(__name__)


class AvatarScheduler(Scheduler):
    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config,
        lora_config,
        pipeline_parallel_size: int = 1,
        output_proc_callback: Optional[Callable] = None,
    ):
        super().__init__(scheduler_config, cache_config, lora_config, pipeline_parallel_size, output_proc_callback)

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        if num_gpu_blocks:
            num_gpu_blocks //= pipeline_parallel_size

        num_cpu_blocks = self.cache_config.num_cpu_blocks
        if num_cpu_blocks:
            num_cpu_blocks //= pipeline_parallel_size

        self.tts_block_manager = BlockSpaceManager.get_block_space_manager_class("selfattn")(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
        )

        self._tts_seq_group_metadata_cache: List[PyObjectCache] = []
        self._tts_scheduled_seq_group_cache: List[PyObjectCache] = []
        for i in range(self.num_cache_iters):
            self._tts_seq_group_metadata_cache.append(PyObjectCache(seq_group_metadata_builder))
            self._tts_scheduled_seq_group_cache.append(PyObjectCache(scheduled_seq_group_builder))

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, bool]:
        # Schedule sequence groups.
        # This function call changes the internal states of the scheduler
        # such as self.running, self.swapped, and self.waiting.
        scheduler_start_time = time.perf_counter()

        scheduler_outputs: SchedulerOutputs = self._schedule()
        now = time.time()

        if not self.cache_config.enable_prefix_caching:
            llm_common_computed_block_nums = []
            tts_common_computed_block_nums = []

        allow_async_output_proc: bool = self.use_async_output_proc

        # Create input data structures.
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, scheduled_seq_group in enumerate(scheduler_outputs.scheduled_seq_groups):
            seq_group: AvatarSequenceGroup = scheduled_seq_group.seq_group
            token_chunk_size = scheduled_seq_group.token_chunk_size

            seq_group.llm_seq_group.maybe_set_first_scheduled_time(now)
            seq_group.tts_seq_group.maybe_set_first_scheduled_time(now)

            llm_seq_group_metadata = self._seq_group_metadata_cache[self.cache_id].get_object()
            llm_seq_group_metadata.seq_data.clear()
            llm_seq_group_metadata.block_tables.clear()
            tts_seq_group_metadata = self._tts_seq_group_metadata_cache[self.cache_id].get_object()
            tts_seq_group_metadata.seq_data.clear()
            tts_seq_group_metadata.block_tables.clear()

            # seq_id -> SequenceData
            llm_seq_data: Dict[int, SequenceData] = {}
            tts_seq_data: Dict[int, SequenceData] = {}
            # seq_id -> physical block numbers
            llm_block_tables: Dict[int, List[int]] = {}
            tts_block_tables: Dict[int, List[int]] = {}

            tts_encoder_seq_data, tts_cross_block_table = self._allocate_tts_cross_table(seq_group.tts_seq_group)

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                llm_seq_id = seq.llm_seq.seq_id
                llm_seq_data[llm_seq_id] = seq.llm_seq.data
                llm_block_tables[llm_seq_id] = self.block_manager.get_block_table(seq.llm_seq)
                self.block_manager.access_all_blocks_in_seq(seq.llm_seq, now)

                tts_seq_id = seq.tts_seq.seq_id
                tts_seq_data[tts_seq_id] = seq.tts_seq.data
                tts_block_tables[tts_seq_id] = self.tts_block_manager.get_block_table(seq.tts_seq)
                self.tts_block_manager.access_all_blocks_in_seq(seq.tts_seq, now)

            if self.cache_config.enable_prefix_caching:
                llm_common_computed_block_nums = self.block_manager.get_common_computed_block_ids(
                    seq_group.llm_seq_group.get_seqs(
                        status=SequenceStatus.RUNNING
                    )
                )
                tts_common_computed_block_nums = self.tts_block_manager.get_common_computed_block_ids(
                    seq_group.tts_seq_group.get_seqs(
                        status=SequenceStatus.RUNNING
                    )
                )

            do_sample = True
            is_prompt = seq_group.is_prefill()
            # We should send the metadata to workers when the first prefill
            # is sent. Subsequent requests could be chunked prefill or decode.
            is_first_prefill = False
            if is_prompt:
                seqs = seq_group.get_seqs()
                # Prefill has only 1 sequence.
                assert len(seqs) == 1
                num_computed_tokens = seqs[0].llm_seq.data.get_num_computed_tokens()
                is_first_prefill = num_computed_tokens == 0
                # In the next iteration, all prompt tokens are not computed.
                # It means the prefill is chunked, and we don't need sampling.
                # NOTE: We use get_len instead of get_prompt_len because when
                # a sequence is preempted, prefill includes previous generated
                # output tokens.
                if (token_chunk_size + num_computed_tokens < seqs[0].llm_seq.data.get_len()):
                    do_sample = False

            # It assumes the scheduled_seq_groups is ordered by
            # prefill < decoding.
            if is_first_prefill or not self.scheduler_config.send_delta_data:
                llm_seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.llm_seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=llm_seq_data,
                    sampling_params=seq_group.llm_seq_group.sampling_params,
                    block_tables=llm_block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.llm_seq_group.pooling_params,
                    token_chunk_size=token_chunk_size,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=llm_common_computed_block_nums,
                    encoder_seq_data=None,
                    cross_block_table=None,
                    state=seq_group.llm_seq_group.state,
                    token_type_ids=seq_group.llm_seq_group.token_type_ids,
                    # `multi_modal_data` will only be present for the 1st comm
                    # between engine and worker.
                    # the subsequent comms can still use delta, but
                    # `multi_modal_data` will be None.
                    multi_modal_data=(seq_group.llm_seq_group.multi_modal_data if scheduler_outputs.num_prefill_groups > 0 else None),
                    multi_modal_placeholders=(seq_group.llm_seq_group.multi_modal_placeholders if scheduler_outputs.num_prefill_groups > 0 else None),
                    mm_processor_kwargs=seq_group.llm_seq_group.mm_processor_kwargs,
                    prompt_adapter_request=seq_group.prompt_adapter_request,
                )
                tts_seq_group_metadata = SequenceGroupMetadata(
                    request_id=seq_group.tts_seq_group.request_id,
                    is_prompt=is_prompt,
                    seq_data=tts_seq_data,
                    sampling_params=seq_group.tts_seq_group.sampling_params,
                    block_tables=tts_block_tables,
                    do_sample=do_sample,
                    pooling_params=seq_group.tts_seq_group.pooling_params,
                    token_chunk_size=8,
                    lora_request=seq_group.lora_request,
                    computed_block_nums=tts_common_computed_block_nums,
                    encoder_seq_data=tts_encoder_seq_data,
                    cross_block_table=tts_cross_block_table,
                    state=seq_group.tts_seq_group.state,
                    token_type_ids=seq_group.tts_seq_group.token_type_ids,
                )
            else:
                # When SPMD mode is enabled, we only send delta data except for
                # the first request to reduce serialization cost.
                llm_seq_data_delta = {}
                for id, data in llm_seq_data.items():
                    llm_seq_data_delta[id] = data.get_delta_and_reset()
                llm_seq_group_metadata = SequenceGroupMetadataDelta(
                    llm_seq_data_delta,
                    seq_group.llm_seq_group.request_id,
                    llm_block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=token_chunk_size,
                    computed_block_nums=llm_common_computed_block_nums,
                )
                tts_seq_data_delta = {}
                for id, data in tts_seq_data.items():
                    tts_seq_data_delta[id] = data.get_delta_and_reset()
                tts_seq_group_metadata = SequenceGroupMetadataDelta(
                    tts_seq_data_delta,
                    seq_group.tts_seq_group.request_id,
                    tts_block_tables,
                    is_prompt,
                    do_sample=do_sample,
                    token_chunk_size=8,
                    computed_block_nums=tts_common_computed_block_nums,
                )

            seq_group_metadata_list.append(
                AvatarSequenceGroupMetadata(
                    llm_seq_group_metadata=llm_seq_group_metadata,
                    tts_seq_group_metadata=tts_seq_group_metadata
                )
            )

            if allow_async_output_proc:
                allow_async_output_proc = self._allow_async_output_proc(seq_group.llm_seq_group)

        # Now that the batch has been created, we can assume all blocks in the
        # batch will have been computed before the next scheduling invocation.
        # This is because the engine assumes that a failure in model execution
        # will crash the vLLM instance / will not retry.
        for scheduled_seq_group in scheduler_outputs.scheduled_seq_groups:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group.llm_seq_group,
                scheduled_seq_group.token_chunk_size
            )
            self.tts_block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group.tts_seq_group,
                8
            )

        self._seq_group_metadata_cache[self.next_cache_id].reset()
        self._tts_seq_group_metadata_cache[self.next_cache_id].reset()

        scheduler_time = time.perf_counter() - scheduler_start_time
        # Add this to scheduler time to all the sequences that are currently
        # running. This will help estimate if the scheduler is a significant
        # component in the e2e latency.
        for seq_group in self.running:
            if seq_group is not None and seq_group.llm_seq_group.metrics is not None:
                if seq_group.llm_seq_group.metrics.scheduler_time is not None:
                    seq_group.llm_seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.llm_seq_group.metrics.scheduler_time = scheduler_time

            if seq_group is not None and seq_group.tts_seq_group.metrics is not None:
                if seq_group.tts_seq_group.metrics.scheduler_time is not None:
                    seq_group.tts_seq_group.metrics.scheduler_time += scheduler_time
                else:
                    seq_group.tts_seq_group.metrics.scheduler_time = scheduler_time

        # Move to next cache (if exists)
        self.cache_id = self.next_cache_id

        # Return results
        return (seq_group_metadata_list, scheduler_outputs, allow_async_output_proc)

    def _schedule_prefills(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
                that are currently running

        Returns:
            SchedulerPrefillOutputs.
        """
        if budget.remaining_token_budget() == 0:
            # Do nothing: Can't add any more prefill anyway
            return SchedulerPrefillOutputs(
                seq_groups=[],
                ignored_seq_groups=[],
                num_lookahead_slots=self._get_num_lookahead_slots(
                    is_prefill=True, enable_chunking=enable_chunking),
            )
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group: AvatarSequenceGroup = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, ("Waiting sequence group should have only one prompt sequence.")

            if (partial_prefill_metadata is not None and not partial_prefill_metadata.can_schedule(seq_group)):
                leftover_waiting_sequences.appendleft(seq_group)
                waiting_queue.popleft()
                continue
            
            llm_num_new_tokens_uncached, llm_num_new_tokens_cached = (
                self._get_num_new_uncached_and_cached_tokens(
                    seq_group.llm_seq_group,
                    SequenceStatus.WAITING,
                    enable_chunking,
                    budget,
                    partial_prefill_metadata=partial_prefill_metadata,
                )
            )
            llm_num_new_tokens = llm_num_new_tokens_uncached + llm_num_new_tokens_cached

            if not enable_chunking:
                llm_num_prompt_tokens = waiting_seqs[0].llm_seq.get_len()
                assert llm_num_new_tokens == llm_num_prompt_tokens

            llm_prompt_limit = self._get_prompt_limit(seq_group.llm_seq_group)
            if llm_num_new_tokens > llm_prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d",
                    llm_num_new_tokens,
                    llm_prompt_limit,
                )
                for seq in waiting_seqs:
                    seq.llm_seq.status = SequenceStatus.FINISHED_IGNORED
                    seq.tts_seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            num_lookahead_slots: int = 0
            if self.scheduler_config.is_multi_step and enable_chunking:
                num_lookahead_slots = self._get_num_lookahead_slots(True, enable_chunking)

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group.llm_seq_group,
                num_lookahead_slots=num_lookahead_slots
            )
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    llm_num_new_tokens,
                    num_lookahead_slots,
                )
                for seq in waiting_seqs:
                    seq.llm_seq.status = SequenceStatus.FINISHED_IGNORED
                    seq.tts_seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            if (budget.num_batched_tokens  >= self.scheduler_config.max_num_batched_tokens):
                # We've reached the budget limit - since there might be
                # continuous prefills in the running queue, we should break
                # to avoid scheduling any new prefills.
                break

            num_new_seqs = seq_group.llm_seq_group.get_max_num_running_seqs()
            if llm_num_new_tokens_uncached == 0 or not budget.can_schedule(
                num_new_tokens=llm_num_new_tokens_uncached,
                num_new_seqs=num_new_seqs,
            ):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            if partial_prefill_metadata is not None:
                partial_prefill_metadata.maybe_increment_partial_prefills(seq_group)

            if enable_chunking and self.scheduler_config.is_multi_step:
                blocks_to_copy: List[Tuple[int, int]] = []
                # init_multi_step_from_lookahead_slots happens in append_slots
                self._append_slots(seq_group, blocks_to_copy, enable_chunking)
                # This assert will trip when a copy-on-write happens. This is
                # not a concern as the very first sequence-group block
                # allocation happens above. Still, we have the assert to
                # catch any edge-cases.
                assert not blocks_to_copy
            else:
                seq_group.llm_seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking,
                )
                seq_group.tts_seq_group.init_multi_step_from_lookahead_slots(
                    num_lookahead_slots,
                    num_scheduler_steps=self.scheduler_config.
                    num_scheduler_steps,
                    is_multi_step=self.scheduler_config.is_multi_step,
                    enable_chunking=enable_chunking,
                )

            seq_groups.append(
                ScheduledSequenceGroup(
                    seq_group=seq_group,
                    token_chunk_size=llm_num_new_tokens
                )
            )
            budget.add_num_batched_tokens(
                seq_group.llm_seq_group.request_id,
                num_batched_tokens=llm_num_new_tokens_uncached,
                num_cached_tokens=llm_num_new_tokens_cached,
            )
            budget.add_num_seqs(seq_group.llm_seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True,
                enable_chunking=enable_chunking
            ),
        )

    def _schedule_default(self) -> SchedulerOutputs:
        """Schedule queued requests.

        The current policy is designed to optimize the throughput. First,
        it batches as many prefill requests as possible. And it schedules
        decodes. If there's a pressure on GPU memory, decode requests can
        be swapped or preempted.
        """
        # Include running requests to the budget.
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        # Make sure we include num running seqs before scheduling prefill,
        # so that we don't schedule beyond max_num_seqs for prefill.
        for seq_group in self.running:
            seq_group: AvatarSequenceGroup
            budget.add_num_seqs(
                seq_group.llm_seq_group.request_id,
                max(
                    seq_group.llm_seq_group.get_max_num_running_seqs(),
                    seq_group.tts_seq_group.get_max_num_running_seqs()
                )
            )

        curr_loras = (
            set(seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None
        )

        prefills = SchedulerPrefillOutputs.create_empty()
        running_scheduled = SchedulerRunningOutputs.create_empty()
        swapped_in = SchedulerSwappedInOutputs.create_empty()

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            prefills = self._schedule_prefills(
                budget,
                curr_loras,
                enable_chunking=False
            )

        if len(prefills.seq_groups
               ) == 0 and self.scheduler_config.policy == "priority":
            self._schedule_priority_preemption(budget)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(
                budget,
                curr_loras,
                enable_chunking=False
            )

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if (len(running_scheduled.preempted) + len(running_scheduled.swapped_out) == 0):
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups]
            )

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = len(running_scheduled.preempted) + len(running_scheduled.swapped_out)

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens +
            budget.num_cached_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )

    def _schedule_running(
        self,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
        partial_prefill_metadata: Optional[PartialPrefillMetadata] = None,
    ) -> SchedulerRunningOutputs:
        """Schedule sequence groups that are running.

        Running queue should include decode and chunked prefill requests.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any decodes are preempted.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any decodes are preempted.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.
            partial_prefill_metadata: information about the partial prefills
            that are currently running

        Returns:
            SchedulerRunningOutputs.
        """
        ret: SchedulerRunningOutputs = self._scheduler_running_outputs_cache[self.cache_id].get_object()
        ret.blocks_to_swap_out.clear()
        ret.blocks_to_copy.clear()
        ret.decode_seq_groups.clear()
        ret.prefill_seq_groups.clear()
        ret.preempted.clear()
        ret.swapped_out.clear()

        ret.num_lookahead_slots = self._get_num_lookahead_slots(
            is_prefill=False, enable_chunking=enable_chunking)

        ret.decode_seq_groups_list.clear()
        ret.prefill_seq_groups_list.clear()

        # Blocks that need to be swapped or copied before model execution.
        blocks_to_swap_out: List[Tuple[int, int]] = ret.blocks_to_swap_out
        blocks_to_copy: List[Tuple[int, int]] = ret.blocks_to_copy

        decode_seq_groups: List[ScheduledSequenceGroup] = ret.decode_seq_groups
        prefill_seq_groups: List[ScheduledSequenceGroup] = ret.prefill_seq_groups
        preempted: List[SequenceGroup] = ret.preempted
        swapped_out: List[SequenceGroup] = ret.swapped_out

        running_queue = self.running
        assert len(self._async_stopped) == 0
        while running_queue:
            seq_group: AvatarSequenceGroup = running_queue[0]
            # We discard the cached tokens info here because we don't need it
            # for running sequence:
            #   1. If a sequence is running with chunked prefill, the cached
            #      tokens info was already used for the first prefill.
            #   2. If a sequence is running with non-chunked prefill, then
            #      there it's a decoding sequence, and the cached tokens info is
            #      irrelevant.
            
            llm_num_uncached_new_tokens, _ = self._get_num_new_uncached_and_cached_tokens(
                seq_group.llm_seq_group,
                SequenceStatus.RUNNING,
                enable_chunking,
                budget,
                partial_prefill_metadata,
            )

            num_running_tokens = llm_num_uncached_new_tokens
            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # With async postprocessor, an extra decode run is done
            # to process the final tokens. The check below avoids this extra
            # decode run when the model max len is reached, in order to avoid
            # a memory overflow.
            if (self.use_async_output_proc and seq_group.llm_seq_group.seqs[0].get_len()
                    > self.scheduler_config.max_model_len):
                self._async_stopped.append(seq_group)
                continue

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group.llm_seq_group, enable_chunking):
                budget.subtract_num_batched_tokens(
                    seq_group.llm_seq_group.request_id,
                    num_running_tokens
                )
                num_running_seqs = seq_group.llm_seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(
                    seq_group.llm_seq_group.request_id,
                    num_running_seqs
                )

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True
                if self.use_async_output_proc:
                    assert self.output_proc_callback is not None
                    self.output_proc_callback(
                        request_id=victim_seq_group.request_id)

                    # It may be that the async pending "victim_seq_group"
                    # becomes finished, in which case we simply free it.
                    if victim_seq_group.is_finished():
                        self._free_finished_seq_group(victim_seq_group)
                        do_preempt = False

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group.llm_seq_group, blocks_to_copy, enable_chunking)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = (
                    self._scheduled_seq_group_cache[
                        self.cache_id
                    ].get_object()
                )
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(
                    seq_group.llm_seq_group.request_id,
                    num_running_tokens
                )
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.llm_seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.llm_seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        self._scheduler_running_outputs_cache[self.next_cache_id].reset()
        self._scheduled_seq_group_cache[self.next_cache_id].reset()
        self._tts_scheduled_seq_group_cache[self.next_cache_id].reset()

        return ret

    def _allocate_and_set_running(self, seq_group: AvatarSequenceGroup) -> None:
        self.block_manager.allocate(seq_group.llm_seq_group)
        self.tts_block_manager.allocate(seq_group.tts_seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.llm_seq.status = SequenceStatus.RUNNING
            seq.tts_seq.status = SequenceStatus.RUNNING

    def _allocate_tts_cross_table(self, seq_group: SequenceGroup):
        if seq_group.is_encoder_decoder():
            # Encoder associated with SequenceGroup
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            encoder_seq_data = encoder_seq.data
            # Block table for cross-attention
            # Also managed at SequenceGroup level
            cross_block_table = self.tts_block_manager.get_cross_block_table(seq_group)
        else:
            encoder_seq_data = None
            cross_block_table = None

        return encoder_seq_data, cross_block_table

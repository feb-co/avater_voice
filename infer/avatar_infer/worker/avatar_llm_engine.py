from typing import (TYPE_CHECKING, Callable, ClassVar, Deque, Dict, Iterable,
                    List, Mapping, NamedTuple, Optional)

from vllm.engine.llm_engine import LLMEngine
from vllm.core.scheduler import ScheduledSequenceGroup, SchedulerOutputs
from vllm.sequence import SequenceGroupMetadata, CompletionSequenceGroupOutput, SequenceOutput

from avatar_infer.generation.sequence import AvatarSamplerOutput


class AvatarLLMEngine(LLMEngine):
    def _advance_to_next_step(
        self,
        output: AvatarSamplerOutput,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        scheduled_seq_groups: List[ScheduledSequenceGroup]
    ) -> None:
        """Given model output from a single run, append the tokens to the
        sequences. This is normally done inside output processor, but it is
        required if the worker is to perform async forward pass to next step.
        """
        for seq_group_metadata, sequence_group_outputs, scheduled_seq_group in zip(seq_group_metadata_list, output, scheduled_seq_groups):
            seq_group = scheduled_seq_group.seq_group

            if seq_group.is_finished():
                continue

            # if self.scheduler_config.is_multi_step:
            #     # Updates happen only if the sequence is prefill
            #     self._update_num_computed_tokens_for_multi_step_prefill(
            #         seq_group, seq_group_metadata,
            #         seq_group.state.num_steps == 1
            #     )
            # else:
            #     token_chunk_size = (
            #         seq_group_metadata.token_chunk_size
            #         if seq_group_metadata.token_chunk_size
            #         is not None else 0
            #     )
            #     seq_group.update_num_computed_tokens(token_chunk_size)

            if seq_group_metadata.do_sample:
                assert len(seq_group.seqs) == 1
                assert len(sequence_group_outputs.samples) == 1, ("Async output processor expects a single sample (i.e sampling_params.n == 1)")
                llm_sample: SequenceOutput = sequence_group_outputs.llm_samples[0]
                tts_sample: SequenceOutput = sequence_group_outputs.tts_samples[0]

                tts_seq = seq_group.seqs[0]
                llm_seq = seq_group.encoder_seq

                tts_seq.data.update_num_computed_tokens(seq_group_metadata.token_chunk_size)
                llm_seq.data.update_num_computed_tokens(llm_seq.data.get_len())

                llm_seq.append_token_id(llm_sample.output_token, llm_sample.logprobs)
                for tts_token in tts_sample.output_token:
                    tts_probs = {tts_token: llm_sample.logprobs[llm_sample.output_token]}
                    tts_seq.append_token_id(tts_token, tts_probs)

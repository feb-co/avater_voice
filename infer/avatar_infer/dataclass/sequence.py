import msgspec
from functools import reduce
from typing import Any, Callable, Iterator, Dict, List, Mapping, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.inputs import SingletonInputs, SingletonInputsAdapter
from vllm.sequence import (
    SequenceData,
    Sequence,
    SequenceGroup,
    SequenceStage,
    SequenceStatus,
    SequenceDataDelta,
    SequenceGroupMetadata,
    CompletionSequenceGroupOutput,
    SequenceOutput,
    PromptLogprobs,
    SampleLogprobs,
)


class TTSSequenceData(SequenceData):
    """Data associated with a sequence.

    Args:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output. Set to an empty list if
            None.

    Attributes:
        prompt_token_ids: The token IDs of the prompt.
        output_token_ids: The token IDs of the output.
        cumulative_logprob: The cumulative log probability of the output.
    """
    # NOTE: we cannot use Union[List, array] because msgspec cannot support
    # union of 2 list types.
    _prompt_token_ids: List
    _output_token_ids: List[int] = msgspec.field(default_factory=list)

    ### The below fields should not be passed as an argument ###
    _cumulative_logprob: float = 0.0
    _prompt_token_ids_tuple: Tuple[int, ...] = msgspec.field(default_factory=tuple)
    # The number of tokens that are computed (that run against the model).
    _num_computed_tokens: int = 0
    # The number of tokens with prefix cache hit.
    _num_cached_tokens: int = 0
    _stage: SequenceStage = SequenceStage.PREFILL
    _cached_all_token_ids: List[int] = msgspec.field(default_factory=list)

    # It is used to get delta input. It is reset when `get_delta_and_reset`
    # is called.
    _new_appended_tokens: List[int] = msgspec.field(default_factory=list)

    # It is used to compute mrope_position_ids.
    _mrope_position_delta: Optional[int] = None

    @staticmethod
    def from_prompt_token_counts(*token_counts: Tuple[int, int]) -> "TTSSequenceData":
        """
        Construct a :class:`TTSSequenceData` instance by concatenating
        prompt token sequences.

        Each tuple represents one token sequence, expressed in the form
        :code:`(token_id, count)`.
        """
        if len(token_counts) == 0:
            return TTSSequenceData.from_seqs([])

        prompt_token_ids_arr = []
        for token_id, count in token_counts:
            prompt_token_ids_arr.extend([token_id] * count)

        return TTSSequenceData(prompt_token_ids_arr)

    @staticmethod
    def from_seqs(
        prompt_token_ids: GenericSequence[int],
        output_token_ids: Optional[GenericSequence[int]] = None,
    ) -> "TTSSequenceData":
        """
        Construct a :class:`TTSSequenceData` instance from prompt and output
        token sequences.
        """
        if output_token_ids is None:
            return TTSSequenceData(prompt_token_ids)

        return TTSSequenceData(
            prompt_token_ids,
            _output_token_ids=output_token_ids
        )

    def __post_init__(self) -> None:
        self._prompt_token_ids_tuple: Tuple[int, ...] = tuple(self._prompt_token_ids)
        self._update_cached_all_tokens()

    def _update_cached_all_tokens(self):
        self._cached_all_token_ids: List[int] = list(self._prompt_token_ids + self._output_token_ids)
    
    @property
    def prompt_token_ids(self) -> Tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        self._prompt_token_ids = new_prompt_token_ids
        self.__post_init__()

    @property
    def output_token_ids(self) -> Tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self, new_output_token_ids: GenericSequence[int]) -> None:
        self._output_token_ids = new_output_token_ids
        self._update_cached_all_tokens()


class TTSSequence(Sequence):
    def __init__(
        self,
        seq_id: int,
        inputs: SingletonInputs,
        block_size: int,
        eos_token_id: Optional[int] = None,
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> None:
        self.seq_id = seq_id
        self.inputs = SingletonInputsAdapter(inputs)
        self.block_size = block_size
        self.eos_token_id = eos_token_id
        self.lora_request = lora_request
        self.prompt_adapter_request = prompt_adapter_request

        self.data = TTSSequenceData.from_seqs(self.prompt_token_ids)
        self.output_logprobs: SampleLogprobs = []
        self.output_text = ""

        self.status = SequenceStatus.WAITING
        self.stop_reason: Union[int, str, None] = None

        # These are used to keep track of delta outputs
        self._last_output_token_ids_offset: int = 0
        self._last_output_text_offset: int = 0

        # Used for incremental detokenization
        self.prefix_offset = 0
        self.read_offset = 0
        # Input + output tokens
        self.tokens: Optional[List[str]] = None


class AvatarSequence:
    def __init__(
        self,
        llm_seq: Sequence,
        tts_seq: Sequence,
    ) -> None:
        self.llm_seq = llm_seq
        self.tts_seq = tts_seq

    def is_prefill(self) -> bool:
        return (self.tts_seq.data.stage == SequenceStage.PREFILL and self.llm_seq.data.stage == SequenceStage.PREFILL)


class AvatarSequenceGroup:
    def __init__(
        self,
        llm_seq_group: Optional[SequenceGroup] = None,
        tts_seq_group: Optional[SequenceGroup] = None,
    ) -> None:
        self.llm_seq_group = llm_seq_group
        self.tts_seq_group = tts_seq_group

    @property
    def lora_request(self):
        return self.llm_seq_group.lora_request
    
    @property
    def prompt_adapter_request(self):
        return self.llm_seq_group.prompt_adapter_request
    
    @property
    def pooled_data(self):
        return None

    def get_seqs(
        self,
        status: Optional[SequenceStatus] = None,
    ) -> List[AvatarSequence]:
        seqs = [AvatarSequence(llm_seq, tts_seq) for (llm_seq, tts_seq) in zip(self.llm_seq_group.seqs, self.tts_seq_group.seqs)]

        if status is None:
            return seqs

        if self.tts_seq_group.is_single_seq and self.llm_seq_group.is_single_seq:
            return seqs if self.tts_seq_group.first_seq.status == status and self.llm_seq_group.first_seq.status == status else []

        return [AvatarSequence(llm_seq, tts_seq) for (llm_seq, tts_seq) in zip(self.llm_seq_group.seqs, self.tts_seq_group.seqs) if (tts_seq.status == status and llm_seq.status == status)]

    def is_prefill(self) -> bool:
        return self.llm_seq_group.is_prefill()
    
    def is_finished(self) -> bool:
        return self.llm_seq_group.is_finished() and self.tts_seq_group.is_finished()
    
    def get_last_token_latency(self) -> float:
        """Returns the latency of the last token."""
        assert not self.is_prefill(), (
            "seq_group.get_last_token_latency() should not be called "
            "if the seq_group is in prefill phase."
        )
        return max(
            self.llm_seq_group.last_token_latency,
            self.tts_seq_group.last_token_latency
        )
    
    def num_seqs(self, status: Optional[SequenceStatus] = None) -> int:
        # Optimization. We don't need to call get_seqs if we don't need to
        # filter by states.
        seqs = [AvatarSequence(llm_seq, tts_seq) for (llm_seq, tts_seq) in zip(self.llm_seq_group.seqs, self.tts_seq_group.seqs)]

        if status is None:
            return len(seqs)

        if self.tts_seq_group.is_single_seq and self.llm_seq_group.is_single_seq:
            return 1 if (seqs[0].llm_seq.status == status and seqs[0].tts_seq.status == status) else 0

        return len(self.get_seqs(status))


class AvatarSequenceGroupMetadata(
        msgspec.Struct,
        tag=True,  # type: ignore[call-arg]
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True):  # type: ignore[call-arg]
    llm_seq_group_metadata: SequenceGroupMetadata
    tts_seq_group_metadata: SequenceGroupMetadata


class AvatarCompletionSequenceGroupOutput(CompletionSequenceGroupOutput):
    """
    A wrapper that extends CompletionSequenceGroupOutput with additional TTS data.
    This wrapper must be created using the from_outputs classmethod, not directly constructed.
    """
    tts_samples: List[SequenceOutput] = msgspec.field(default_factory=list)
    tts_prompt_logprobs: Optional[PromptLogprobs] = None

    @classmethod
    def from_outputs(
        cls,
        llm_output: CompletionSequenceGroupOutput,
        tts_output: CompletionSequenceGroupOutput
    ) -> "CompletionSequenceGroupOutput":
        """
        Factory method to create an AvatarOutputWrapper from LLM and TTS outputs.
        This properly initializes the msgspec.Struct fields.
        """
        # Create with LLM values
        wrapper = cls(
            samples=llm_output.samples,
            prompt_logprobs=llm_output.prompt_logprobs
        )

        # Add TTS values
        wrapper.tts_samples = tts_output.samples
        wrapper.tts_prompt_logprobs = tts_output.prompt_logprobs
        
        return wrapper
    
    @property
    def llm_samples(self):
        """Access LLM samples directly (same as .samples)"""
        return self.samples
    
    @property 
    def llm_prompt_logprobs(self):
        """Access LLM prompt logprobs directly (same as .prompt_logprobs)"""
        return self.prompt_logprobs

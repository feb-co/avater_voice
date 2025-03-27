from array import array
from functools import reduce
from typing import Any, Callable, Iterator, Dict, List, Mapping, Optional
from typing import Sequence as GenericSequence
from typing import Set, Tuple, Union

import msgspec

from vllm.sequence import (
    SequenceStage,
    SequenceDataDelta,
    CompletionSequenceGroupOutput,
    PromptLogprobs,
    SequenceOutput
)


TOKEN_ID_ARRAY_TYPE = "l"

VLLM_INVALID_TOKEN_ID = -1


def array_full_2d(token_id: int, count: int, code_layers: int):
    """:class:`array` equivalent of :func:`numpy.full`."""
    # 创建一个[count, code_layers]维度的array,每个元素都是token_id
    result = []
    for _ in range(count):
        result.extend([token_id] * code_layers)
    return array(TOKEN_ID_ARRAY_TYPE, result)


class AvatarSequenceData(msgspec.Struct, omit_defaults=True):
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
    _prompt_token_ids: array
    _output_token_ids: array = msgspec.field(default_factory=lambda: array(TOKEN_ID_ARRAY_TYPE, []))

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
    def from_prompt_token_counts(code_layers: int, *token_counts: Tuple[int, int]) -> "AvatarSequenceData":
        """
        Construct a :class:`AvatarSequenceData` instance by concatenating
        prompt token sequences.

        Each tuple represents one token sequence, expressed in the form
        :code:`(token_id, count)`.
        """
        if len(token_counts) == 0:
            return AvatarSequenceData.from_seqs([])

        prompt_token_ids_arr = reduce(
            array.__iadd__,
            (array_full_2d(token_id, count, code_layers) for token_id, count in token_counts),
        )

        return AvatarSequenceData(prompt_token_ids_arr)

    @staticmethod
    def from_seqs(
        prompt_token_ids: GenericSequence[int],
        output_token_ids: Optional[GenericSequence[int]] = None,
    ) -> "AvatarSequenceData":
        """
        Construct a :class:`AvatarSequenceData` instance from prompt and output
        token sequences.
        """
        prompt_token_ids_arr = array(TOKEN_ID_ARRAY_TYPE,
                                     prompt_token_ids)

        if output_token_ids is None:
            return AvatarSequenceData(prompt_token_ids_arr)

        output_token_ids_arr = array(TOKEN_ID_ARRAY_TYPE,
                                     output_token_ids)

        return AvatarSequenceData(prompt_token_ids_arr,
                            _output_token_ids=output_token_ids_arr)

    def __post_init__(self) -> None:
        assert self._prompt_token_ids.typecode == "l"
        assert self._output_token_ids.typecode == "l"
        self._prompt_token_ids_tuple: Tuple[int, ...] = tuple(
            self._prompt_token_ids)
        self._update_cached_all_tokens()

    def _update_cached_all_tokens(self):
        assert isinstance(self._prompt_token_ids, array)
        assert isinstance(self._output_token_ids, array)
        self._cached_all_token_ids: List[int] = list(self._prompt_token_ids +
                                                     self._output_token_ids)

    @property
    def cumulative_logprob(self) -> float:
        return self._cumulative_logprob

    @property
    def prompt_token_ids(self) -> Tuple[int, ...]:
        return self._prompt_token_ids_tuple

    @prompt_token_ids.setter
    def prompt_token_ids(self, new_prompt_token_ids) -> None:
        raise NotImplementedError

    @property
    def prompt_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        return self._prompt_token_ids

    @property
    def output_token_ids(self) -> Tuple[int, ...]:
        return tuple(self._output_token_ids)

    @output_token_ids.setter
    def output_token_ids(self,
                         new_output_token_ids: GenericSequence[int]) -> None:
        self._output_token_ids = array(TOKEN_ID_ARRAY_TYPE,
                                       new_output_token_ids)
        self._update_cached_all_tokens()

    @property
    def output_token_ids_array(self) -> array:
        """Return the prompt token ids in array type.

        Note that the array is in "I" type, and it is not compatible
        with torch.long (2 bytes vs 4 bytes). So beware of the usage.
        """
        assert isinstance(self._output_token_ids, array)
        return self._output_token_ids

    @property
    def mrope_position_delta(self) -> Optional[int]:
        return self._mrope_position_delta

    @mrope_position_delta.setter
    def mrope_position_delta(self, new_mrope_position_delta):
        self._mrope_position_delta = new_mrope_position_delta

    def append_token_id(self, token_id: int, logprob: float) -> None:
        self._output_token_ids.append(token_id)
        self._new_appended_tokens.append(token_id)
        self._cached_all_token_ids.append(token_id)
        self._cumulative_logprob += logprob

    def get_len(self) -> int:
        return len(self._output_token_ids) + len(self._prompt_token_ids)

    def get_prompt_len(self) -> int:
        return len(self._prompt_token_ids)

    def get_output_len(self) -> int:
        return len(self._output_token_ids)

    def get_token_ids(self) -> List[int]:
        return self._cached_all_token_ids

    def get_prefix_token_ids(
            self, num_tokens: int
    ) -> Tuple[Tuple[int, ...], Optional[Tuple[int, ...]]]:
        """Get prefix tokens, and make the return value hashable"""
        prompt_length = self.get_prompt_len()
        if num_tokens > prompt_length:
            return (self._prompt_token_ids_tuple,
                    tuple(self._output_token_ids[:num_tokens - prompt_length]))
        else:
            return (self._prompt_token_ids_tuple[:num_tokens], None)

    def get_num_computed_tokens(self) -> int:
        """Return the number of prefill tokens that are already computed."""
        return self._num_computed_tokens

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        """Update number of tokens computed so far."""
        self._num_computed_tokens += num_new_computed_tokens
        assert self._num_computed_tokens <= self.get_len(), (
            self._num_computed_tokens, self.get_len())
        # If all tokens are computed, it means it is in decoding phase.
        if self.get_num_uncomputed_tokens() == 0:
            self._stage = SequenceStage.DECODE

    def get_num_cached_tokens(self) -> int:
        """Return the number of tokens with prefix cache hit."""
        return self._num_cached_tokens

    def update_num_cached_tokens(self, num_cached_tokens: int):
        """Update the number of tokens with prefix cache hit."""
        self._num_cached_tokens = num_cached_tokens

    def reset_state_for_recompute(self) -> None:
        """Reset the number of computed tokens from this sequence. It is
        supposed to be called when a sequence needs to be started from
        the beginning again (e.g., sequence is preempted).
        """
        self._num_computed_tokens = 0
        self._stage = SequenceStage.PREFILL
        self._new_appended_tokens = []

    def get_num_uncomputed_tokens(self) -> int:
        """Return the number of prefill tokens that are not computed."""
        # we use `get_len()` which includes prompt_len + output_len instead
        # of prompt_len here. This is because during recompute we need to
        # prefill for both prompt and output.
        return self.get_len() - self.get_num_computed_tokens()

    def get_last_token_id(self) -> int:
        if not self._output_token_ids:
            return self._prompt_token_ids[-1]
        return self._output_token_ids[-1]

    def get_prompt_token_ids(self) -> Tuple[int, ...]:
        return self.prompt_token_ids

    def get_output_token_ids(self) -> Tuple[int, ...]:
        return self.output_token_ids

    def get_delta_and_reset(self) -> SequenceDataDelta:
        delta = SequenceDataDelta(self._new_appended_tokens,
                                  self._cumulative_logprob,
                                  self.get_num_computed_tokens(), self.stage)
        # Reset delta state.
        self._new_appended_tokens = []
        return delta

    def apply_delta(self, delta: SequenceDataDelta):
        self._num_computed_tokens = delta.new_num_computed_tokens
        self._cumulative_logprob = delta.new_cumulative_logprob
        self._stage = delta.new_stage
        self._output_token_ids.extend(delta.new_output_token_ids)
        self._cached_all_token_ids.extend(delta.new_output_token_ids)

    @property
    def stage(self) -> SequenceStage:
        return self._stage

    def __repr__(self) -> str:
        return (f"SequenceData("
                f"prompt_token_ids={self._prompt_token_ids}, "
                f"output_token_ids={self.output_token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"get_num_computed_tokens={self.get_num_computed_tokens()})")


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


class AvatarSamplerOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    llm_outputs: List[CompletionSequenceGroupOutput]
    tts_outputs: List[CompletionSequenceGroupOutput]

    # Preserve original SamplerOutput attributes
    sampled_token_probs = None
    sampled_token_ids = None
    spec_decode_worker_metrics = None

    def __getitem__(self, idx: int) -> CompletionSequenceGroupOutput:
        """Returns a CompletionSequenceGroupOutput-compatible wrapper at the specified index"""
        if idx >= len(self.llm_outputs) or idx >= len(self.tts_outputs):
            raise IndexError(f"Index {idx} out of range")

        return AvatarCompletionSequenceGroupOutput.from_outputs(
            llm_output=self.llm_outputs[idx],
            tts_output=self.tts_outputs[idx]
        )

    def __setitem__(
        self,
        idx: int,
        value: Union[CompletionSequenceGroupOutput, Tuple[CompletionSequenceGroupOutput, CompletionSequenceGroupOutput]]
    ):
        """Sets outputs at the specified index"""
        if isinstance(value, AvatarCompletionSequenceGroupOutput):
            # Handle our wrapper class
            self.llm_outputs[idx] = CompletionSequenceGroupOutput(
                samples=value.samples,
                prompt_logprobs=value.prompt_logprobs
            )
            self.tts_outputs[idx] = CompletionSequenceGroupOutput(
                samples=value.tts_samples,
                prompt_logprobs=value.tts_prompt_logprobs
            )
        elif isinstance(value, tuple) and len(value) == 2:
            # Handle tuple of (llm, tts)
            self.llm_outputs[idx] = value[0]
            self.tts_outputs[idx] = value[1]
        elif isinstance(value, CompletionSequenceGroupOutput):
            # Handle standard CompletionSequenceGroupOutput
            self.llm_outputs[idx] = value
            # Create a placeholder TTS output if needed
            if idx >= len(self.tts_outputs):
                self.tts_outputs.append(CompletionSequenceGroupOutput(
                    samples=[],
                    prompt_logprobs=None
                ))
        else:
            raise ValueError("Value must be a CompletionSequenceGroupOutput or a tuple of (llm_output, tts_output)")

    def __iter__(self) -> Iterator[CompletionSequenceGroupOutput]:
        """Returns an iterator that yields wrapped CompletionSequenceGroupOutput objects"""
        for i in range(len(self)):
            yield self.__getitem__(i)

    def __len__(self):
        """Returns the number of output pairs, which is the minimum length of both output lists"""
        return min(len(self.llm_outputs), len(self.tts_outputs))

    def __eq__(self, other: object):
        """Compares if two AvatarSamplerOutput objects are equal"""
        return (isinstance(other, self.__class__) and 
                self.llm_outputs == other.llm_outputs and 
                self.tts_outputs == other.tts_outputs)

    def __repr__(self) -> str:
        """Shows shapes of tensors instead of their values to reduce output noise"""
        sampled_token_probs_repr = ("None" if self.sampled_token_probs is None
                                   else getattr(self.sampled_token_probs, 'shape', str(self.sampled_token_probs)))
        sampled_token_ids_repr = ("None" if self.sampled_token_ids is None 
                                 else getattr(self.sampled_token_ids, 'shape', str(self.sampled_token_ids)))

        return (
            f"AvatarSamplerOutput("
            f"llm_outputs={self.llm_outputs}, "
            f"tts_outputs={self.tts_outputs}, "
            f"sampled_token_probs={sampled_token_probs_repr}, "
            f"sampled_token_ids={sampled_token_ids_repr}, "
            f"spec_decode_worker_metrics={self.spec_decode_worker_metrics})"
        )

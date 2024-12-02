import torch
from torch import nn
from abc import abstractmethod

from mini.module.streaming import StreamingModule, State
from mini.quantization.base import QuantizedResult


class CompressionModel(StreamingModule[State]):
    """Base API for all compression model that aim at being used as audio tokenizers
    with a language model.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> QuantizedResult: ...

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.encode`."""
        ...

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """See `MimiModel.decode`."""
        ...

    @abstractmethod
    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode from the discrete codes to continuous latent space."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int: ...

    @property
    @abstractmethod
    def frame_rate(self) -> float: ...

    @property
    @abstractmethod
    def sample_rate(self) -> int: ...

    @property
    @abstractmethod
    def cardinality(self) -> int: ...

    @property
    @abstractmethod
    def num_codebooks(self) -> int: ...

    @property
    @abstractmethod
    def total_codebooks(self) -> int: ...

    @abstractmethod
    def set_num_codebooks(self, n: int):
        """Set the active number of codebooks used by the quantizer."""
        ...

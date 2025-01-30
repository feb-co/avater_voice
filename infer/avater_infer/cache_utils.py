import torch
from typing import Any, Dict, List, Optional, Tuple

from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.deprecation import deprecate_kwarg


class AvaterCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for Avater generative models.

    It stores the three different cache:
    1. Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is `[batch_size, num_heads, seq_len, head_dim]`.

    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

        >>> inputs = tokenizer(text="My name is Qwen2", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> past_key_values = DynamicCache()
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        DynamicCache()
        ```
    """

    def __init__(self, llm_attention_cache: Cache, self_attention_cache: Cache, cross_attention_cache: Cache) -> None:
        super().__init__()
        self.llm_attention_cache = llm_attention_cache
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache

        self.is_updated = {}
        for layer_idx in range(len(self.cross_attention_cache.key_cache)):
            self.is_updated[layer_idx] = bool(self.cross_attention_cache.get_seq_length(layer_idx) > 0)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.llm_attention_cache.key_cache[layer_idx],
                self.llm_attention_cache.value_cache[layer_idx],
                self.self_attention_cache.value_cache[layer_idx] if layer_idx < len(self.self_attention_cache) else None,
                self.self_attention_cache.value_cache[layer_idx] if layer_idx < len(self.self_attention_cache) else None,
                self.cross_attention_cache.value_cache[layer_idx] if layer_idx < len(self.self_attention_cache) else None,
                self.cross_attention_cache.value_cache[layer_idx] if layer_idx < len(self.self_attention_cache) else None,
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return max(len(self.llm_attention_cache), len(self.self_attention_cache))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # check if empty list because in case of static cache it will be a tensors and we can't check `if not torch.Tensor`
        return self.self_attention_cache.get_seq_length(layer_idx)

    def reset(self):
        if hasattr(self.self_attention_cache, "reset"):
            self.self_attention_cache.reset()
        if hasattr(self.cross_attention_cache, "reset"):
            self.cross_attention_cache.reset()
        elif not hasattr(self.cross_attention_cache, "reset") and not hasattr(self.self_attention_cache, "reset"):
            raise ValueError(
                "Neither self nor cross-attention cache have valid `.reset()` methods. `.reset()` should "
                "only be called on compatible cache classes, such as `StaticCache` or `SlidingWindowCache`. "
                f"Got {self.self_attention_cache.__str__()} for the self attention cache and "
                f"{self.cross_attention_cache.__str__()} for the cross attention cache."
            )
        for layer_idx in self.is_updated:
            self.is_updated[layer_idx] = False

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        self.self_attention_cache.reorder_cache(beam_idx)
        self.cross_attention_cache.reorder_cache(beam_idx)

    def check_dynamic_cache(self, method: str):
        if not (
            isinstance(self.self_attention_cache, DynamicCache)
            and isinstance(self.cross_attention_cache, DynamicCache)
            and isinstance(self.llm_attention_cache, DynamicCache)
        ):
            raise ValueError(
                f"`{method}` is only defined for dynamic cache, got {self.self_attention_cache.__str__()} for the self "
                f"attention cache and {self.cross_attention_cache.__str__()} for the cross attention cache."
            )

    # TODO(gante, sanchit-gandhi): move following functionality into `.generate`
    def crop(self, maximum_length: int):
        """Crop the past key values up to a new `maximum_length` in terms of tokens. `maximum_length` can also be
        negative to remove `maximum_length` tokens. This is used in assisted decoding and contrastive search."""
        self.check_dynamic_cache(self.crop.__name__)
        self.self_attention_cache.crop(maximum_length)

    def batch_split(self, full_batch_size: int, split_size: int) -> "List[AvaterCache]":
        """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
        `_split_model_inputs()` in `generation.utils`"""
        self.check_dynamic_cache(self.batch_split.__name__)
        llm_attention_cache = self.llm_attention_cache.batch_split(full_batch_size, split_size)
        self_attention_cache = self.self_attention_cache.batch_split(full_batch_size, split_size)
        cross_attention_cache = self.cross_attention_cache.batch_split(full_batch_size, split_size)

        out = []
        for llm_attn, self_attn, cross_attn in zip(llm_attention_cache, self_attention_cache, cross_attention_cache):
            out.append(AvaterCache(llm_attn, self_attn, cross_attn))
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["AvaterCache"]) -> "AvaterCache":
        """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
        `generation.utils`"""
        llm_attention_cache = DynamicCache()
        self_attention_cache = DynamicCache()
        cross_attention_cache = DynamicCache()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.llm_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.llm_attention_cache.value_cache[idx] for current in splits], dim=0)
            llm_attention_cache.update(layer_keys, layer_values, idx)

            layer_keys = torch.cat([current.self_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.self_attention_cache.value_cache[idx] for current in splits], dim=0)
            self_attention_cache.update(layer_keys, layer_values, idx)

            layer_keys = torch.cat([current.cross_attention_cache.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.cross_attention_cache.value_cache[idx] for current in splits], dim=0)
            cross_attention_cache.update(layer_keys, layer_values, idx)
        return cls(llm_attention_cache, self_attention_cache, cross_attention_cache)

    def batch_repeat_interleave(self, repeats: int):
        """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_repeat_interleave.__name__)
        self.llm_attention_cache.batch_repeat_interleave(repeats)
        self.self_attention_cache.batch_repeat_interleave(repeats)
        self.cross_attention_cache.batch_repeat_interleave(repeats)

    def batch_select_indices(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        self.check_dynamic_cache(self.batch_select_indices.__name__)
        self.llm_attention_cache.batch_select_indices(indices)
        self.self_attention_cache.batch_select_indices(indices)
        self.cross_attention_cache.batch_select_indices(indices)

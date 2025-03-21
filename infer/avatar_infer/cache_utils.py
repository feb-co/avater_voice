import torch
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache


class AvatarTokenCache(Cache):
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.token_id_cache: torch.Tensor = None
        self.token_state_cache: List[torch.Tensor] = []
        self.update_event = threading.Event()
        self.last_update_time = time.time()

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.token_id_cache, self.token_state_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.token_id_cache, self.token_state_cache[layer_idx])

    def __len__(self):
        return self.token_id_cache.size(1) if self.token_id_cache is not None else 0

    def update(
        self,
        layer_idx: int,
        token_states: torch.Tensor = None,
        token_ids: Optional[torch.Tensor] = None,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if token_states is not None:
            # Update the number of seen tokens
            if layer_idx == 0:
                self._seen_tokens += token_states.shape[1]

            # Update the state cache
            if len(self.token_state_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.token_state_cache), layer_idx):
                    self.token_state_cache.append([])
                self.token_state_cache.append(token_states)
            elif len(self.token_state_cache[layer_idx]) == 0:  # fills previously skipped layers; checking for tensor causes errors
                self.token_state_cache[layer_idx] = token_states
            else:
                self.token_state_cache[layer_idx] = torch.cat([self.token_state_cache[layer_idx], token_states], dim=1)

        # Update the token cache
        if token_ids is not None:
            if isinstance(token_ids, int):
                assert self.token_id_cache is not None, "token_id_cache error !"
                temp_token_ids = torch.zeros_like(self.token_id_cache)[:, :1]
                temp_token_ids[:] = token_ids
                token_ids = temp_token_ids

            if self.token_id_cache is not None:
                self.token_id_cache = torch.cat([self.token_id_cache, token_ids], dim=1)
            else:
                self.token_id_cache = token_ids

            self.last_update_time = time.time()
            self.update_event.set()
            self.update_event.clear()
        return self.token_id_cache, self.token_state_cache[layer_idx]

    def endswith(self, token_id: int) -> bool:
        if self.token_id_cache is not None:
            return (self.token_id_cache[:, -1] == token_id).all()
        else:
            return False

    def reset(self):
        self._seen_tokens = 0
        self.token_id_cache: torch.Tensor = None
        self.token_state_cache: List[torch.Tensor] = []
        self.last_update_time = time.time()
        self.last_token_length = 0
        self.token_growth_rate = 0


class AvatarCache(Cache):
    def __init__(
        self,
        llm_attention_cache: DynamicCache,
        self_attention_cache: DynamicCache,
        cross_attention_cache: DynamicCache,
        avatar_token_cache: AvatarTokenCache
    ) -> None:
        super().__init__()
        self.llm_attention_cache = llm_attention_cache
        self.self_attention_cache = self_attention_cache
        self.cross_attention_cache = cross_attention_cache
        self.avatar_token_cache = avatar_token_cache

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

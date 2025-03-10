"""PyTorch Avater Voice Base."""

from torch import nn

from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel

from .configuration_voice import AvaterVoiceConfig


logger = logging.get_logger(__name__)


class AvaterVoicePreTrainedModel(PreTrainedModel):
    config_class = AvaterVoiceConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer", "TTSAdapterLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

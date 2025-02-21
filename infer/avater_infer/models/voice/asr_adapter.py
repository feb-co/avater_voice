"""PyTorch Avater ASR adapter."""

from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from transformers import AutoModel
from transformers.utils import logging
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, Wav2Vec2BaseModelOutput
from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer
from transformers.models.wavlm.modeling_wavlm import WavLMModel, WavLMEncoderLayerStableLayerNorm
from transformers.models.llama.modeling_llama import ACT2FN, LlamaRMSNorm

from .configuration_voice import AvaterVoiceConfig


logger = logging.get_logger(__name__)


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, check_fn):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    logger.info(f"Applying fsdp activation checkpointing...")
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


def infer_conv_output_dim(in_channels, input_dim, out_channels):
    sample_seq_len = 200
    sample_bsz = 10
    x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
    x = torch.nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
    x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
    x = x.transpose(1, 2)
    mb, seq = x.size()[:2]
    return x.contiguous().view(mb, seq, -1).size(-1)


class Conv1dSubsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def forward(self, x):
        bsz, in_seq_len, _ = x.size()  # B x T x (C x D)
        x = x.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x


class ASRAdapterMLP(nn.Module):
    def __init__(self, config: AvaterVoiceConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.asr_adapter_hidden_size
        self.intermediate_size = config.asr_adapter_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class ASRAdapter(nn.Module):
    def __init__(self, input_dim: int, config: AvaterVoiceConfig):
        super().__init__()
        self.subsample = Conv1dSubsampler(input_dim, input_dim, config.asr_adapter_hidden_size, [3, 3])
        self.adapter = ASRAdapterMLP(config)
        self.adapter_layer_norm = LlamaRMSNorm(config.asr_adapter_hidden_size, eps=config.rms_norm_eps)
        self.projector = nn.Linear(config.asr_adapter_hidden_size, config.asr_adapter_hidden_size)
    
    def forward(self, hidden_states: torch.Tensor):
        # down sampling
        hidden_states = self.subsample(hidden_states)

        # adapter
        residual = hidden_states
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.adapter_layer_norm(hidden_states)
        hidden_states = residual + hidden_states

        # projector
        hidden_states = self.projector(hidden_states)
        
        return hidden_states


class AvaterASRPreTrainedModel(PreTrainedModel):
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


class ASREncoder(AvaterASRPreTrainedModel):
    def __init__(self, config: AvaterVoiceConfig):
        super().__init__(config)
        self.config = config

        # Whiper
        self.whisper_encoder = AutoModel.from_pretrained(config.whisper_path).encoder
        self.whisper_adapter = ASRAdapter(config.whisper_hidden_size, config)

        # WavLM
        self.wavlm_encoder = WavLMModel.from_pretrained(config.wavlm_path)
        self.wavlm_adapter = ASRAdapter(config.wavlm_hidden_size, config)

        # concat projector
        self.concat_proj = nn.Linear(config.asr_adapter_hidden_size*2, config.hidden_size)

        # frozen param
        for param in self.whisper_encoder.parameters():
            param.requires_grad = False

        for param in self.wavlm_encoder.parameters():
            param.requires_grad = False

        # fsdp
        apply_fsdp_checkpointing(
            self.whisper_encoder,
            check_fn=lambda submodule: isinstance(submodule, WhisperEncoderLayer)
        )
        apply_fsdp_checkpointing(
            self.wavlm_encoder,
            check_fn=lambda submodule: isinstance(submodule, WavLMEncoderLayerStableLayerNorm)
        )

    def forward(
        self,
        audio_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        wavlm_features: Optional[torch.FloatTensor] = None,
        wavlm_attention_mask: Optional[torch.LongTensor] = None,
        audio_positions: Optional[torch.LongTensor] = None,
    ):
        # whisper
        whisper_outputs: BaseModelOutput = self.whisper_encoder(
            input_features=audio_features,
            attention_mask=audio_attention_mask
        )
        whisper_hidden_state = self.whisper_adapter(whisper_outputs.last_hidden_state)

        # wavlm
        wavlm_outputs: Wav2Vec2BaseModelOutput = self.wavlm_encoder(
            input_values=wavlm_features,
            attention_mask=wavlm_attention_mask
        )
        wavlm_hidden_state = self.whisper_adapter(whisper_outputs.last_hidden_state)

        # concat
        concat_features = torch.zeros([audio_positions.size(0), torch.max(audio_positions[:, 2]), self.config.asr_adapter_hidden_size*2]).to(wavlm_hidden_state)
        concat_features[:, :, :self.config.asr_adapter_hidden_size] = whisper_hidden_state[:, :concat_features.size(1), :]
        concat_features[:, :, self.config.asr_adapter_hidden_size:] = wavlm_hidden_state[:, :concat_features.size(1), :]
        audio_hidden_state = self.concat_proj(concat_features)
        
        return audio_hidden_state

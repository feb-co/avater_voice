"""PyTorch LLaMA ASR model."""

from typing import List, Optional, Tuple, Union

import torch

from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from avater_infer.models.voice import ASREncoder, AvaterASRPreTrainedModel
from avater_infer.cache_utils import AvaterCache

from .configuration_llama_asr import LlamaASRConfig


logger = logging.get_logger(__name__)


class LlamaASRForCausalLM(AvaterASRPreTrainedModel, GenerationMixin):
    config_class = LlamaASRConfig

    def __init__(self, config: LlamaASRConfig):
        super().__init__(config)

        self.config = config

        self.asr_encoder = ASREncoder(config)

        if config.llm_path:
            self.llm: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(config.llm_path)
        else:
            self.llm: LlamaForCausalLM = LlamaForCausalLM(config)

        # Initialize weights and apply final processing
        self.post_init()

        # frozen llm
        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

    def __repr__(self):
        return self.asr_encoder.__repr__()

    def get_encoder(self,):
        return self.llm

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.LongTensor] = None,
        wavlm_features: Optional[torch.FloatTensor] = None,
        wavlm_attention_mask: Optional[torch.LongTensor] = None,
        audio_positions: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[AvaterCache] = None,
        text_labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        
        bsz, tgt_len = input_ids.size()
        inputs_embeds = self.llm.model.embed_tokens(input_ids)

        if tgt_len > 1:
            # audio encoder
            audio_embeds = self.asr_encoder(
                audio_features=audio_features,
                audio_attention_mask=audio_attention_mask,
                wavlm_features=wavlm_features,
                wavlm_attention_mask=wavlm_attention_mask,
                audio_positions=audio_positions,
            )

            # concat audio embeds
            for batch_idx, audio_start_p, audio_len in audio_positions:
                inputs_embeds[batch_idx, audio_start_p:audio_start_p+audio_len, :] = audio_embeds[batch_idx, :, :]

        # llm
        encoder_outputs: CausalLMOutputWithPast = self.llm(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values.llm_attention_cache if past_key_values is not None else None,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            cache_position=cache_position,
        )

        loss = None
        if text_labels is not None:
            loss = self.loss_function(logits=encoder_outputs.logits, labels=text_labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=encoder_outputs.logits,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

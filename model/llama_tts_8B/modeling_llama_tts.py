"""PyTorch LLaMA TTS model."""

from typing import List, Optional, Tuple, Union

import torch

from transformers.utils import logging
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import get_checkpoint_shard_files
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from avater_infer.models.llama import TTSAdapter, LlamaTTSPreTrainedModel
from avater_infer.cache_utils import AvaterCache
from avater_infer.modeling_utils import get_archive_file
from avater_infer.modeling_outputs import AdapterModelOutputWithPastAndCrossAttentions, Seq2SeqCausalLMOutputWithCrossAttentions


from .configuration_llama_tts import LlamaTTSConfig


logger = logging.get_logger(__name__)


class LlamaTTSForCausalLM(LlamaTTSPreTrainedModel, GenerationMixin):
    def __init__(self, config: LlamaTTSConfig):
        super().__init__(config)
        
        self.config = config
        self.audio_vocab_size = config.audio_vocab_size

        self.llm = LlamaForCausalLM(config)
        self.tts_adapter = TTSAdapter(config)

        # Initialize weights and apply final processing
        self.post_init()

        # frozen llm
        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

    def __repr__(self):
        return self.tts_adapter.__repr__()

    def get_encoder(self,):
        return self.llm

    def load_llm_state_dict(self, llm_pretrained_model_name_or_path):
        archive_file, is_sharded = get_archive_file(llm_pretrained_model_name_or_path)
        if is_sharded:
            archive_file, sharded_metadata = get_checkpoint_shard_files(
                llm_pretrained_model_name_or_path,
                archive_file,
            )

        state_dict = None

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_state_dict_keys = list(state_dict.keys())

        (
            self.llm,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
            offload_index,
            error_msgs,
        ) = self._load_pretrained_model(
            self.llm,
            state_dict,
            loaded_state_dict_keys,  # XXX: rename?
            archive_file,
            llm_pretrained_model_name_or_path
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        valid_tokens_pos: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_input_ids: List[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        encoder_decoder_attention_mask: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_labels: List[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        past_key_values: Optional[AvaterCache] = None,
        **kwargs,
    ) -> Union[Tuple, Seq2SeqCausalLMOutputWithCrossAttentions]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs: CausalLMOutputWithPast = self.llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
                cache_position=cache_position,
            )
            encoder_outputs_hidden_state = encoder_outputs.hidden_states[-1]
        else:
            encoder_outputs_hidden_state = encoder_outputs.hidden_states[-1]

        if valid_tokens_pos is not None:
            batch_size, seq_len, h_dim = encoder_outputs_hidden_state.size()
            pos_bias = (torch.arange(batch_size).view(-1, 1) * seq_len).to(valid_tokens_pos)
            select_index = (valid_tokens_pos+pos_bias).view(-1).to(encoder_outputs_hidden_state.device)
            encoder_outputs_hidden_state = encoder_outputs_hidden_state.view(-1, h_dim).index_select(
                0, select_index
            ).contiguous().view(batch_size, -1, h_dim)

        if len(decoder_input_ids.size()) == 2:
            decoder_input_ids = decoder_input_ids.view(-1, self.config.code_layers, decoder_input_ids.size(-1))
        else:
            assert len(decoder_input_ids.size()) == 3, "decoder_input_ids shape must equal [bsz, code_layers, tgt_len]"

        bsz, code_layers, tgt_len = decoder_input_ids.size()
        decoder_outputs: AdapterModelOutputWithPastAndCrossAttentions = self.tts_adapter(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs_hidden_state,
            encoder_attention_mask=encoder_decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        decoder_logits = decoder_outputs.logits.view(
            bsz, tgt_len, code_layers, -1
        ).transpose(1,2).contiguous()
        decoder_logits = decoder_logits.view(-1, tgt_len, decoder_logits.size(-1))# [bsz * code_layers, tgt_len, audio_vocab]

        # Loss
        loss = None
        if decoder_labels is not None:
            kwargs.pop("labels")
            loss = self.loss_function(
                logits=decoder_logits,
                labels=decoder_labels.view(-1, tgt_len),
                vocab_size=decoder_logits.size(-1),
                **kwargs
            )

        return Seq2SeqCausalLMOutputWithCrossAttentions(
            loss=loss,
            encoder_logits=encoder_outputs.logits,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            decoder_logits=decoder_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

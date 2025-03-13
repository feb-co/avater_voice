import time
import torch
from typing import List, Optional, Tuple, Union

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.tokenization_utils import BatchEncoding
from transformers import AutoModelForCausalLM, GenerationConfig

from avater_infer.cache_utils import AvaterCache
from avater_infer.models.voice.tokenization_voice import AvaterVoiceTokenizer
from avater_infer.modeling_outputs import AdapterModelOutputWithPastAndCrossAttentions

# Voice generation config parameters
voice_generation_config = {
  "do_sample": True,
  "temperature": 0.1,
  "top_p": 0.01,
  "_from_model_config": True,
  "bos_token_id": 128000,
  "eos_token_id": 2049,
  "decoder_start_token_id": 2048,
  "output_hidden_states": True,
  "max_length": 512
}


class VoiceGenerator(PreTrainedModel, GenerationMixin):
    _supports_cache_class = True

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AvaterVoiceTokenizer, cache: AvaterCache) -> None:
        """
        Initialize Voice Generator with model, tokenizer and cache
        
        Args:
            model: The TTS adapter model for voice generation
            tokenizer: Tokenizer for processing audio tokens
            cache: Shared cache for storing KV states and token info
        """
        super().__init__(model.config)

        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.generation_config = GenerationConfig.from_dict(voice_generation_config)
        self.cuda_stream = torch.cuda.Stream()

        self.test_time = time.time()

    def _prepare_inputs(self, conversation):
        """
        Prepare input tensors for model generation
        
        Args:
            conversation: Not directly used but kept for API consistency
            
        Returns:
            Dict containing prepared inputs for the model
        """
        # Initialize with beginning-of-audio token
        inputs = {}
        input_ids = [[self.tokenizer.audio_special_token["boa_token"]]] * self.tokenizer.code_layer
        inputs["input_ids"] = input_ids

        # Create batch and move to correct device
        inputs = BatchEncoding(inputs, tensor_type="pt")
        inputs = {key: tensor.to(self.model.device, non_blocking=True) for key, tensor in inputs.items()}
        inputs["past_key_values"] = self.cache

        return inputs

    def run(self, conversation):
        with torch.cuda.stream(self.cuda_stream):
            # All operations in this context will be queued in this stream
            outputs = GenerationMixin.generate(
                self, 
                **self._prepare_inputs(conversation),
                generation_config=self.generation_config
            )
            self.test_time = time.time() - self.test_time
            return outputs

    def continue_forward(self, input_ids: Optional[torch.LongTensor], past_key_values: Optional[AvaterCache]):
        """
        Determine if voice generation should wait for LLM generation
        
        Args:
            input_ids: Current input token IDs
            past_key_values: The shared cache
            
        Returns:
            Boolean indicating whether to wait for more LLM tokens
        """
        # Get current LLM token count
        llm_tokens_length = len(past_key_values.avater_token_cache)

        # Get current voice token count
        voice_tokens_length = past_key_values.self_attention_cache.get_seq_length(0) + input_ids.shape[1]

        # Wait if we need more LLM tokens and LLM hasn't finished generating
        if_wait = (
            self.tokenizer.get_text_token_requirement(voice_tokens_length) > llm_tokens_length and
            not past_key_values.avater_token_cache.endswith(self.tokenizer.text_tokenizer.eos_token_id)
        )

        return if_wait

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        past_key_values: Optional[AvaterCache] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass through the voice model
        
        Waits for LLM generation if needed, then processes the encoder states
        to generate voice tokens
        """
        # Wait for LLM generation to complete if necessary
        start_wait = time.time()
        while self.continue_forward(input_ids, past_key_values):
            past_key_values.avater_token_cache.update_event.wait(timeout=0.1)
            if time.time() - start_wait > 1.0:
                print(f"Warning: Timeout waiting for tokens after {time.time() - start_wait:.2f}s")
                break

        # Get encoder states and prepare attention mask
        encoder_input_ids, encoder_hidden_state = past_key_values.avater_token_cache[0]
        encoder_cache_length = past_key_values.cross_attention_cache.get_seq_length(0)

        # Get only the new portion of encoder hidden states
        bsz, src_len = encoder_input_ids.size()
        tgt_len = past_key_values.self_attention_cache.get_seq_length(0) + input_ids.shape[1]
        encoder_hidden_state = encoder_hidden_state[:, encoder_cache_length:].to(self.model.device)

        # Convert text-to-audio attention mask for each item in batch
        encoder_decoder_attention_mask = torch.LongTensor([
            self.tokenizer.convert_t2a_attention_mask(
                encoder_input_ids[idx].tolist(), 
                tgt_len,
                remove_assert=True
            ) for idx in range(bsz)
        ]).to(self.model.device)[:, -1, :]

        # Reshape input_ids to match the expected format [bsz, code_layers, tgt_len]
        if len(input_ids.size()) == 2:
            input_ids = input_ids.view(-1, self.model.config.code_layers, input_ids.size(-1))
        else:
            assert len(input_ids.size()) == 3, "input_ids shape must equal [bsz, code_layers, tgt_len]"

        # Run the decoder model
        bsz, code_layers, tgt_len = input_ids.size()
        decoder_outputs: AdapterModelOutputWithPastAndCrossAttentions = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            encoder_hidden_states=encoder_hidden_state,
            encoder_attention_mask=encoder_decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Reshape logits to the expected format [bsz * code_layers, tgt_len, audio_vocab]
        decoder_logits = decoder_outputs.logits.view(
            bsz, tgt_len, code_layers, -1
        ).transpose(1,2).contiguous()
        decoder_logits = decoder_logits.view(-1, tgt_len, decoder_logits.size(-1))
        decoder_outputs.logits = decoder_logits

        return decoder_outputs

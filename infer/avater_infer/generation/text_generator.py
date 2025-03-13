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


# LLM generation config parameters
llm_generation_config = {
  "do_sample": True,
  "temperature": 0.4,
  "top_p": 0.01,
  "_from_model_config": True,
  "bos_token_id": 128000,
  "eos_token_id": 128009,
  "decoder_start_token_id": 2048,
  "output_hidden_states": True,
  "max_length": 512
}


class LLMGenerator(PreTrainedModel, GenerationMixin):
    _supports_cache_class = True

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AvaterVoiceTokenizer, cache: AvaterCache) -> None:
        """
        Initialize LLM Generator with model, tokenizer and cache
        
        Args:
            model: The LLM model for text generation
            tokenizer: Tokenizer for processing text
            cache: Shared cache for storing KV states and token info
        """
        super().__init__(model.config)

        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.generation_config = GenerationConfig.from_dict(llm_generation_config)
        self.cuda_stream = torch.cuda.Stream()
        
        self.test_time = time.time()

    def _prepare_inputs(self, conversation):
        """
        Prepare input tensors for model generation
        
        Args:
            conversation: The conversation history to process
            
        Returns:
            Dict containing prepared inputs for the model
        """
        # Format and tokenize conversation
        formatted_chat = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(formatted_chat, add_special_tokens=False)

        # Create batch and move to correct device
        inputs = {"input_ids": [input_ids]}
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

            self.forward(
                input_ids=outputs[:, -1:],
                past_key_values=self.cache
            )
            self.test_time = time.time() - self.test_time
            return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[AvaterCache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass through the LLM model
        
        Updates the token cache with hidden states and token IDs
        """
        # Set default values for configuration options
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run the model forward pass
        encoder_outputs: CausalLMOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values.llm_attention_cache,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        # Update token cache based on input context
        if input_ids.size(1) == 1:
            # For single token updates, include token IDs
            past_key_values.avater_token_cache.update(
                layer_idx=0,
                token_states=encoder_outputs.hidden_states[-1][:, -1:],
                token_ids=input_ids
            )

        # Update past_key_values in the output
        encoder_outputs.past_key_values = past_key_values
        return encoder_outputs

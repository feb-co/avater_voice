from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, Mapping

import torch
from torch import nn

from transformers import AutoModelForCausalLM

from vllm.inputs import (
    INPUT_REGISTRY,
    EncoderDecoderInputs,
    DummyData,
    InputContext,
    token_inputs
)
from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import get_sampler
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models import llama, VllmModelForTextGeneration

from .tts_adapter import TTSAdapter
from avatar_infer.models_vllm.layers.sample import get_avatar_sampler
from avatar_infer.models.voice.configuration_voice import AvatarVoiceConfig
from avatar_infer.dataclass.sequence import TTSSequenceData, AvatarSamplerOutput
from avatar_infer.utils import tts_codes_to_token


def dummy_tts_data_for_llama_voice(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):
    avatar_config: AvatarVoiceConfig = ctx.model_config.hf_config
    # tokenizer = cached_tokenizer_from_config(ctx.model_config)

    # mm_encoder = tokenizer.mistral.instruct_tokenizer.mm_encoder
    # image_token_id = mm_encoder.special_ids.img

    # mm_config = ctx.get_mm_config()
    # num_images = mm_config.limit_per_prompt.get("image", 1)

    # # dummy size
    # size = 256
    # image = Image.new("RGB", (size, size), color=0)

    # encoding = tokenizer.instruct.mm_encoder(ImageChunk(image=image))
    # image_feature_size = len(encoding.tokens)
    # num_image_tokens = image_feature_size * num_images
    # seq_data = SequenceData.from_prompt_token_counts(
    #     (image_token_id, num_image_tokens),
    #     (0, seq_len - num_image_tokens),
    # )

    # mm_data = {"image": num_images * [image]}
    # mm_placeholders = {
    #     "image":
    #     consecutive_placeholder_ranges(num_items=num_images,
    #                                    item_size=image_feature_size)
    # }

    seq_data = TTSSequenceData.from_prompt_token_counts((0, seq_len))
    return DummyData(seq_data)


def input_processor_for_llama_voice(ctx: InputContext, inputs, **mm_processor_kwargs):
    avatar_config: AvatarVoiceConfig = ctx.model_config.hf_config
    
    boa_tokens = [avatar_config.boa_token_id] * avatar_config.code_layers

    deocder = token_inputs(
        prompt_token_ids=[tts_codes_to_token(
            boa_tokens,
        )]
    )
    return EncoderDecoderInputs(encoder=inputs, decoder=deocder)


@INPUT_REGISTRY.register_dummy_data(dummy_tts_data_for_llama_voice)
@INPUT_REGISTRY.register_input_processor(input_processor_for_llama_voice)
class LlamaVoiceForCausalLM(nn.Module, VllmModelForTextGeneration):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # init config
        config: AvatarVoiceConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config
        self.dtype = vllm_config.model_config.dtype

        # init model
        self.llm = llama.LlamaForCausalLM(vllm_config=vllm_config, prefix="llm")        
        self.tts_adapter = TTSAdapter(vllm_config=vllm_config, prefix="tts_adapter")

        # re-init args
        self.llm_hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers

        # sample param
        self.llm_sampler = get_sampler()
        self.tts_sampler = get_avatar_sampler()
        if get_pp_group().is_last_rank:
            self.llm_logits_processor = LogitsProcessor(
                self.llm.unpadded_vocab_size,
                config.vocab_size,
                getattr(config, "logit_scale", 1.0)
            )
            self.tts_logits_processor = LogitsProcessor(
                config.audio_vocab_size,
                config.audio_vocab_size,
                getattr(config, "logit_scale", 1.0)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        encoder_attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        encoder_hidden_states = self.llm(
            input_ids=encoder_input_ids,
            positions=encoder_positions,
            kv_caches=kv_caches[:self.llm_hidden_layers],
            attn_metadata=encoder_attn_metadata,
            intermediate_tensors=intermediate_tensors
        )

        if not get_pp_group().is_last_rank or isinstance(encoder_hidden_states, IntermediateTensors):
            return encoder_hidden_states

        decoder_hidden_states = self.tts_adapter(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches[self.llm_hidden_layers:],
            attn_metadata=attn_metadata
        )

        return IntermediateTensors({
            "llm_hidden_states": encoder_hidden_states,
            "tts_hidden_states": decoder_hidden_states
        })

    def compute_logits(
        self,
        hidden_states: IntermediateTensors,
        sampling_metadata: SamplingMetadata,
        encoder_sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        llm_hidden_states = hidden_states["llm_hidden_states"]
        llm_logits = self.llm_logits_processor(
            self.llm.lm_head,
            llm_hidden_states,
            encoder_sampling_metadata
        )

        tts_hidden_states = hidden_states["tts_hidden_states"]
        tts_logits_cat = []
        for idx in range(self.config.code_layers):
            tts_logits = self.tts_logits_processor(
                self.tts_adapter.adapter_head.head_block[idx],
                tts_hidden_states[..., idx*self.config.tts_adapter_hidden_size:(idx+1)*self.config.tts_adapter_hidden_size],
                sampling_metadata
            )
            tts_logits_cat.append(tts_logits)
        tts_logits_cat = torch.cat(tts_logits_cat, dim=-1).view(-1, self.config.code_layers, self.tts_adapter.adapter_head.vocab_size)
        return IntermediateTensors({
            "llm_logits": llm_logits,
            "tts_logits": tts_logits_cat
        })

    def sample(
        self,
        logits: IntermediateTensors,
        sampling_metadata: SamplingMetadata,
        encoder_sampling_metadata: SamplingMetadata
    ) -> Optional[AvatarSamplerOutput]:
        llm_logits = logits["llm_logits"]
        tts_logits = logits["tts_logits"]
        llm_next_tokens = self.llm_sampler(llm_logits, encoder_sampling_metadata)
        tts_next_tokens = self.tts_sampler(tts_logits, sampling_metadata)
        return AvatarSamplerOutput(
            llm_outputs=llm_next_tokens.outputs,
            tts_outputs=tts_next_tokens.outputs,
            model_forward_time=llm_next_tokens.model_forward_time,
            model_execute_time=llm_next_tokens.model_execute_time
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        loaded_params: Set[str] = set()

        # load llm weight
        if self.config.llm_path: # only for test
            llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_path)
            llm_state_dict = llm_model.state_dict()
            llm_weights = llm_state_dict.items()
            llm_loaded_params = {f"llm.{param}" for param in self.llm.load_weights(llm_weights)}
            del llm_model, llm_state_dict
            torch.cuda.empty_cache()
        else:
            llm_loaded_params = {f"llm.{param}" for param in self.llm.load_weights(weights)}

        # load tts adapter weight  
        if self.config.tts_path: # only for test
            tts_model = AutoModelForCausalLM.from_pretrained(self.config.tts_path, trust_remote_code=True).tts_adapter
            tts_state_dict = tts_model.state_dict()
            tts_weights = tts_state_dict.items()
            tts_loaded_params = {f"tts_adapter.{param}" for param in self.tts_adapter.load_weights(tts_weights)}
            del tts_model, tts_state_dict
            torch.cuda.empty_cache()
        else:
            tts_loaded_params = {f"tts_adapter.{param}" for param in self.tts_adapter.load_weights(weights)}

        loaded_params.update(llm_loaded_params)
        loaded_params.update(tts_loaded_params)
        return loaded_params

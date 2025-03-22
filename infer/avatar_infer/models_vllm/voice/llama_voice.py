"""Inference-only LLaMA Voice model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union, Mapping

import torch
from torch import nn

from transformers import AutoModelForCausalLM

from vllm.inputs import (INPUT_REGISTRY, DecoderOnlyInputs, DummyData, InputContext, token_inputs)
from vllm.attention import AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.loader import BitsAndBytesModelLoader, LoadConfig, LoadFormat
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models import llama, VllmModelForTextGeneration

from .tts_adapter import TTSAdapter
from avatar_infer.models.voice.configuration_voice import AvatarVoiceConfig
from avatar_infer.generation.sequence import VoiceSequenceData


def dummy_data_for_llama_voice(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):
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

    seq_data = VoiceSequenceData.from_prompt_token_counts(avatar_config.code_layers, (0, seq_len))
    return DummyData(seq_data)


@INPUT_REGISTRY.register_dummy_data(dummy_data_for_llama_voice)
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
        self.llm__hidden_layers = vllm_config.model_config.hf_config.num_hidden_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        r"""
        Args:
            input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            positions
                Positions of *decoder* input sequence tokens.
            encoder_input_ids
                Indices of *encoder* input sequence tokens in the vocabulary.
            encoder_positions:
                Positions of *encoder* input sequence tokens.
            kv_caches:
                Layer-wise list of KV cache tensors
            attn_metadata:
                vLLM Attention metadata structure
        Returns:
            Model output torch.Tensor
        """
        # Run encoder attention if a non-zero number of encoder tokens are provided as input
        encoder_hidden_states = self.llm(
            input_ids=encoder_input_ids,
            positions=encoder_positions,
            kv_caches=kv_caches[:self.llm__hidden_layers],
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors
        )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_hidden_states = self.tts_adapter(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            kv_caches=kv_caches[self.llm__hidden_layers:],
            attn_metadata=attn_metadata
        )

        return IntermediateTensors({
            "encoder_hidden_states": encoder_hidden_states,
            "decoder_hidden_states": decoder_hidden_states
        })

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

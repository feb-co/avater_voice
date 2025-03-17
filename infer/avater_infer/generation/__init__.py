from vllm import ModelRegistry

from .base import AvaterForGeneration
from avater_infer.models_vllm.voice.llama_voice import LlamaVoiceForCausalLM


ModelRegistry.register_model("LlamaVoiceForCausalLM", LlamaVoiceForCausalLM)

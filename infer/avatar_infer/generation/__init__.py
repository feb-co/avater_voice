from vllm import ModelRegistry

from .avatar_llm import AvatarLLM
from avatar_infer.models_vllm.voice.llama_voice import LlamaVoiceForCausalLM


ModelRegistry.register_model("LlamaVoiceForCausalLM", LlamaVoiceForCausalLM)

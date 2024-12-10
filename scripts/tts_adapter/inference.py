import torch
from typing import Tuple
from transformers import AutoConfig, AutoModel


def load_llama_tts():
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_config(model_config, trust_remote_code=True)
    

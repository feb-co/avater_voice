import sys
import torch
from typing import Tuple
from transformers import AutoConfig, AutoModel


def load_llama_tts(model_path, llm_path):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_config(model_config, trust_remote_code=True)
    model.load_llm_state_dict(llm_path)
    return model




if __name__ == "__main__":
    model_path = sys.argv[1]
    llm_path = sys.argv[2]
    model = load_llama_tts(model_path, llm_path)

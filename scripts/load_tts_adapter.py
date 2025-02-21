import os
import sys
import torch
from typing import Optional, List
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.llama_tts_8B.configuration_llama_tts import LlamaTTSConfig
from model.llama_tts_8B.modeling_llama_tts import LlamaTTS


def load_llama_tts(model_path, llm_path):
    model_config: LlamaTTSConfig = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model: LlamaTTS = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    model.load_llm_state_dict(llm_path)
    return model_config, model


def forward_model(
    model: LlamaTTS,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    dummy_valid_tokens_pos: torch.LongTensor,
    decoder_input_ids: List[torch.LongTensor],
    decoder_attention_mask: Optional[torch.Tensor],
    encoder_decoder_attention_mask: Optional[torch.Tensor],
    dummy_labels: List[torch.LongTensor]
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        valid_tokens_pos=dummy_valid_tokens_pos,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        encoder_decoder_attention_mask=encoder_decoder_attention_mask,
        labels=dummy_labels
    )
    return outputs


if __name__ == "__main__":
    model_path = sys.argv[1]
    llm_path = sys.argv[2]

    # Load
    model_config, model = load_llama_tts(model_path, llm_path)
    print(model_config, flush=True)

    # parameter
    text_vocab_size = model_config.vocab_size
    code_size = model_config.code_size
    audio_special_tokens = model_config.audio_special_tokens
    code_layers = model_config.code_layers

    # dummpy dataset
    dummy_input_ids = torch.randint(0, text_vocab_size, (2, 1024), dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    dummy_valid_tokens_pos = torch.tensor([
        [idx for idx in range(24, 1014)]+[0 for _ in range(10)],
        [idx for idx in range(24, 1024)],
    ])
    dummy_input_ids[0, -10:] = 0
    dummy_attention_mask[0, -10:] = 0

    dummy_decoder_input_ids = [
        torch.randint((code_size+audio_special_tokens)*idx, (code_size+audio_special_tokens)*(idx+1), (2, 256), dtype=torch.long)
        for idx in range(code_layers)
    ]
    dummy_labels = [
        dummy_decoder_input_ids[idx].clone() - (code_size+audio_special_tokens) * idx
        for idx in range(code_layers)
    ]
    dummy_decoder_attention_mask = torch.ones_like(dummy_decoder_input_ids[0])
    dummy_encoder_decoder_attention_mask = torch.ones((2, 1, 256, 1000), dtype=torch.long)
    for idx in range(code_layers):
        dummy_decoder_input_ids[idx][1, -56:] = 0
        dummy_labels[idx][1, -56:] = 0
    dummy_decoder_attention_mask[1, -56:] = 0
    dummy_encoder_decoder_attention_mask[0, :, :, -10:] = 0
    dummy_encoder_decoder_attention_mask[1, :, -56:, :] = 0

    # forward
    outputs = forward_model(
        model,
        dummy_input_ids,
        dummy_attention_mask,
        dummy_valid_tokens_pos,
        dummy_decoder_input_ids,
        dummy_decoder_attention_mask,
        dummy_encoder_decoder_attention_mask,
        dummy_labels
    )
    print(outputs)

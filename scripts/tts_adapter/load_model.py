import os
import sys
import torch
from typing import Optional
from transformers import AutoConfig, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.llama_tts.configuration_llama_tts import LlamaTTSConfig
from model.llama_tts.modeling_llama_tts import LlamaTTS


def load_llama_tts(model_path, llm_path):
    model_config: LlamaTTSConfig = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model: LlamaTTS = AutoModel.from_config(model_config, trust_remote_code=True)
    model.load_llm_state_dict(llm_path)
    return model_config, model


def forward_model(
    model: LlamaTTS,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    decoder_input_ids: torch.LongTensor,
    decoder_attention_mask: Optional[torch.Tensor],
    encoder_decoder_attention_mask: Optional[torch.Tensor],
):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        encoder_decoder_attention_mask=encoder_decoder_attention_mask,
        labels=decoder_input_ids
    )
    return outputs


if __name__ == "__main__":
    model_path = sys.argv[1]
    llm_path = sys.argv[2]

    # Load
    model_config, model = load_llama_tts(model_path, llm_path)
    print(model_config, flush=True)

    # Forward
    text_vocab_size = model_config.vocab_size
    audio_vocab_size = model_config.audio_vocab_size

    dummy_input_ids = torch.randint(0, text_vocab_size, (2, 2048), dtype=torch.long)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    dummy_attention_mask[0, -10:] = 0

    dummy_decoder_input_ids = torch.randint(0, audio_vocab_size, (2, 512), dtype=torch.long)
    dummy_decoder_attention_mask = torch.ones_like(dummy_decoder_input_ids)
    dummy_decoder_attention_mask[1, -20:] = 0
    dummy_encoder_decoder_attention_mask = torch.randint(0, 1, (2, 1, 512, 2048), dtype=torch.long)

    outputs = forward_model(
        model,
        dummy_input_ids,
        dummy_attention_mask,
        dummy_decoder_input_ids,
        dummy_decoder_attention_mask,
        dummy_encoder_decoder_attention_mask
    )
    print(outputs)

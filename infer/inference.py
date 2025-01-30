import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
    return tokenizer, model


def inference_tts(model, tokenizer, input_text):
    prefix_text_template = """<|start_header_id|>system<|end_header_id|>

You are Ray Dalio, and you are chatting with the user via voice.<|eot_id|><|start_header_id|>user<|end_header_id|>

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    text_template = """{content}<|eot_id|>"""

    # init encoder input
    prefix_input_ids = tokenizer.encode(prefix_text_template)
    text_input_ids = tokenizer.encode(text_template.format(content=input_text), add_special_tokens=False)
    input_ids = torch.LongTensor([prefix_input_ids+text_input_ids])
    valid_tokens_pos=torch.arange(len(prefix_input_ids), len(prefix_input_ids)+len(text_input_ids)).view(1, -1).to(input_ids)

    # init decoder input
    decoder_input_ids = torch.LongTensor([[tokenizer.audio_code_shift([tokenizer.audio_special_token["boa_token"]], layer_idx=idx) for idx in range(model.config.code_layers)]]).to(input_ids)

    inputs = {
        "input_ids": input_ids.to(model.device),
        "valid_tokens_pos": valid_tokens_pos.to(model.device),
        "decoder_input_ids": decoder_input_ids.to(model.device),
    }

    # generate
    model.generate(**inputs, max_length=4096)


if __name__ == "__main__":
    model_name_and_path = sys.argv[1]

    tokenizer, model = load_model_tokenizer(model_name_and_path)
    inference_tts(model, tokenizer, "hi, I am Ray Dalio.")

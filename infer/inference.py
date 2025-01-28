import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, model


def inference_tts(model, tokenizer, input_text):
    prefix_text_template = """<|start_header_id|>system<|end_header_id|>

You are Ray Dalio, and you are chatting with the user via voice.<|eot_id|><|start_header_id|>user<|end_header_id|>

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    text_template = """{content}<|eot_id|>"""

    prefix_input_ids = tokenizer.encode(prefix_text_template)
    input_ids = tokenizer.encode(text_template.format(content=input_text))
    model.generate(
        inputs=torch.LongTensor([prefix_input_ids+input_ids]),
        valid_tokens_pos=torch.range(len(prefix_input_ids), len(prefix_input_ids)+len(input_ids)),
        max_length=4096
    )


if __name__ == "__main__":
    model_name_and_path = sys.argv[1]

    tokenizer, model = load_model_tokenizer(model_name_and_path)
    inference_tts(model, tokenizer, "hi, I am Ray Dalio.")

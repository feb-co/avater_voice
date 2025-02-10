import os
import sys
import torch
from audiotools import AudioSignal


os.environ["AVATER_LLM_PATH"] = "/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct"
os.environ["AVATER_TEXT_TOKENIZER_PATH"] = "/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct"
os.environ["AVATER_AUDIO_TOKENIZER_PATH"] = "/mnt/ceph/huggingface/AvateAduio-tokenizer"


from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache

from avater_infer.models.patcher import patch_model, patch_init
from avater_infer.models.llama.configuration_voice import LlamaVoiceConfig
from avater_infer.cache_utils import AvaterCache


audio_generation_config = {
  "do_sample": True,
  "temperature": 0.1,
  "top_p": 0.1,
  "_from_model_config": True,
  "bos_token_id": 128000,
  "eos_token_id": 2049,
  "decoder_start_token_id": 2048,
  "output_hidden_states": True,
  "max_length": 512
}



def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model_config = LlamaVoiceConfig.from_pretrained(model_path)
    generation_config = GenerationConfig.from_dict(audio_generation_config)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)

    patch_init(model, tokenizer)
    model, tokenizer = patch_model(model, tokenizer)
    return tokenizer, model, generation_config


def inference_tts(model, tokenizer, generation_config, input_text):
    prefix_text_template = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please repeat the following user's input (which may contain the two special symbols <|SHORT_WAIT|> and <|LONG_WAIT|>):

```
{content}
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    text_template = """{content}<|end_of_text|>"""

    # init encoder input
    prefix_input_ids = tokenizer.encode(prefix_text_template.format(content=input_text))
    text_input_ids = tokenizer.encode(text_template.format(content=input_text), add_special_tokens=False)
    input_ids = torch.LongTensor([prefix_input_ids+text_input_ids])
    valid_tokens_pos=torch.arange(len(prefix_input_ids), len(prefix_input_ids)+len(text_input_ids)).view(1, -1).to(input_ids)

    # init decoder input
    decoder_input_ids = torch.LongTensor([tokenizer.audio_special_token["boa_token"]] * model.config.code_layers)
    decoder_input_ids = decoder_input_ids.view(-1, 1).to(input_ids)

    # init cache
    past_key_values = AvaterCache(
        DynamicCache(),
        DynamicCache(),
        DynamicCache(),
    )

    inputs = {
        "input_ids": input_ids.to(model.device),
        "valid_tokens_pos": valid_tokens_pos.to(model.device),
        "decoder_input_ids": decoder_input_ids.to(model.device),
        "past_key_values": past_key_values,
        "encoder_decoder_attention_mask": torch.ones_like(valid_tokens_pos).to(model.device)
    }

    # generate
    outputs = model.generate(**inputs, generation_config=generation_config)
    audio_codes = outputs[:, 1:-1]

    audio = tokenizer.decode(audio_codes.view(1, audio_codes.size(0), audio_codes.size(-1)))
    audio = AudioSignal(audio, sample_rate=16000)
    audio.to("cpu")
    audio.write("/mnt/ceph/licheng/test.wav")


if __name__ == "__main__":
    model_name_and_path = sys.argv[1]

    tokenizer, model, generation_config = load_model_tokenizer(model_name_and_path)
    inference_tts(
        model, tokenizer, generation_config,
        "Hi, I am Ray Dalio. Who are you?"
    )

import os
import time
import sys
import torch
import asyncio
import soundfile as sf
from audiotools import AudioSignal

from transformers.cache_utils import DynamicCache


os.environ["AVATER_LLM_PATH"] = "/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct"
os.environ["AVATER_TEXT_TOKENIZER_PATH"] = "/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct"
os.environ["AVATER_AUDIO_TOKENIZER_PATH"] = "/mnt/ceph/huggingface/AvateAduio-tokenizer"
os.environ["AVATER_TTS_PATH"] = "/mnt/ceph/licheng/chat_model/tts/llama3.1_tts_8b/tts_2502_synthesis_from_sft/checkpoint-20000/"
os.environ["AVATER_WHISPER_PATH"] = "/mnt/ceph/huggingface/whisper-large-v3/"
os.environ["AVATER_WAVLM_PATH"] = "/mnt/ceph/huggingface/wavlm-large/"


from avater_infer.generation import AvaterForGeneration
from avater_infer.models.patcher import patch_model
from avater_infer.cache_utils import AvaterCache



def inference_tts(model, tokenizer, generation_config, input_text):
    model, tokenizer = patch_model(model, tokenizer)

    prefix_text_template = """<|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please repeat the following user's input (which may contain the two special symbols <|SHORT_WAIT|> and <|LONG_WAIT|>):

```
{content}
```<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    text_template = """{content}<|eot_id|>"""

    # from datasets import DatasetDict, load_dataset, load_from_disk
    # tokenized_data = load_from_disk("/mnt/ceph/licheng/data-bin/train_data_tts_golden")
    # audio_codes = torch.LongTensor(tokenized_data["train"][0]["decoder_input_ids"])[:, 1:-1].to(tokenizer.device)

    # audio = tokenizer.decode(audio_codes.view(1, audio_codes.size(0), audio_codes.size(-1)))
    # audio = AudioSignal(audio, sample_rate=24000)
    # audio.to("cpu")
    # audio.write("/mnt/ceph/licheng/test.wav")
    # import pdb; pdb.set_trace()

    # signal = AudioSignal("/mnt/ceph/licheng/azure_1_24.wav")
    # signal = AudioSignal("/mnt/ceph/licheng/1a55d9a4-e90a-11ef-86a7-4a67e220d664.wav")
    array, sampling_rate = sf.read("/mnt/ceph/licheng/1a55d9a4-e90a-11ef-86a7-4a67e220d664.wav")
    # signal = AudioSignal("/mnt/ceph/licheng/voice_experiment/TTS/data/synthesis_data/chat_faq/waves/f3530414a89811efb0c64a67e220d664.wav")
    _, codes = tokenizer.encode(text=None, audio_signal={"array": array})
    print(len(codes[0]), codes, flush=True)

    # init encoder input
    prefix_input_ids = tokenizer.encode(prefix_text_template.format(content=input_text))
    text_input_ids = tokenizer.encode(text_template.format(content=input_text), add_special_tokens=False)
    input_ids = torch.LongTensor([prefix_input_ids+text_input_ids])
    valid_tokens_pos=torch.arange(len(prefix_input_ids), len(prefix_input_ids)+len(text_input_ids)).view(1, -1).to(input_ids)

    # inputs = {
    #     "labels": None,
    #     "input_ids": input_ids.to(model.device),
    #     "attention_mask": None,
    #     "valid_tokens_pos": valid_tokens_pos.to(model.device),
    #     "decoder_input_ids": torch.LongTensor([codes]).to(model.device),
    #     "decoder_attention_mask": None,
    #     "encoder_decoder_attention_mask": torch.LongTensor([tokenizer.convert_t2a_attention_mask(text_input_ids, codes)]).to(model.device),
    #     "decoder_labels": torch.LongTensor([codes]).to(model.device)
    # }
    # with torch.no_grad():
    #     output = model(**inputs)
    # print("------", output.loss, flush=True)

    # init decoder input
    decoder_input_ids = torch.LongTensor([tokenizer.audio_special_token["boa_token"]] * model.config.code_layers)
    decoder_input_ids = decoder_input_ids.view(-1, 1).to(input_ids)
    encoder_decoder_attention_mask = tokenizer.convert_t2a_attention_mask(text_input_ids, decoder_input_ids, remove_assert=True)

    # init cache
    past_key_values = AvaterCache(
        DynamicCache(),
        DynamicCache(),
        DynamicCache(),
    )

    inputs = {
        "text_input_ids": text_input_ids,
        "input_ids": input_ids.to(model.device),
        "valid_tokens_pos": valid_tokens_pos.to(model.device),
        "decoder_input_ids": decoder_input_ids.to(model.device),
        "past_key_values": past_key_values,
        "encoder_decoder_attention_mask": torch.LongTensor([encoder_decoder_attention_mask]).to(model.device),
    }

    # generate
    outputs = model.generate(**inputs, voice_tokenizer=tokenizer, generation_config=generation_config)
    print(outputs.size(), outputs, flush=True)
    audio_codes = outputs[:, 1:-1]

    audio = tokenizer.decode(audio_codes.view(1, audio_codes.size(0), audio_codes.size(-1)))
    audio = AudioSignal(audio, sample_rate=24000)
    audio.to("cpu")
    audio.write("/mnt/ceph/licheng/test.wav")


async def inference_voice_chat_t1a2(model, conversation):
    avater_generator = AvaterForGeneration(model)
    llm_outputs, voice_outputs = await avater_generator.chat(conversation)
    
    llm_outputs = avater_generator.tokenizer.text_tokenizer.decode(llm_outputs[0])
    voice_outputs = voice_outputs[:, 1:-1].to(avater_generator.tokenizer.device)
    voice_outputs = avater_generator.tokenizer.decode(voice_outputs.view(1, voice_outputs.size(0), voice_outputs.size(-1)))

    audio = AudioSignal(voice_outputs, sample_rate=24000)
    audio.to("cpu")
    audio.write("/mnt/ceph/licheng/test.wav")


if __name__ == "__main__":
    model_name_and_path = sys.argv[1]

    # TTS Task
    # os.environ["AVATER_LLM_PATH"] = "/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct"
    # os.environ["AVATER_TEXT_TOKENIZER_PATH"] = "/mnt/ceph/huggingface/Meta-Llama-3.1-8B-Instruct"
    # os.environ["AVATER_AUDIO_TOKENIZER_PATH"] = "/mnt/ceph/huggingface/AvateAduio-tokenizer"
    # tokenizer, model, generation_config = load_model_tokenizer(model_name_and_path)
    # inference_tts(
    #     model, tokenizer, generation_config,
    #     "So I used it. I worked out of my two bedroom apartment when a pal from hps who I shared the apartment with moved out."
    #     # "The more you do this, the more you will be able to see things from a higher level and develop and refine great principles to help you make better decisions."
    #     # "That's great. And, uh, thank you for talking with me."
    # )

    # Voice Chat T1A2 Task
    asyncio.run(inference_voice_chat_t1a2(
        model_name_and_path,
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """Please repeat the following user's input (which may contain the two special symbols <|SHORT_WAIT|> and <|LONG_WAIT|>):

```
How can I assist you today?
```"""}
        ]
    ))
    # asyncio.run(inference_voice_chat_t1a2(
    #     model_name_and_path,
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": """hi"""}
    #     ]
    # ))

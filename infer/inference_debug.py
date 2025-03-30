import os
import sys
import torch
import asyncio
import soundfile as sf

from transformers.cache_utils import DynamicCache

from avatar_infer.generation.patch import apply_patch

apply_patch()

from avatar_infer.models.patcher import patch_model
from avatar_infer.cache_utils import AvatarCache

from vllm import LLM
from vllm import EngineArgs
from vllm.utils import FlexibleArgumentParser



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
    past_key_values = AvatarCache(
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


def main(args, chat_conversation):
    args: dict = vars(parser.parse_args())
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    # Create an LLM
    avatar_generator = LLM(**args)

    # Create sampling params object
    sampling_params = avatar_generator.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    def print_outputs(outputs):
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}")
            print(f"Generated text: {generated_text!r}")
        print("-" * 80)

    print("=" * 80)

    outputs = avatar_generator.chat(chat_conversation, sampling_params, use_tqdm=True)
    print_outputs(outputs)

    llm_outputs = avatar_generator.tokenizer.text_tokenizer.decode(llm_outputs[0])
    voice_outputs = voice_outputs[:, 1:-1].to(avatar_generator.tokenizer.device)
    voice_outputs = avatar_generator.tokenizer.decode(voice_outputs.view(1, voice_outputs.size(0), voice_outputs.size(-1)))

    if isinstance(voice_outputs, torch.Tensor):
        voice_outputs = voice_outputs.detach().cpu().numpy()

    sf.write("/mnt/ceph/licheng/test.wav", voice_outputs, avatar_generator.tokenizer.audio_tokenizer_sample_rate)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()

    # Add engine args
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
 
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)

    args = parser.parse_args()

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
    chat_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": """Please repeat the following user's input (which may contain the two special symbols <|SHORT_WAIT|> and <|LONG_WAIT|>):

```
How can I assist you today?
```"""}
    ]
    main(args, chat_conversation)
    # asyncio.run(inference_voice_chat_t1a2(
    #     model_name_and_path,
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": """hi"""}
    #     ]
    # ))

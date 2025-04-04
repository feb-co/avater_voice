import os
import sys
import torch
import asyncio
import soundfile as sf

from avatar_infer.generation.patch import apply_patch

apply_patch()

from vllm import LLM
from vllm import EngineArgs
from vllm.utils import FlexibleArgumentParser


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

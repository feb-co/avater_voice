import torch
from typing import Tuple
from transformers import AutoConfig, AutoModel


def count_parameters(model: "torch.nn.Module") -> Tuple[int, int]:
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def test_load(model_path):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print("***** model config *****", flush=True)
    print(model_config, flush=True)
    model = AutoModel.from_config(model_config, trust_remote_code=True)

    trainable_params, all_param = count_parameters(model)
    param_stats = "trainable params: {:.4f}B || all params: {:.4f}B || trainable%: {:.4f}".format(
        trainable_params/(1e9), all_param/(1e9), 100 * trainable_params / all_param
    )
    print("***** model param *****", flush=True)
    print(param_stats, flush=True)
    print("***** model arch *****", flush=True)
    print(model, flush=True)


if __name__ == "__main__":
    model_path = "/mnt/ceph/licheng/avater_voice/model/llama_tts"
    test_load(model_path)

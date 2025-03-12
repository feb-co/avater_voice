import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.modeling_utils import (
    _add_variant,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
)

from avater_infer.models.patcher import patch_init


def get_archive_file(
    pretrained_model_name_or_path,
    subfolder="",
    use_safetensors=True,
    variant=None,
):
    if use_safetensors is not False and os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
    ):
        # Load from a safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
        )
    elif use_safetensors is not False and os.path.isfile(
        os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
        )
    ):
        # Load from a sharded safetensors checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
        )
        is_sharded = True
    elif not use_safetensors and os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
    ):
        # Load from a PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
        )
    elif not use_safetensors and os.path.isfile(
        os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
    ):
        # Load from a sharded PyTorch checkpoint
        archive_file = os.path.join(
            pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
        )
        is_sharded = True
    else:
        raise ValueError(
            "we don't support other type 'archive_file' now!"
        )
    
    return archive_file, is_sharded


def load_model_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype=torch.bfloat16
        ).to(device)
    except:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(
            config, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device)

    patch_init(model, tokenizer)
    model.eval()
    return tokenizer, model

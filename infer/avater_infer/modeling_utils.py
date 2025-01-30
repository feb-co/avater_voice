import os

from transformers.modeling_utils import (
    _add_variant,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
)



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

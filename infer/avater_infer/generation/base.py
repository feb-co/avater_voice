
from avater_infer.modeling_utils import load_model_tokenizer


# audio_generation_config = {
#   "do_sample": True,
#   "temperature": 0.01,
#   "top_p": 0.01,
#   "_from_model_config": True,
#   "bos_token_id": 128000,
#   "eos_token_id": 2049,
#   "decoder_start_token_id": 2048,
#   "output_hidden_states": True,
#   "max_length": 512
# }
# generation_config = GenerationConfig.from_dict(generation_config)


class AvaterForGeneration:
    def __init__(
        self,
        model: str,
    ) -> None:
        self.tokenizer, self.model = load_model_tokenizer(model)

    def chat(
        self,
        conversation,
    ):
        pass

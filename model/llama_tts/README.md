# LLaMA TTS 8B

This is a PyTorch implementation of LLaMA TTS, a text-to-speech model based on LLaMA with 8B parameters.

## Installation

First install the required dependencies:

```bash
# Install Mimi audio tokenizer
cd tokenizer/audio/moshi_mimi
pip install -e .

# Install Whisper for audio feature extraction
pip install openai-whisper
```

## Model Architecture

LLaMA TTS consists of three main components:
1. A frozen LLaMA base model (8B parameters)
2. A TTS adapter with cross-attention mechanism
3. Mimi audio tokenizer for discrete audio representation

The TTS adapter contains:
- Self-attention layers
- Cross-attention layers to LLaMA hidden states
- Feed-forward networks
- Layer normalization

## Usage

### 1. Load Model

```python
from transformers import AutoConfig, AutoModel
from model.llama_tts_8B.modeling_llama_tts import LlamaTTS

def load_model(model_path, llm_path):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_config(model_config, trust_remote_code=True)
    model.load_llm_state_dict(llm_path)
    return model_config, model
```

### 2. Prepare Dataset

```python
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_dir,
    use_fast=False, 
    split_special_tokens=False,
    padding_side="right",
    trust_remote_code=True
)

# Load dataset
dataset = load_single_dataset(
    data_dir=audio_dir,
    data_files=[audio_file],
    tokenizer_dir=tokenizer_dir
)
```

### 3. Model Forward Pass

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    valid_tokens_pos=valid_tokens_pos,
    decoder_input_ids=decoder_input_ids,
    decoder_attention_mask=decoder_attention_mask,
    encoder_decoder_attention_mask=encoder_decoder_attention_mask,
    labels=labels
)
```

## Model Configuration

Key configuration parameters:

```python
config = LlamaTTSConfig(
    audio_special_tokens=8,
    code_size=2048,
    code_layers=8,
    tts_adapter_hidden_layers=6,
    tts_adapter_hidden_size=1024,
    tts_adapter_intermediate_size=2744,
    tts_adapter_attention_heads=16,
    tts_adapter_dropout=0.0,
    tts_adapter_attention_dropout=0.0,
    tie_audio_embeddings=False
)
```

## Citation

If you use this model, please cite:

```
@article{llama_tts,
  title={LLaMA TTS: Text-to-Speech Adaptation of Large Language Models},
  author={...},
  year={2024}
}
```
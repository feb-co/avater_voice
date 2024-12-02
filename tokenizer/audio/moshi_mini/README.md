# Moshi - Mini

This is the PyTorch implementation for Mimi.

## Installation

```bash
cd tokenizer/audio/moshi_mini
pip install -e .
```

## Usage

```python
from audiotools import AudioSignal
from mini.model import MimiTokenizer

def inference_mini(wav_file, outfile):
    """
    Moshi
    """
    print("---------------------------")
    print("Inference audio by Moshi model...")
    signal = AudioSignal(wav_file)
    signal.to("cuda")
    mimi = MimiTokenizer.load_from_checkpoint(
        cpt_dir=xxx,
        "cuda"
    )

    # inference
    print("singal duration:", signal.signal_duration)
    with torch.no_grad():
        codes = mimi.encode(signal.audio_data)
        audio_hat = mimi.decode(codes)
    y = AudioSignal(audio_hat, signal.sample_rate)
    y.to("cpu")

    print("model sample rate:", mimi.sample_rate, "signal sample rate:", signal.sample_rate, "frame_rate: ", mimi.frame_rate, "frame size:", int(mimi.sample_rate / mimi.frame_rate))
    for n_layer, code in enumerate(codes):
        print("quant layer:", n_layer, "code shape:", code.shape)
    print("input shape:", signal.audio_data.shape, "recons shape:", y.audio_data.shape)

    # Write to file
    y.write(outfile)
```

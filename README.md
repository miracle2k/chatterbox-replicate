# Chatterbox TTS Persian - Replicate

Replicate deployment for [Chatterbox TTS Persian](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi), a high-quality Persian/Farsi text-to-speech model.

## Model

- **Base**: [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox) multilingual TTS
- **Fine-tuned weights**: [Thomcles/Chatterbox-TTS-Persian-Farsi](https://huggingface.co/Thomcles/Chatterbox-TTS-Persian-Farsi)
- **License**: CC BY-NC 4.0 (non-commercial use only)

## Usage

```python
import replicate

output = replicate.run(
    "your-username/chatterbox-persian:latest",
    input={
        "text": "سلام! به آزمایش تبدیل متن به گفتار خوش آمدید.",
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
    }
)
print(output)  # URL to generated audio
```

## Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | - | Persian text to synthesize |
| `exaggeration` | float | 0.5 | Exaggeration factor (0.0-1.0). Higher = more expressive |
| `cfg_weight` | float | 0.5 | Classifier-free guidance weight (0.0-1.0) |

## Deployment

1. Create a model on [replicate.com/create](https://replicate.com/create)
2. Add `REPLICATE_API_TOKEN` to GitHub repository secrets
3. Run the "Push to Replicate" workflow with your model name

## Local Development

```bash
# Install Cog
curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)
chmod +x /usr/local/bin/cog

# Download weights
cog run script/download_weights

# Run prediction
cog predict -i text="سلام، حال شما چطور است؟"
```

## Model Size

Total download: ~3.2GB
- Base model (s3gen + ve): ~1.07GB
- Persian weights (t3_fa.safetensors): ~2.14GB


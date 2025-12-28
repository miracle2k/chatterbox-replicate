"""
Chatterbox TTS Persian Predictor for Replicate.

Runs the Chatterbox multilingual model with Persian fine-tuned weights
from Thomcles/Chatterbox-TTS-Persian-Farsi.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath

# Local weights paths (copied into image during build)
LOCAL_WEIGHTS_DIR = Path("/src/weights")
LOCAL_CHATTERBOX_DIR = LOCAL_WEIGHTS_DIR / "chatterbox"
LOCAL_PERSIAN_WEIGHTS = LOCAL_WEIGHTS_DIR / "persian" / "t3_fa.safetensors"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory."""
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from safetensors.torch import load_file as load_safetensors

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading Chatterbox multilingual model on {self.device}...")

        # Check if local weights exist (baked into image)
        if LOCAL_CHATTERBOX_DIR.exists():
            print(f"Loading base model from local weights: {LOCAL_CHATTERBOX_DIR}")
            self.model = ChatterboxMultilingualTTS.from_pretrained(
                str(LOCAL_CHATTERBOX_DIR), device=self.device
            )
        else:
            print("Loading base model from HuggingFace Hub...")
            self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

        # Load Persian fine-tuned weights
        if LOCAL_PERSIAN_WEIGHTS.exists():
            print(f"Loading Persian weights from: {LOCAL_PERSIAN_WEIGHTS}")
            persian_weights_path = LOCAL_PERSIAN_WEIGHTS
        else:
            # Fallback to HuggingFace download (requires token)
            from huggingface_hub import hf_hub_download

            hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            print("Downloading Persian weights from HuggingFace Hub...")
            persian_weights_path = hf_hub_download(
                repo_id="Thomcles/Chatterbox-TTS-Persian-Farsi",
                filename="t3_fa.safetensors",
                token=hf_token,
            )

        print("Loading Persian weights into t3 layer...")
        t3_state = load_safetensors(str(persian_weights_path), device="cpu")
        self.model.t3.load_state_dict(t3_state)
        self.model.t3.to(self.device).eval()

        # Get sample rate from model
        self.sample_rate = getattr(self.model, "sr", 24000)
        print(f"Model loaded successfully. Sample rate: {self.sample_rate}")

    def predict(
        self,
        text: str = Input(
            description="Persian text to synthesize.",
            default="سلام! به آزمایش تبدیل متن به گفتار خوش آمدید.",
        ),
        exaggeration: float = Input(
            description="Exaggeration factor for voice characteristics. Higher = more expressive.",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        cfg_weight: float = Input(
            description="Classifier-free guidance weight. Higher = more adherence to text.",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
    ) -> CogPath:
        """Generate Persian speech from text."""
        import torchaudio as ta

        # Generate audio (language_id=None for Persian fine-tuned model)
        wav = self.model.generate(
            text,
            language_id=None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        # Save to temp file
        output_path = Path(tempfile.mktemp(suffix=".wav"))
        ta.save(str(output_path), wav, self.sample_rate)

        return CogPath(output_path)

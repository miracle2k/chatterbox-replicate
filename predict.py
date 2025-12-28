"""
Chatterbox TTS Persian Predictor for Replicate.

Runs the Chatterbox multilingual model with Persian fine-tuned weights
from Thomcles/Chatterbox-TTS-Persian-Farsi.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cog import BasePredictor, Input, Path as CogPath

# Model references
BASE_MODEL_REPO = "ResembleAI/chatterbox"
PERSIAN_WEIGHTS_REPO = "Thomcles/Chatterbox-TTS-Persian-Farsi"
PERSIAN_WEIGHTS_FILE = "t3_fa.safetensors"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory."""
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors

        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading Chatterbox multilingual model on {self.device}...")
        self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)

        # Download and load Persian fine-tuned weights
        print(f"Downloading Persian weights from {PERSIAN_WEIGHTS_REPO}...")
        persian_weights_path = hf_hub_download(
            repo_id=PERSIAN_WEIGHTS_REPO,
            filename=PERSIAN_WEIGHTS_FILE,
        )

        print("Loading Persian weights into t3 layer...")
        t3_state = load_safetensors(persian_weights_path, device="cpu")
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

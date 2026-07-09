"""Inference wrapper: score audio (file or numpy array) as HUMAN vs AI.

Scores overlapping 3 s windows and returns per-window + average AI probability,
so the app can show a rolling authenticity meter.
"""

from pathlib import Path

import numpy as np
import torch

from .model import VoiceCNN, make_melspec, pick_device, SAMPLE_RATE, CLIP_SAMPLES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "voice_trends" / "voice_cnn_best.pt"
W2V_DIR = ROOT / "voice_trends" / "wav2vec2_finetuned"


class VoiceAuthenticity:
    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        self.device = device or pick_device()
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model = VoiceCNN().to(self.device).eval()
        self.model.load_state_dict(ckpt["state_dict"])
        self.melspec = make_melspec().to(self.device)
        self.val_acc = ckpt.get("val_acc")
        self.trained_on = ckpt.get("gpu", "unknown GPU")

    @torch.no_grad()
    def score_array(self, wav: np.ndarray, sr: int, hop_seconds: float = 1.5):
        """Return (avg_ai_prob, [per-window ai_prob])."""
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1 if wav.shape[-1] <= 2 else 0)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / max(peak, 1e-6)

        t = torch.from_numpy(wav)
        if sr != SAMPLE_RATE:
            import torchaudio
            t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)

        hop = int(hop_seconds * SAMPLE_RATE)
        windows = []
        if len(t) <= CLIP_SAMPLES:
            w = torch.nn.functional.pad(t, (0, CLIP_SAMPLES - len(t)))
            windows.append(w)
        else:
            for start in range(0, len(t) - CLIP_SAMPLES + 1, hop):
                windows.append(t[start:start + CLIP_SAMPLES])

        batch = torch.stack(windows).to(self.device)
        x = self.melspec(batch).unsqueeze(1)
        probs = torch.softmax(self.model(x), dim=1)[:, 1]  # index 1 == "ai"
        probs = probs.cpu().numpy().tolist()
        return float(np.mean(probs)), probs

    def score_file(self, path, **kw):
        import librosa
        wav, sr = librosa.load(path, sr=None, mono=True)
        return self.score_array(wav, sr, **kw)


class W2VAuthenticity(VoiceAuthenticity):
    """Fine-tuned wav2vec2-base detector (preferred when available).

    Same sliding-window interface as the CNN; only the per-window model differs.
    """

    def __init__(self, model_dir=W2V_DIR, device=None):
        from transformers import AutoModelForAudioClassification
        import json

        self.device = device or pick_device()
        self.model = AutoModelForAudioClassification.from_pretrained(
            str(model_dir)).to(self.device).eval()
        self.ai_index = int(self.model.config.label2id.get("ai", 1))
        log = json.loads((Path(model_dir) / "finetune_log.json").read_text())
        self.val_acc = log.get("best_val_acc")
        self.trained_on = log.get("gpu", "unknown GPU")

    @torch.no_grad()
    def _score_windows(self, batch: torch.Tensor):
        batch = (batch - batch.mean(dim=-1, keepdim=True)) / \
                (batch.std(dim=-1, keepdim=True) + 1e-7)
        logits = self.model(input_values=batch.to(self.device)).logits
        return torch.softmax(logits, dim=1)[:, self.ai_index]

    @torch.no_grad()
    def score_array(self, wav: np.ndarray, sr: int, hop_seconds: float = 1.5):
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1 if wav.shape[-1] <= 2 else 0)
        peak = np.abs(wav).max()
        if peak > 0:
            wav = wav / max(peak, 1e-6)

        t = torch.from_numpy(wav)
        if sr != SAMPLE_RATE:
            import torchaudio
            t = torchaudio.functional.resample(t, sr, SAMPLE_RATE)

        hop = int(hop_seconds * SAMPLE_RATE)
        windows = []
        if len(t) <= CLIP_SAMPLES:
            windows.append(torch.nn.functional.pad(t, (0, CLIP_SAMPLES - len(t))))
        else:
            for start in range(0, len(t) - CLIP_SAMPLES + 1, hop):
                windows.append(t[start:start + CLIP_SAMPLES])

        probs = self._score_windows(torch.stack(windows)).cpu().numpy().tolist()
        return float(np.mean(probs)), probs


def load_best_detector(device=None):
    """Prefer the fine-tuned wav2vec2 model; fall back to the CNN baseline."""
    if (W2V_DIR / "model.safetensors").exists() or (W2V_DIR / "pytorch_model.bin").exists():
        try:
            return W2VAuthenticity(device=device), "wav2vec2-base (fine-tuned)"
        except Exception as e:
            print(f"wav2vec2 load failed ({e}); falling back to CNN")
    return VoiceAuthenticity(device=device), "CNN baseline"

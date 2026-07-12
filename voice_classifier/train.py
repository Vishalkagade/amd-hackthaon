"""Train the HUMAN vs AI voice classifier.

Works identically on NVIDIA (CUDA) and AMD (ROCm) GPUs — ROCm PyTorch exposes
the GPU through the same torch.cuda API.

Usage:
    python -m voice_classifier.train --epochs 12 --out voice_trends
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader

from .model import (
    VoiceCNN, make_melspec, pick_device,
    SAMPLE_RATE, CLIP_SAMPLES, LABELS,
)

ROOT = Path(__file__).resolve().parent.parent


def codec_artifacts(wav: torch.Tensor) -> torch.Tensor:
    """Approximate lossy-codec (mp3/AAC) damage without an ffmpeg dependency.

    Zeroes a random slice of high-frequency STFT bins and quantises the rest —
    the perceptual-coder behaviours that a detector could otherwise mistake for
    synthesis artefacts.
    """
    spec = torch.stft(wav, n_fft=512, hop_length=128,
                      window=torch.hann_window(512), return_complex=True)
    n_bins = spec.shape[0]
    cutoff = random.randint(int(n_bins * 0.55), n_bins)
    spec[cutoff:] = 0                                    # coder drops HF content
    mag, phase = spec.abs(), spec.angle()
    step = mag.max() / random.choice([64, 128, 256])     # coarse quantisation
    mag = (mag / (step + 1e-9)).round() * step
    spec = torch.polar(mag, phase)
    out = torch.istft(spec, n_fft=512, hop_length=128,
                      window=torch.hann_window(512), length=len(wav))
    return out


class VoiceDataset(Dataset):
    """Loads wavs, cuts a random 3 s window, applies phone-call-style augmentation.

    Augmentation is what makes this generalise to real microphones and phone
    audio (the failure mode of off-the-shelf detectors on our own voices):
      - random gain
      - additive noise at random SNR
      - random 8 kHz round-trip (telephone bandwidth)
    """

    def __init__(self, files, labels, train: bool):
        self.files = files
        self.labels = labels
        self.train = train
        self.resample_down = torchaudio.transforms.Resample(SAMPLE_RATE, 8000)
        self.resample_up = torchaudio.transforms.Resample(8000, SAMPLE_RATE)

    def __len__(self):
        return len(self.files)

    def _load(self, path):
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim > 1:
            wav = wav.mean(1)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        return wav

    def __getitem__(self, i):
        wav = self._load(self.files[i])

        if len(wav) < CLIP_SAMPLES:
            wav = torch.nn.functional.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        elif self.train:
            start = random.randint(0, len(wav) - CLIP_SAMPLES)
            wav = wav[start:start + CLIP_SAMPLES]
        else:
            wav = wav[:CLIP_SAMPLES]

        if self.train:
            wav = wav * random.uniform(0.5, 1.4)
            if random.random() < 0.5:
                snr_db = random.uniform(8, 35)
                noise = torch.randn_like(wav)
                sig_p, noise_p = wav.pow(2).mean(), noise.pow(2).mean()
                k = torch.sqrt(sig_p / (noise_p * 10 ** (snr_db / 10) + 1e-9))
                wav = wav + k * noise
            if random.random() < 0.3:
                wav = self.resample_up(self.resample_down(wav))
            # Lossy-codec artefacts must not become a "synthetic" cue: our TTS
            # clips arrive mp3-encoded while LibriSpeech does not, so without
            # this the model learns "compressed = AI" and flags real phone
            # recordings. Applied to BOTH classes.
            if random.random() < 0.4:
                wav = codec_artifacts(wav)
            wav = wav.clamp(-1, 1)

        return wav, self.labels[i]


def build_splits(data_dir: Path):
    """Voice-disjoint split: validation voices/speakers never appear in training."""
    from .splits import build_disjoint_splits
    tr_f, tr_y, va_f, va_y = build_disjoint_splits(data_dir)
    return VoiceDataset(tr_f, tr_y, True), VoiceDataset(va_f, va_y, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "voice_trends"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"device={device} ({gpu_name})")

    tr, va = build_splits(Path(args.data))
    print(f"train={len(tr)}  val={len(va)}")
    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(va, batch_size=args.batch_size, num_workers=args.workers)

    # Clone-val (unseen-speaker XTTS clones) and ITW-val (unseen-speaker
    # real-world deepfakes): part of the checkpoint-selection metric so
    # generalization learning isn't discarded.
    from .splits import clone_splits, itw_splits
    _, clone_val, _ = clone_splits(Path(args.data))
    dl_cv = None
    if clone_val:
        cv = VoiceDataset(clone_val, [LABELS.index("ai")] * len(clone_val), False)
        dl_cv = DataLoader(cv, batch_size=args.batch_size, num_workers=args.workers)
        print(f"clone-val={len(cv)}")
    itw_va_f, itw_va_y = itw_splits()["val"]
    dl_itw = None
    if itw_va_f:
        iv = VoiceDataset(itw_va_f, itw_va_y, False)
        dl_itw = DataLoader(iv, batch_size=args.batch_size, num_workers=args.workers)
        print(f"itw-val={len(iv)}")

    model = VoiceCNN().to(device)
    melspec = make_melspec().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tr_loss, n = time.time(), 0.0, 0
        for wav, y in dl_tr:
            wav, y = wav.to(device), y.to(device)
            with torch.no_grad():
                x = melspec(wav).unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(y)
            n += len(y)
        sched.step()

        model.eval()

        def loader_acc(loader):
            correct, total = 0, 0
            with torch.no_grad():
                for wav, y in loader:
                    wav, y = wav.to(device), y.to(device)
                    x = melspec(wav).unsqueeze(1)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += len(y)
            return correct / max(total, 1)

        acc = loader_acc(dl_va)
        cv_acc = loader_acc(dl_cv) if dl_cv else None
        itw_acc = loader_acc(dl_itw) if dl_itw else None
        parts = [m for m in (acc, cv_acc, itw_acc) if m is not None]
        select = sum(parts) / len(parts)
        history.append({"epoch": epoch, "train_loss": tr_loss / n, "val_acc": acc,
                        "clone_val_acc": cv_acc, "itw_val_acc": itw_acc,
                        "seconds": round(time.time() - t0, 1)})
        fmt = lambda v: f"{v:.4f}" if v is not None else "  -   "
        print(f"epoch {epoch:2d}  loss {tr_loss/n:.4f}  val_acc {acc:.4f}  "
              f"clone_val {fmt(cv_acc)}  itw_val {fmt(itw_acc)}  "
              f"({history[-1]['seconds']}s)")

        if select > best_acc:
            best_acc = select
            torch.save({
                "state_dict": model.state_dict(),
                "labels": LABELS,
                "val_acc": acc,
                "gpu": gpu_name,
                "epoch": epoch,
            }, out_dir / "voice_cnn_best.pt")

    (out_dir / "training_log.json").write_text(json.dumps({
        "gpu": gpu_name,
        "torch": torch.__version__,
        "best_val_acc": best_acc,
        "history": history,
    }, indent=2))
    print(f"best val_acc={best_acc:.4f}  saved to {out_dir}/voice_cnn_best.pt")


if __name__ == "__main__":
    main()

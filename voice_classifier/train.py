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
            wav = wav.clamp(-1, 1)

        return wav, self.labels[i]


def build_splits(data_dir: Path, val_frac=0.12):
    files, labels = [], []
    for label_idx, name in enumerate(LABELS):
        for f in sorted((data_dir / name).glob("*.wav")):
            files.append(str(f))
            labels.append(label_idx)
    idx = list(range(len(files)))
    random.Random(123).shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx, train_idx = set(idx[:n_val]), idx[n_val:]
    tr = VoiceDataset([files[i] for i in train_idx], [labels[i] for i in train_idx], True)
    va = VoiceDataset([files[i] for i in sorted(val_idx)], [labels[i] for i in sorted(val_idx)], False)
    return tr, va


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
        correct, total = 0, 0
        with torch.no_grad():
            for wav, y in dl_va:
                wav, y = wav.to(device), y.to(device)
                x = melspec(wav).unsqueeze(1)
                correct += (model(x).argmax(1) == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": tr_loss / n, "val_acc": acc,
                        "seconds": round(time.time() - t0, 1)})
        print(f"epoch {epoch:2d}  loss {tr_loss/n:.4f}  val_acc {acc:.4f}  "
              f"({history[-1]['seconds']}s)")

        if acc > best_acc:
            best_acc = acc
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

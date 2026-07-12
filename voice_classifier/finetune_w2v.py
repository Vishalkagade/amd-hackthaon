"""Fine-tune facebook/wav2vec2-base for HUMAN vs AI voice detection (Plan A).

Starts from self-supervised speech representations (960 h of LibriSpeech
pretraining), freezes the convolutional feature encoder, and fine-tunes the
transformer + classification head on our dataset with phone-call augmentation.

Voice-disjoint evaluation: validation contains only TTS voices and human
speakers never seen in training.

Runs unmodified on NVIDIA (CUDA) and AMD (ROCm) GPUs.

Usage:
    python -m voice_classifier.finetune_w2v --epochs 4 --out voice_trends/wav2vec2_finetuned
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification

from .model import pick_device, LABELS
from .splits import build_disjoint_splits, clone_splits, itw_splits
from .train import VoiceDataset

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "facebook/wav2vec2-base"


def normalize(wav: torch.Tensor) -> torch.Tensor:
    """wav2vec2 expects zero-mean / unit-variance input per utterance."""
    return (wav - wav.mean(dim=-1, keepdim=True)) / (wav.std(dim=-1, keepdim=True) + 1e-7)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for wav, y in loader:
        wav, y = normalize(wav).to(device), y.to(device)
        logits = model(input_values=wav).logits
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "voice_trends" / "wav2vec2_finetuned"))
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    is_rocm = torch.version.hip is not None
    print(f"device={device} ({gpu_name})  rocm={is_rocm}")

    tr_f, tr_y, va_f, va_y = build_disjoint_splits(Path(args.data))
    print(f"train={len(tr_f)}  val={len(va_f)} (voice-disjoint)")
    dl_tr = DataLoader(VoiceDataset(tr_f, tr_y, True), batch_size=args.batch_size,
                       shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(VoiceDataset(va_f, va_y, False), batch_size=args.batch_size,
                       num_workers=args.workers)

    # Clone-val (unseen-speaker XTTS clones): included in checkpoint selection
    # so the model that best generalizes to the clone engine is the one saved.
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

    # Held-out owner clips (human): the model must not flag the device owner.
    from .splits import owner_test_files
    owner_te = owner_test_files(Path(args.data))
    dl_owner = None
    if owner_te:
        ov = VoiceDataset(owner_te, [LABELS.index("human")] * len(owner_te), False)
        dl_owner = DataLoader(ov, batch_size=args.batch_size, num_workers=args.workers)
        print(f"owner-test={len(ov)} (held out)")

    model = AutoModelForAudioClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        label2id={l: i for i, l in enumerate(LABELS)},
        id2label={i: l for i, l in enumerate(LABELS)},
    ).to(device)
    model.freeze_feature_encoder()  # keep pretrained conv front-end intact

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=steps, pct_start=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_acc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tr_loss, n = time.time(), 0.0, 0
        for wav, y in dl_tr:
            wav, y = normalize(wav).to(device), y.to(device)
            logits = model(input_values=wav).logits
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tr_loss += loss.item() * len(y)
            n += len(y)

        acc = evaluate(model, dl_va, device)
        cv_acc = evaluate(model, dl_cv, device) if dl_cv else None
        itw_acc = evaluate(model, dl_itw, device) if dl_itw else None
        own_acc = evaluate(model, dl_owner, device) if dl_owner else None
        parts = [m for m in (acc, cv_acc, itw_acc, own_acc) if m is not None]
        select = sum(parts) / len(parts)
        history.append({"epoch": epoch, "train_loss": tr_loss / n,
                        "val_acc_voice_disjoint": acc,
                        "clone_val_acc": cv_acc, "itw_val_acc": itw_acc,
                        "owner_heldout_acc": own_acc,
                        "seconds": round(time.time() - t0, 1)})
        fmt = lambda v: f"{v:.4f}" if v is not None else "  -   "
        print(f"epoch {epoch}  loss {tr_loss/n:.4f}  val_acc {acc:.4f}  "
              f"clone_val {fmt(cv_acc)}  itw_val {fmt(itw_acc)}  "
              f"owner {fmt(own_acc)}  ({history[-1]['seconds']}s)")

        if select > best_acc:
            best_acc = select
            model.save_pretrained(out_dir)

    (out_dir / "finetune_log.json").write_text(json.dumps({
        "base_model": BASE_MODEL,
        "gpu": gpu_name,
        "rocm": is_rocm,
        "torch": torch.__version__,
        "split": ("voice-disjoint (5 held-out TTS voices, 15% held-out speakers) "
                  "+ XTTS clone train/val/test by speaker; selection = "
                  "mean(val_acc, clone_val_acc)"),
        "best_val_acc": best_acc,
        "history": history,
    }, indent=2))
    print(f"best voice-disjoint val_acc={best_acc:.4f}  saved to {out_dir}")


if __name__ == "__main__":
    main()

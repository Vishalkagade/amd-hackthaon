"""Voice enrollment: teach the detector what the *device owner* sounds like.

Why this exists: our human class (LibriSpeech + In-the-Wild) contains studio read
speech and compressed internet audio — but no modern consumer phone recordings,
which are clean, close-mic'd, noise-suppressed and mp3-encoded. Our AI class
(edge-tts) *is* clean and mp3-encoded, so the model learned "clean + compressed =
synthetic" and flagged real phone recordings as AI.

The product answer is the honest one: a shipping app enrolls your voice on setup.
This script slices the owner's recordings into clips, holding out a contiguous
tail chunk that training never sees, so we can verify the model generalises to
unseen audio of the owner rather than memorising the enrolled clips.

    python -m voice_classifier.enroll --test-frac 0.25
"""

import argparse
import shutil
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "hard_samples"
OUT_TRAIN = ROOT / "data" / "owner_train"
OUT_TEST = ROOT / "data" / "owner_test"
SR = 16000
CLIP_SECONDS = 6.0
MIN_RMS = 0.005  # drop near-silent slices


def slice_file(path: Path, test_frac: float):
    y, _ = librosa.load(path, sr=SR, mono=True)
    n = int(CLIP_SECONDS * SR)
    clips = [y[i:i + n] for i in range(0, len(y) - n + 1, n)]
    clips = [c for c in clips if np.sqrt((c ** 2).mean()) > MIN_RMS]
    # Hold out a CONTIGUOUS tail — random clip-level holdout would leak
    # neighbouring audio from the same breath/sentence into training.
    cut = int(len(clips) * (1 - test_frac))
    return clips[:cut], clips[cut:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="vishal_*",
                    help="glob (in hard_samples/) matching the owner's recordings")
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    audio = (".wav", ".mp3", ".m4a", ".flac")
    seen, files = set(), []
    for f in sorted(SRC_DIR.glob(args.pattern)):
        if f.suffix.lower() in audio and "clone" not in f.name and f.name not in seen:
            seen.add(f.name)
            files.append(f)
    if not files:
        raise SystemExit(f"no owner recordings matching {args.pattern!r}")

    for d in (OUT_TRAIN, OUT_TEST):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    n_tr = n_te = 0
    for f in files:
        tr, te = slice_file(f, args.test_frac)
        print(f"{f.name:32s} -> {len(tr)} train, {len(te)} test clips")
        for c in tr:
            sf.write(OUT_TRAIN / f"owner_{n_tr:04d}.wav", c, SR)
            n_tr += 1
        for c in te:
            sf.write(OUT_TEST / f"owner_{n_te:04d}.wav", c, SR)
            n_te += 1

    print(f"\nDONE  train={n_tr} clips ({n_tr*CLIP_SECONDS/60:.1f} min)  "
          f"test={n_te} clips ({n_te*CLIP_SECONDS/60:.1f} min, never trained on)")


if __name__ == "__main__":
    main()

"""Generate a TRUE voice-clone attack test set with XTTS-v2 (Coqui).

Unlike the edge-tts training data (plain TTS, one engine family), XTTS-v2 is a
different engine AND performs actual voice cloning: each output mimics a specific
held-out LibriSpeech speaker from their own reference audio. This measures
engine-disjoint, clone-of-known-speaker generalization — the realistic attack.

Run inside the TTS venv (Coqui deps are isolated there):
    COQUI_TOS_AGREED=1 .venv-tts/bin/python -m voice_classifier.make_xtts_clones --n 150

License note: XTTS-v2 weights are under the Coqui Public Model License
(non-commercial). Used here only to generate an evaluation set.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from .splits import build_disjoint_splits, _human_speaker

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = ROOT / "data_raw"
OUT = DATA / "xtts_clones"
SR = 16000
REF_SECONDS = 10.0
MIN_WORDS, MAX_WORDS = 8, 30


def held_out_human_files():
    """Held-out speakers -> list of their wav paths (from the disjoint split)."""
    _, _, va_f, va_y = build_disjoint_splits(DATA)
    by_speaker = {}
    for f, y in zip(va_f, va_y):
        if y == 0:  # human
            p = Path(f)
            by_speaker.setdefault(_human_speaker(p), []).append(p)
    return by_speaker


def build_reference(files, out_path):
    """Concatenate a speaker's clips up to REF_SECONDS as XTTS reference audio."""
    chunks, total = [], 0.0
    for f in files:
        y, _ = librosa.load(f, sr=SR, mono=True)
        chunks.append(y)
        total += len(y) / SR
        if total >= REF_SECONDS:
            break
    sf.write(out_path, np.concatenate(chunks), SR)


def load_transcripts():
    texts = []
    for trans in (RAW / "LibriSpeech").rglob("*.trans.txt"):
        for line in trans.read_text().splitlines():
            _, text = line.split(" ", 1)
            n = len(text.split())
            if MIN_WORDS <= n <= MAX_WORDS:
                texts.append(text.capitalize())
    random.Random(99).shuffle(texts)
    return texts


def itw_train_refs(max_speakers=30):
    """Reference wavs from ITW train-bucket speakers (noisy internet audio).

    Cloning from noisy real-world references is the realistic attack; clones
    made only from clean LibriSpeech references miss that domain.
    """
    from .splits import itw_splits, ITW_DIR
    import csv
    files, labels = itw_splits()["train"]
    human = [f for f, y in zip(files, labels) if y == 0]
    spk_of = {}
    with (ITW_DIR / "meta.csv").open() as fh:
        for r in csv.DictReader(fh):
            spk_of[str(ITW_DIR / r["file"])] = r["speaker"]
    by_spk = {}
    for f in human:
        by_spk.setdefault(spk_of.get(f, "?"), []).append(f)
    speakers = sorted(by_spk)[:max_speakers]
    return {s: sorted(by_spk[s]) for s in speakers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--itw-refs", action="store_true",
                    help="clone ITW train-bucket speakers (noisy refs) instead "
                         "of held-out LibriSpeech speakers; output goes to "
                         "data/xtts_clones_itwref (training-only data)")
    args = ap.parse_args()

    out_dir = (DATA / "xtts_clones_itwref") if args.itw_refs else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_dir = out_dir / "_refs"
    refs_dir.mkdir(exist_ok=True)

    by_speaker = itw_train_refs() if args.itw_refs else held_out_human_files()
    speakers = sorted(by_speaker)
    print(f"{len(speakers)} reference speakers: {speakers[:8]}...")

    refs = {}
    for spk in speakers:
        safe = spk.replace(" ", "_").replace("/", "_")
        ref = refs_dir / f"{safe}.wav"
        if not ref.exists():
            build_reference(by_speaker[spk], ref)
        refs[spk] = str(ref)

    texts = load_transcripts()

    import torch
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading XTTS-v2 on {device} ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tmp = out_dir / "_tmp.wav"
    made = 0
    for i in tqdm(range(args.n), desc="xtts clones"):
        spk = speakers[i % len(speakers)]
        safe = spk.replace(" ", "_").replace("/", "_")
        out_path = out_dir / f"xtts_{safe}_{i:04d}.wav"
        if out_path.exists():
            made += 1
            continue
        try:
            tts.tts_to_file(text=texts[i], speaker_wav=refs[spk],
                            language="en", file_path=str(tmp))
            y, _ = librosa.load(tmp, sr=SR, mono=True)  # XTTS outputs 24 kHz
            if len(y) >= SR:
                sf.write(out_path, y, SR)
                made += 1
        except Exception as e:
            print(f"skip {i}: {e}")
    if tmp.exists():
        tmp.unlink()
    print(f"DONE  clones={made}  -> {out_dir}")


if __name__ == "__main__":
    main()

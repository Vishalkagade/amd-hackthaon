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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    refs_dir = OUT / "_refs"
    refs_dir.mkdir(exist_ok=True)

    by_speaker = held_out_human_files()
    speakers = sorted(by_speaker)
    print(f"{len(speakers)} held-out speakers: {speakers}")

    refs = {}
    for spk in speakers:
        ref = refs_dir / f"{spk}.wav"
        if not ref.exists():
            build_reference(by_speaker[spk], ref)
        refs[spk] = str(ref)

    texts = load_transcripts()

    import torch
    from TTS.api import TTS
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading XTTS-v2 on {device} ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    tmp = OUT / "_tmp.wav"
    made = 0
    for i in tqdm(range(args.n), desc="xtts clones"):
        spk = speakers[i % len(speakers)]
        out_path = OUT / f"xtts_{spk}_{i:04d}.wav"
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
    print(f"DONE  clones={made}  -> {OUT}")


if __name__ == "__main__":
    main()

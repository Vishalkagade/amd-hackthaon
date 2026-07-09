"""Voice-disjoint train/validation split.

The honest way to evaluate a deepfake detector: the validation set contains
ONLY synthesizer voices and human speakers the model never saw in training.
Random clip-level splits leak voice identity and inflate accuracy.

AI clips are named ai_{i:05d}.wav where voice = EDGE_VOICES[i % 26]
(see prepare_data.py). Human clips are LibriSpeech {speaker}-{chapter}-{utt}.wav.
"""

from pathlib import Path

from .model import LABELS

N_EDGE_VOICES = 26
# Held-out synthesizer voices (mixed gender/accent): AriaNeural, ChristopherNeural,
# MichelleNeural, SoniaNeural, de-KatjaNeural
HELDOUT_VOICE_IDS = {2, 7, 12, 18, 24}
# Fraction of human speakers held out
HELDOUT_SPEAKER_FRAC = 0.15


def _ai_voice_id(path: Path) -> int:
    return int(path.stem.split("_")[1]) % N_EDGE_VOICES


def _human_speaker(path: Path) -> str:
    return path.stem.split("-")[0]


def build_disjoint_splits(data_dir: Path):
    """Return (train_files, train_labels, val_files, val_labels)."""
    data_dir = Path(data_dir)
    tr_f, tr_y, va_f, va_y = [], [], [], []

    ai_label = LABELS.index("ai")
    human_label = LABELS.index("human")

    for f in sorted((data_dir / "ai").glob("*.wav")):
        if _ai_voice_id(f) in HELDOUT_VOICE_IDS:
            va_f.append(str(f)); va_y.append(ai_label)
        else:
            tr_f.append(str(f)); tr_y.append(ai_label)

    speakers = sorted({_human_speaker(f) for f in (data_dir / "human").glob("*.wav")})
    n_held = max(1, int(len(speakers) * HELDOUT_SPEAKER_FRAC))
    held_speakers = set(speakers[::max(1, len(speakers) // n_held)][:n_held])

    for f in sorted((data_dir / "human").glob("*.wav")):
        if _human_speaker(f) in held_speakers:
            va_f.append(str(f)); va_y.append(human_label)
        else:
            tr_f.append(str(f)); tr_y.append(human_label)

    return tr_f, tr_y, va_f, va_y


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data"
    tr_f, tr_y, va_f, va_y = build_disjoint_splits(root)
    print(f"train={len(tr_f)} (ai={sum(tr_y)})  val={len(va_f)} (ai={sum(va_y)})")

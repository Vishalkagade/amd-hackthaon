"""Voice-disjoint train/validation split.

The honest way to evaluate a deepfake detector: the validation set contains
ONLY synthesizer voices and human speakers the model never saw in training.
Random clip-level splits leak voice identity and inflate accuracy.

AI clips are named ai_{i:05d}.wav where voice = EDGE_VOICES[i % 26]
(see prepare_data.py). Human clips are LibriSpeech {speaker}-{chapter}-{utt}.wav.
"""

import csv
import random
from pathlib import Path

from .model import LABELS

ROOT = Path(__file__).resolve().parent.parent
ITW_DIR = ROOT / "data_raw" / "release_in_the_wild"

N_EDGE_VOICES = 26
# Held-out synthesizer voices (mixed gender/accent): AriaNeural, ChristopherNeural,
# MichelleNeural, SoniaNeural, de-KatjaNeural
HELDOUT_VOICE_IDS = {2, 7, 12, 18, 24}
# Fraction of human speakers held out
HELDOUT_SPEAKER_FRAC = 0.15

# XTTS-v2 clone split (engine-disjoint hardening), 3-way by speaker:
#   train speakers  -> clones added to training (oversampled)
#   val speakers    -> used ONLY for checkpoint selection (never trained on)
#   test speakers   -> tier-2/3 evaluation, never seen anywhere
CLONE_TEST_SPEAKERS = {"1320", "2094", "3000", "4446", "61", "7176"}
CLONE_VAL_SPEAKERS = {"1988", "652"}
# Repeat each training clone this many times per epoch (loss-weighting via sampling)
CLONE_OVERSAMPLE = 4

# In-the-Wild (Frank & Schoenherr): real-world celebrity deepfakes incl. voice
# conversion + genuine recordings of the same people. Split BY SPEAKER
# (70/10/20 train/val/test) so evaluation speakers are never trained on.
ITW_TRAIN_CAP = 4000  # per-label cap so ITW doesn't swamp the other sources
# Voice enrollment (see enroll.py): the device owner's own voice, so the model
# knows the one human it must never mistake for a clone. Oversampled because a
# few minutes of audio is tiny next to thousands of other clips.
OWNER_OVERSAMPLE = 12
ITW_VAL_CAP = 600     # checkpoint selection stays fast
# Demo speakers: force into TEST so they are never trained on and never
# influence checkpoint selection.
ITW_FORCE_TEST = {"Donald Trump", "Barack Obama"}


def _ai_voice_id(path: Path) -> int:
    return int(path.stem.split("_")[1]) % N_EDGE_VOICES


def _human_speaker(path: Path) -> str:
    return path.stem.split("-")[0]


def _clone_speaker(path: Path) -> str:
    return path.stem.split("_")[1]  # xtts_{speaker}_{i}.wav


def itw_splits():
    """In-the-Wild -> {'train'|'val'|'test': (files, labels)}, speaker-disjoint.

    meta.csv rows: file,speaker,label with label in {bona-fide, spoof}.
    Returns empty lists if the dataset is not downloaded.
    """
    empty = {k: ([], []) for k in ("train", "val", "test")}
    meta = ITW_DIR / "meta.csv"
    if not meta.exists():
        return empty

    rows = list(csv.DictReader(meta.open()))
    speakers = sorted({r["speaker"] for r in rows})
    bucket_of = {}
    for i, spk in enumerate(speakers):
        bucket_of[spk] = "train" if i % 10 < 7 else ("val" if i % 10 == 7 else "test")
    for spk in ITW_FORCE_TEST:
        bucket_of[spk] = "test"

    ai_label = LABELS.index("ai")
    human_label = LABELS.index("human")
    out = {k: ([], []) for k in ("train", "val", "test")}
    for r in rows:
        f = ITW_DIR / r["file"]
        if not f.exists():
            # Compact upload bundles ship 16-bit FLAC instead of 32-bit float WAV.
            f = f.with_suffix(".flac")
            if not f.exists():
                continue
        y = ai_label if r["label"].strip() == "spoof" else human_label
        files, labels = out[bucket_of[r["speaker"]]]
        files.append(str(f))
        labels.append(y)

    # Cap training per label so ITW doesn't swamp LibriSpeech/edge-tts/XTTS.
    files, labels = out["train"]
    rng = random.Random(2024)
    by_label = {}
    for f, y in zip(files, labels):
        by_label.setdefault(y, []).append(f)
    capped_f, capped_y = [], []
    for y, fl in by_label.items():
        rng.shuffle(fl)
        fl = fl[:ITW_TRAIN_CAP]
        capped_f += fl
        capped_y += [y] * len(fl)
    out["train"] = (capped_f, capped_y)

    files, labels = out["val"]
    if len(files) > ITW_VAL_CAP:
        idx = list(range(len(files)))
        rng.shuffle(idx)
        idx = sorted(idx[:ITW_VAL_CAP])
        out["val"] = ([files[i] for i in idx], [labels[i] for i in idx])
    return out


def clone_splits(data_dir: Path):
    """Split XTTS clones by speaker into (train, val, test) file lists."""
    clones = sorted((Path(data_dir) / "xtts_clones").glob("xtts_*.wav"))
    train, val, test = [], [], []
    for f in clones:
        spk = _clone_speaker(f)
        if spk in CLONE_TEST_SPEAKERS:
            test.append(str(f))
        elif spk in CLONE_VAL_SPEAKERS:
            val.append(str(f))
        else:
            train.append(str(f))
    return train, val, test


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

    # Engine-disjoint hardening: add the train-speaker XTTS clones to training
    # (val stays pure edge-tts, so tier-1 numbers remain comparable).
    # Oversampled: a few hundred clone clips vs ~700 edge-tts clips would be
    # swamped in the loss otherwise.
    clone_train, _, _ = clone_splits(data_dir)
    clone_train = clone_train * CLONE_OVERSAMPLE
    tr_f += clone_train
    tr_y += [ai_label] * len(clone_train)

    # In-the-Wild train-speaker portion (real-world deepfakes incl. voice
    # conversion, and genuine compressed internet audio for the human class).
    itw_f, itw_y = itw_splits()["train"]
    tr_f += itw_f
    tr_y += itw_y

    # XTTS clones made from NOISY internet references (ITW train speakers) —
    # covers the clone-from-internet-audio attack. Training-only data.
    itwref = sorted((data_dir / "xtts_clones_itwref").glob("xtts_*.wav"))
    itwref = [str(f) for f in itwref] * CLONE_OVERSAMPLE
    tr_f += itwref
    tr_y += [ai_label] * len(itwref)

    # Enrolled owner voice (human). The held-out owner_test/ split is never
    # added here — it is the honest check that enrollment generalises.
    owner = sorted((data_dir / "owner_train").glob("owner_*.wav"))
    owner = [str(f) for f in owner] * OWNER_OVERSAMPLE
    tr_f += owner
    tr_y += [human_label] * len(owner)

    return tr_f, tr_y, va_f, va_y


def owner_test_files(data_dir: Path):
    """Held-out clips of the device owner — never trained on."""
    return [str(f) for f in sorted((Path(data_dir) / "owner_test").glob("owner_*.wav"))]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent / "data"
    tr_f, tr_y, va_f, va_y = build_disjoint_splits(root)
    print(f"train={len(tr_f)} (ai={sum(tr_y)})  val={len(va_f)} (ai={sum(va_y)})")

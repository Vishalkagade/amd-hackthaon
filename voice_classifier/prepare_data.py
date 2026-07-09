"""Build the HUMAN vs AI voice dataset.

HUMAN class : LibriSpeech dev-clean + test-clean (real read speech, many speakers)
AI class    : neural TTS renders of the same LibriSpeech transcripts using
              many different edge-tts voices (2026-era neural voices)

Output layout:
    data/human/*.wav   (16 kHz mono)
    data/ai/*.wav      (16 kHz mono)

Usage:
    python -m voice_classifier.prepare_data --max-human 1200 --max-ai 1200
"""

import argparse
import asyncio
import random
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = ROOT / "data_raw"
SR = 16000

LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
]

# A broad mix of edge-tts neural voices (accents, genders) so the model learns
# "neural TTS artefacts", not one vendor's voice.
EDGE_VOICES = [
    "en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-DavisNeural",
    "en-US-AmberNeural", "en-US-AnaNeural", "en-US-BrandonNeural", "en-US-ChristopherNeural",
    "en-US-CoraNeural", "en-US-ElizabethNeural", "en-US-EricNeural", "en-US-JacobNeural",
    "en-US-MichelleNeural", "en-US-MonicaNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
    "en-GB-LibbyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-GB-ThomasNeural",
    "en-AU-NatashaNeural", "en-AU-WilliamNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural",
    "de-DE-KatjaNeural", "de-DE-ConradNeural",
]


def download_librispeech():
    RAW.mkdir(parents=True, exist_ok=True)
    for url in LIBRISPEECH_URLS:
        name = url.split("/")[-1]
        tar_path = RAW / name
        marker = RAW / (name + ".done")
        if marker.exists():
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(url, tar_path)
        print(f"Extracting {name} ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(RAW)
        tar_path.unlink()
        marker.touch()


def collect_librispeech(max_files: int):
    """Convert FLACs to 16 kHz wavs in data/human, return transcripts for TTS."""
    out = DATA / "human"
    out.mkdir(parents=True, exist_ok=True)
    flacs = sorted((RAW / "LibriSpeech").rglob("*.flac"))
    random.Random(42).shuffle(flacs)
    flacs = flacs[:max_files]

    transcripts = {}
    for trans in (RAW / "LibriSpeech").rglob("*.trans.txt"):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            transcripts[utt] = text.capitalize()

    texts = []
    for f in tqdm(flacs, desc="human wavs"):
        wav_path = out / (f.stem + ".wav")
        if not wav_path.exists():
            y, _ = librosa.load(f, sr=SR, mono=True)
            if len(y) < SR:  # skip clips under 1 s
                continue
            sf.write(wav_path, y, SR)
        if f.stem in transcripts:
            texts.append(transcripts[f.stem])
    return texts


async def synth_one(text, voice, out_path, rate, pitch):
    import edge_tts

    tmp = out_path.with_suffix(".mp3")
    tts = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await tts.save(str(tmp))
    y, _ = librosa.load(tmp, sr=SR, mono=True)
    tmp.unlink()
    if len(y) < SR:
        return False
    sf.write(out_path, y, SR)
    return True


async def generate_ai(texts, max_files: int, concurrency: int = 8):
    out = DATA / "ai"
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    rng.shuffle(texts)
    texts = texts[:max_files]

    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=len(texts), desc="ai wavs")

    async def job(i, text):
        voice = EDGE_VOICES[i % len(EDGE_VOICES)]
        rate = rng.choice(["-10%", "-5%", "+0%", "+5%", "+10%"])
        pitch = rng.choice(["-20Hz", "-10Hz", "+0Hz", "+10Hz", "+20Hz"])
        out_path = out / f"ai_{i:05d}.wav"
        if not out_path.exists():
            async with sem:
                try:
                    await synth_one(text, voice, out_path, rate, pitch)
                except Exception as e:
                    print(f"skip {i}: {e}")
        pbar.update(1)

    await asyncio.gather(*[job(i, t) for i, t in enumerate(texts)])
    pbar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-human", type=int, default=1200)
    ap.add_argument("--max-ai", type=int, default=1200)
    args = ap.parse_args()

    download_librispeech()
    texts = collect_librispeech(args.max_human)
    print(f"human clips ready; {len(texts)} transcripts available for TTS")
    asyncio.run(generate_ai(texts, args.max_ai))
    n_h = len(list((DATA / 'human').glob('*.wav')))
    n_a = len(list((DATA / 'ai').glob('*.wav')))
    print(f"DONE  human={n_h}  ai={n_a}")


if __name__ == "__main__":
    main()

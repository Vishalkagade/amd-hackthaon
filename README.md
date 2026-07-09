# 🛡️ Scam Call Shield

**Real-time phone-scam and AI-voice-clone detection.**
AMD Developer Hackathon (ACT II) — Track 3: Unicorn (Open Innovation).

Voice can no longer be trusted as identity: 3 seconds of audio is enough to clone
anyone's voice, and AI voice scams surged over 1,200% in 2025 (FBI: $3B+ in losses).
The human ear has lost this arms race — detection has to move to the device, onto
the live call. Scam Call Shield does exactly that, on two independent axes:

1. **What is being said** — local Whisper transcribes the call on-GPU; an open LLM
   served by **Fireworks AI** scores scam patterns in real time (urgency, bank
   impersonation, gift cards, code requests…) and gives the callee plain-language advice.
2. **Who is speaking** — our **own CNN classifier, trained on an AMD GPU**, scores
   every 3-second window for synthetic-voice artefacts and shows a live
   human-vs-AI authenticity meter.

Off-the-shelf anti-spoofing checkpoints misclassified our real microphone voices as
AI — so we built our own dataset (LibriSpeech for human speech, 26 modern neural TTS
voices for AI speech, with phone-call augmentation: noise, gain, 8 kHz codec
round-trip) and trained the model ourselves.

## Architecture

```
                 ┌─ faster-whisper (local GPU) ─→ transcript ─→ Fireworks AI LLM ─→ scam risk + red flags + advice
call audio ──────┤
 (5 s chunks)    └─ Mel-spectrogram CNN (trained on AMD ROCm) ─→ AI-voice probability meter
```

## AMD compute usage (qualification evidence)

- The voice classifier is trained on an **AMD Instinct GPU via ROCm PyTorch** —
  see [`train_on_amd.ipynb`](train_on_amd.ipynb).
- The training log [`voice_trends/training_log.json`](voice_trends/training_log.json)
  records the GPU device name, torch/ROCm version, and per-epoch metrics from the run.
- The code is device-agnostic: ROCm PyTorch exposes AMD GPUs through the standard
  `torch.cuda` API, so `voice_classifier/train.py` runs unmodified on both stacks.
- Fireworks AI serves its open models on AMD accelerators, so the LLM axis also
  runs on AMD silicon.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # on AMD: install ROCm torch wheels first (see requirements.txt)

cp .env.example .env                   # then paste your Fireworks API key

# 1. Build the dataset (LibriSpeech + edge-tts neural voices), ~20 min
python -m voice_classifier.prepare_data --max-human 1200 --max-ai 1200

# 2. Train (minutes on a GPU); checkpoint lands in voice_trends/
python -m voice_classifier.train --epochs 12

# 3. Launch the demo
python app.py                          # http://localhost:7860
```

## Demo

- Record yourself (or upload a call recording) → green meters.
- Play a scam script or a TTS/voice-clone sample → scam meter and AI-voice meter go red
  mid-call, with red flags and advice for the callee.
- "Simulate real-time" analyses the recording in growing 5-second chunks, exactly as a
  live call monitor would.

## Honest limitations

- The voice classifier is trained against 2026-era neural TTS voices; it raises the
  cost of an attack rather than guaranteeing detection of every future cloning system.
- Real telephone audio (8 kHz, codec-compressed) is approximated via augmentation;
  production deployment would fine-tune on genuine call recordings.

## Repo layout

```
app.py                     Gradio demo (transcript + scam meter + authenticity meter)
transcriber.py             faster-whisper wrapper (local GPU ASR)
scam_detector.py           Fireworks AI scam analysis (JSON verdict)
voice_classifier/
  model.py                 mel-spectrogram CNN + device selection (CUDA/ROCm)
  prepare_data.py          LibriSpeech + edge-tts dataset builder
  train.py                 training loop (device-agnostic)
  infer.py                 sliding-window authenticity scoring
train_on_amd.ipynb         the AMD ROCm training notebook
voice_trends/              trained checkpoint + training log
```

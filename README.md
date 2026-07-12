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
AI (domain shift: they were trained on studio audio and 2019-era TTS) — so we built
our own dataset (LibriSpeech for human speech, 26 modern neural TTS voices for AI
speech, with phone-call augmentation: noise, gain, 8 kHz codec round-trip) and
trained two detectors on it.

## Detector evaluation: four tiers of honesty

Most deepfake-detector numbers are inflated by evaluation leakage. We evaluate at
four difficulty tiers, each **speaker- and voice-disjoint** from training:

- **Tier 1 — unseen voice, same engine**: 5 held-out edge-tts voices + 15% held-out
  LibriSpeech speakers.
- **Tier 2/3 — unseen engine + voice clone of a known speaker**: XTTS-v2 cloning
  held-out LibriSpeech speakers from their own reference audio; test clones are
  speaker-disjoint from training clones.
- **Tier 4 — real-world internet deepfakes**: the In-the-Wild dataset (Frank &
  Schönherr) — deepfakes of celebrities/politicians from the internet (many
  engines, incl. voice conversion) + genuine recordings of the same people,
  split by speaker 70/10/20.

| wav2vec2-base fine-tuned (production) | Detection | False positives on real humans |
|---|---|---|
| Tier 1: unseen edge-tts voices | 100% | 16.8% (LibriSpeech held-out speakers) |
| Tier 2/3: XTTS-v2 clones, unseen speakers | 87.4% | — |
| Tier 4: In-the-Wild deepfakes, unseen speakers | **98.8%** | **5.0%** (ITW genuine recordings) |

Hard-sample spot checks (never in any training/val set):

| Clip | ai_prob | Verdict |
|---|---|---|
| Owner's real voice, held-out clips (phone mic) | 0.001 | ✅ human |
| XTTS-v2 clone of the owner, "send €500" | 0.995 | ✅ caught |
| XTTS-v2 clone of the owner, *benign* text | 0.997 | ✅ caught |
| Real Donald Trump speech (noisy, compressed) | 0.002 | ✅ human |
| Real Barack Obama speech | 0.002 | ✅ human |
| XTTS-v2 Trump clone (made from the real clip above) | 0.995 | ✅ caught |
| Meta MMS-TTS sample (third TTS engine, never seen) | 0.989 | ✅ caught |
| Fake Trump from YouTube (unknown engine, likely voice conversion) | 0.002 | ❌ **missed** |

The story these numbers tell (found the hard way):

1. Trained only on edge-tts, both models scored ~100% on tier 1 but **13–23% on
   XTTS clones** — engine-disjoint generalization is the real problem.
2. Adding XTTS clones + In-the-Wild to training, with checkpoint selection on
   `mean(tier1, clone_val, itw_val, owner)`, fixed it — but only for the
   pretrained model. The from-scratch CNN baseline could only "solve" clones by
   flagging everything (46–58% human false positives); wav2vec2's 960 h of
   self-supervised pretraining is what learns actual synthesis artifacts.
3. **The model then flagged the author's own phone recording as AI (p=0.89).**
   Root cause: our human data (LibriSpeech, internet audio) contains no modern
   consumer-phone recordings, while our AI data is uniformly clean and
   mp3-encoded — so the model had quietly learned *"clean + compressed =
   synthetic"*. Two fixes: **codec augmentation** applied to both classes (so
   compression stops being a class cue), and **voice enrollment** — the owner's
   voice added to the human class, exactly as a shipping app would do at setup.
   Result: 0.89 → 0.001 on held-out clips of the owner, while clones of the
   owner still fire at 0.99.
4. **Honest cost:** removing the compression shortcut lost us a YouTube fake-Trump
   clip we had previously caught (0.79 → 0.002). That clip is likely *voice
   conversion* (real human prosody, swapped timbre) — a distinct attack family we
   have no training data for. We report the miss rather than hide it; RVC-style
   training data is the clear next iteration.

Full numbers: [`voice_trends/deepfake_eval.json`](voice_trends/deepfake_eval.json).
Reproduce: `make_xtts_clones.py` (isolated Coqui venv, see Quickstart) →
`train.py` / `finetune_w2v.py` → `eval_deepfake.py`. XTTS-v2 weights are
CPML-licensed (non-commercial) — used only for the evaluation/hardening set.
The app uses the wav2vec2 detector by default and falls back to the CNN if absent.

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

# 2. (optional but recommended) Build the XTTS-v2 clone attack set.
#    Coqui TTS needs its own venv (it pins transformers 4.x / torch 2.8):
python3 -m venv .venv-tts && .venv-tts/bin/pip install "coqui-tts[codec]" "transformers<5" "torch==2.8.*" "torchaudio==2.8.*" librosa soundfile tqdm
COQUI_TOS_AGREED=1 .venv-tts/bin/python -m voice_classifier.make_xtts_clones --n 350

# 3. Train both detectors (minutes on a GPU); checkpoints land in voice_trends/
python -m voice_classifier.train --epochs 14          # CNN baseline
python -m voice_classifier.finetune_w2v --epochs 5    # wav2vec2 fine-tune (production)

# 4. Evaluate the three-tier honesty table
python -m voice_classifier.eval_deepfake

# 5. Launch the demo
python app.py                          # http://localhost:7860
```

## Demo

- Record yourself (or upload a call recording) → green meters.
- Play a scam script or a TTS/voice-clone sample → scam meter and AI-voice meter go red
  mid-call, with red flags and advice for the callee.
- "Simulate real-time" analyses the recording in growing 5-second chunks, exactly as a
  live call monitor would.

## Deployment roadmap (the AMD story)

1. **Today**: deepfake detection fine-tuned and served on AMD Instinct GPUs (ROCm)
   in the cloud; Whisper runs locally on the user's device.
2. **Next**: quantize/distill the wav2vec2 detector so the full shield — ASR,
   scam analysis, voice authentication — runs on-device, and no call audio ever
   leaves the phone.

## Honest limitations

- These are real working points, not a solved problem: ~85% on unseen-speaker
  clones and ~99% on 2022-era internet deepfakes, at 7–10% false positives. The
  detector raises the cost of an attack; brand-new 2026 commercial engines will
  still land hits until they are added to training — which the pipeline makes a
  one-command job (`make_xtts_clones.py` pattern for any new engine → retrain →
  4-tier eval).
- Real telephone audio (8 kHz, codec-compressed) is approximated via augmentation;
  production deployment would fine-tune on genuine call recordings.
- In-the-Wild's deepfakes are ~2022-era; continuous retraining against current
  engines (F5-TTS, OpenVoice, commercial cloning APIs) is the roadmap.
- **Voice conversion (RVC-style) is not yet covered** — see the missed YouTube
  clip above. TTS and cloning synthesise speech from text; voice conversion keeps
  a real human's prosody and swaps only timbre, leaving far fewer artefacts.
  Training data for it is the top priority for the next iteration.
- Enrollment (the owner's voice in the human class) is what a shipping app does at
  setup. Reported numbers use held-out owner clips the model never trained on.

## Repo layout

```
app.py                     Gradio demo (transcript + scam meter + authenticity meter)
transcriber.py             faster-whisper wrapper (local GPU ASR)
scam_detector.py           Fireworks AI scam analysis (JSON verdict)
voice_classifier/
  model.py                 mel-spectrogram CNN + device selection (CUDA/ROCm)
  prepare_data.py          LibriSpeech + edge-tts dataset builder
  splits.py                voice-disjoint split + 3-way XTTS clone split + enrollment
  enroll.py                voice enrollment: owner's voice -> held-out train/test clips
  train.py                 CNN baseline training loop (device-agnostic)
  finetune_w2v.py          wav2vec2-base fine-tuning (production detector)
  make_xtts_clones.py      XTTS-v2 voice-clone attack set (runs in .venv-tts)
  eval_deepfake.py         three-tier evaluation (edge-tts / XTTS clones / humans)
  infer.py                 sliding-window scoring; prefers wav2vec2, falls back to CNN
train_on_amd.ipynb         the AMD ROCm training notebook (trains both detectors)
voice_trends/              trained checkpoints + training logs
```

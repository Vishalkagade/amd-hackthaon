# Slide deck outline — Scam Call Shield (Track 3: Unicorn)

Keep it to 8 slides. Judges skim: one idea per slide, big numbers, screenshots.

## 1. Title
Scam Call Shield — real-time protection against phone scams and AI voice clones.
Name, AMD Developer Hackathon ACT II, Track 3.

## 2. The problem
- Voice is no longer identity: ~3 s of audio clones anyone.
- AI scams up 1,210% in 2025; deepfake vishing up ~1,600% in one quarter.
- FBI: $3B+ losses involving voice cloning (2025); one Arup call cost $25.6M.
- Current defenses (code words, "call back") fail exactly when victims panic.
→ Detection must move from the human ear to the device, on the live call.

## 3. The idea — two independent axes
- WHAT is said: Whisper (local GPU) → Fireworks AI open LLM → scam risk + red flags + advice.
- WHO is speaking: our own CNN → AI-voice probability, every 3 s window.
- A call can fail either check: scam script read by human, or benign words in a cloned voice.

## 4. Why we trained our own detector (the honest-numbers slide)
- Off-the-shelf anti-spoofing models flagged OUR real voices as AI (trained on old TTS, studio audio).
- Our dataset: LibriSpeech (human) + 26 modern neural TTS voices (AI)
  + phone-call augmentation (noise, gain, 8 kHz codec round-trip).
- Then we attacked ourselves: 470 XTTS-v2 voice clones of held-out speakers +
  the In-the-Wild dataset (real internet deepfakes of celebrities).
  First result: near-perfect on TTS, only 13–23% on real voice clones!
- Fix: clone+ITW-augmented training, generalization-aware checkpoint selection →
  wav2vec2: 100% TTS / 84.6% unseen-speaker clones / 98.6% internet deepfakes.
- Live proof: real Trump speech → 0.008 (human ✅); fake Trump from YouTube →
  0.791 (caught ✅); XTTS Trump clone → 0.989 (caught ✅).
- Killer ablation: the from-scratch CNN could only "solve" clones by calling half
  of real humans AI — self-supervised pretraining (wav2vec2) is what learns true
  synthesis artifacts. Also catches Meta MMS-TTS (3rd engine, never seen): p≈0.99.

## 5. AMD + Fireworks under the hood
- Classifier trained on AMD Instinct GPU via ROCm PyTorch (train_on_amd.ipynb;
  training_log.json records the device name).
- Same code runs CUDA/ROCm unmodified — torch.cuda API.
- Scam analysis served by Fireworks AI (gpt-oss-120b), running on AMD silicon.

## 6. Live demo (screenshot slide)
- Screenshot 1: benign human call → both meters green.
- Screenshot 2: AI voice reading bank-fraud script → both meters red,
  red flags listed, plain-language advice shown mid-call.

## 7. Honest numbers & limitations
- Detects 2026-era neural TTS; raises attacker cost, not a silver bullet.
- Phone audio approximated by augmentation; next step: fine-tune on real call recordings.
- Latency: ~X s per 5 s chunk end-to-end on GPU (fill from demo).

## 8. Roadmap + ask
- On-device mobile inference, telephony integration (SIP/Twilio), multilingual (German next).
- Everything open source: <GitHub URL>.

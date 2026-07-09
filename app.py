"""Scam Call Shield — real-time scam + AI-voice detection demo.

Pipeline per audio chunk:
  audio -> faster-whisper (local GPU) -> transcript
        -> Fireworks AI open LLM      -> scam risk score + red flags + advice
  audio -> our CNN (trained on AMD)   -> AI-voice probability

Run:  python app.py
"""

import os
import tempfile

import gradio as gr
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from transcriber import transcribe
from scam_detector import analyze_transcript
from voice_classifier.infer import VoiceAuthenticity

CHUNK_SECONDS = 5.0

voice_model = None
try:
    voice_model = VoiceAuthenticity()
    print(f"voice model loaded (val_acc={voice_model.val_acc:.3f}, "
          f"trained on {voice_model.trained_on})")
except Exception as e:
    print(f"WARNING: voice model not loaded ({e}) — train it first.")


def meter_html(title, value, label, color):
    pct = int(round(value))
    return f"""
    <div style="font-family:sans-serif;margin:4px 0 12px">
      <div style="display:flex;justify-content:space-between;font-weight:600">
        <span>{title}</span><span style="color:{color}">{label}</span>
      </div>
      <div style="background:#e5e7eb;border-radius:8px;height:22px;overflow:hidden">
        <div style="width:{pct}%;height:100%;background:{color};
             transition:width .4s;border-radius:8px"></div>
      </div>
      <div style="font-size:12px;color:#6b7280">{pct}/100</div>
    </div>"""


def scam_meter(result):
    score = result["risk_score"]
    verdict = result["verdict"]
    color = {"safe": "#16a34a", "suspicious": "#d97706", "scam": "#dc2626"}[verdict]
    html = meter_html("📞 Scam risk", score, verdict.upper(), color)
    if result["red_flags"]:
        flags = "".join(f"<li>{f}</li>" for f in result["red_flags"])
        html += f"<div style='font-family:sans-serif'><b>Red flags:</b><ul>{flags}</ul></div>"
    if result["advice"]:
        html += (f"<div style='font-family:sans-serif;background:{color}22;"
                 f"border-left:4px solid {color};padding:8px;border-radius:4px'>"
                 f"<b>Advice:</b> {result['advice']}</div>")
    return html


def voice_meter(ai_prob):
    if ai_prob is None:
        return "<i>Voice model not loaded — run training first.</i>"
    score = ai_prob * 100
    if score < 35:
        label, color = "LIKELY HUMAN", "#16a34a"
    elif score < 65:
        label, color = "UNCERTAIN", "#d97706"
    else:
        label, color = "LIKELY AI VOICE", "#dc2626"
    return meter_html("🎙️ AI-voice probability", score, label, color)


def analyze_call(audio_path, realtime):
    """Generator: yields (transcript, scam_html, voice_html, status)."""
    if not audio_path:
        yield "", "", "", "Please record or upload audio first."
        return

    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    total_sec = len(wav) / sr
    chunk = int(CHUNK_SECONDS * sr)
    n_chunks = max(1, int(np.ceil(len(wav) / chunk))) if realtime else 1

    transcript = ""
    for ci in range(n_chunks):
        end = len(wav) if not realtime else min((ci + 1) * chunk, len(wav))
        cur = wav[:end]
        cur_sec = end / sr

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, cur, sr)
            tmp = f.name
        try:
            transcript, _ = transcribe(tmp)

            ai_prob = None
            if voice_model is not None:
                ai_prob, _ = voice_model.score_array(cur, sr)

            status = (f"⏱️ analysed {cur_sec:.0f}s / {total_sec:.0f}s — "
                      f"Whisper local GPU + Fireworks AI")
            try:
                scam = analyze_transcript(transcript)
                scam_html = scam_meter(scam)
            except Exception as e:
                scam_html = f"<b>Fireworks call failed:</b> {e}"

            yield transcript, scam_html, voice_meter(ai_prob), status
        finally:
            os.unlink(tmp)

    yield transcript, scam_html, voice_meter(ai_prob), \
        f"✅ done — {total_sec:.0f}s of audio analysed"


with gr.Blocks(title="Scam Call Shield", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# 🛡️ Scam Call Shield
Real-time protection against phone scams and AI voice clones.
**Whisper (local GPU)** transcribes the call → **Fireworks AI open LLM** scores scam
risk → **our own CNN (trained on AMD GPU)** detects synthetic voices."""
    )
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                             label="Call audio (record or upload)")
            realtime = gr.Checkbox(True, label="Simulate real-time (analyse in 5 s chunks)")
            btn = gr.Button("🔍 Analyse call", variant="primary")
            status = gr.Markdown("")
        with gr.Column(scale=1):
            scam_out = gr.HTML(label="Scam analysis")
            voice_out = gr.HTML(label="Voice authenticity")
    transcript_out = gr.Textbox(label="Live transcript", lines=6)

    btn.click(analyze_call, [audio, realtime],
              [transcript_out, scam_out, voice_out, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

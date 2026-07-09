"""Local Whisper transcription via faster-whisper (runs on the GPU)."""

from faster_whisper import WhisperModel

_model = None


def get_model(size: str = "small"):
    global _model
    if _model is None:
        try:
            _model = WhisperModel(size, device="cuda", compute_type="float16")
        except Exception:
            # CPU fallback (e.g. no GPU / ROCm CTranslate2 not available)
            _model = WhisperModel(size, device="cpu", compute_type="int8")
    return _model


def transcribe(path: str, language: str | None = None):
    """Return (full_text, [segments]) where each segment is (start, end, text)."""
    model = get_model()
    segments, info = model.transcribe(path, language=language, vad_filter=True)
    segs = [(s.start, s.end, s.text.strip()) for s in segments]
    text = " ".join(t for _, _, t in segs)
    return text, segs

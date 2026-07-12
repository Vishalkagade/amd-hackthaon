"""Scam analysis of a call transcript using an open model served by Fireworks AI."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

SYSTEM_PROMPT = """You are a real-time phone-call scam detector protecting a \
vulnerable person who is currently on this call. You receive a (possibly partial) \
transcript of what the OTHER party has said so far.

Analyse it for known scam patterns: urgency/pressure, requests for money or gift \
cards or crypto, bank/PayPal/'security department' impersonation, government or \
police impersonation, tech-support fraud, grandchild/family-emergency scams, \
requests for codes/passwords/remote access, threats, too-good-to-be-true offers, \
requests to keep the call secret.

Respond with ONLY a JSON object, no markdown fences:
{
  "risk_score": <integer 0-100>,
  "verdict": "safe" | "suspicious" | "scam",
  "red_flags": ["short phrase", ...],
  "advice": "<one or two short sentences of plain-language advice to the callee>"
}
Be decisive. An ordinary friendly/business call is "safe" with a low score."""


def get_client():
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ.get("FIREWORKS_BASE_URL",
                                "https://api.fireworks.ai/inference/v1"),
    )


def model_label() -> str:
    """Human-readable name of the LLM answering, for the UI badge."""
    model = os.environ.get("FIREWORKS_MODEL", "")
    name = model.rsplit("/", 1)[-1] if model else "unknown"
    if "deployments/" in model:
        name = os.environ.get("FIREWORKS_MODEL_NAME", f"dedicated ({name})")
    return name


def analyze_transcript(transcript: str) -> dict:
    if not transcript.strip():
        return {"risk_score": 0, "verdict": "safe", "red_flags": [],
                "advice": "No speech detected yet."}

    client = get_client()
    model = os.environ.get("FIREWORKS_MODEL",
                           "accounts/fireworks/models/llama-v3p3-70b-instruct")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Call transcript so far:\n\"\"\"\n{transcript}\n\"\"\""},
        ],
        temperature=0.1,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        out = {"risk_score": 50, "verdict": "suspicious",
               "red_flags": ["model returned unparseable output"],
               "advice": raw[:200]}
    out["risk_score"] = int(max(0, min(100, out.get("risk_score", 0))))
    if out.get("verdict") not in ("safe", "suspicious", "scam"):
        out["verdict"] = "suspicious"
    out.setdefault("red_flags", [])
    out.setdefault("advice", "")
    return out


if __name__ == "__main__":
    demo = ("Hello, this is the security department of your bank. We detected "
            "suspicious activity. To protect your savings you must transfer your "
            "money to a safe account right now. Do not hang up and do not tell "
            "anyone. Please read me the code we just sent to your phone.")
    print(json.dumps(analyze_transcript(demo), indent=2))

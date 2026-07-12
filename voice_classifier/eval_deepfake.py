"""Three-tier honest evaluation of both detectors.

Tier 1  same-engine, unseen voice   : voice-disjoint val set (edge-tts + LibriSpeech)
Tier 2+3 unseen engine + clone-of-known-speaker : XTTS-v2 clones of the SAME
         held-out speakers (so speaker identity can't be used as a shortcut)

Also reports the human false-positive rate on held-out speakers, because a
detector that cries "AI" at real people is useless (our original pain point).

Usage:
    python -m voice_classifier.eval_deepfake
"""

import json
from pathlib import Path

import numpy as np

from .splits import build_disjoint_splits, clone_splits, itw_splits
from .infer import VoiceAuthenticity, W2VAuthenticity, W2V_DIR

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLONES = DATA / "xtts_clones"
THRESHOLD = 0.5


def score_files(detector, files):
    return np.array([detector.score_file(str(f))[0] for f in files])


def main():
    _, _, va_f, va_y = build_disjoint_splits(DATA)
    va_human = [f for f, y in zip(va_f, va_y) if y == 0]
    va_ai = [f for f, y in zip(va_f, va_y) if y == 1]
    # Only clones of speakers the models never trained on (speaker-disjoint
    # within the clone engine too).
    _, _, clones = clone_splits(DATA)
    if not clones:
        print("note: no XTTS clones present — skipping tiers 2/3 "
              "(run make_xtts_clones.py to include them)")

    detectors = {"cnn_baseline": VoiceAuthenticity()}
    if (W2V_DIR / "model.safetensors").exists():
        detectors["wav2vec2_finetuned"] = W2VAuthenticity()

    from .model import device_info
    report = {"device": device_info(),   # records the GPU the eval ran on
              "n_val_human": len(va_human), "n_val_ai_edge_tts": len(va_ai),
              "n_xtts_clones": len(clones), "threshold": THRESHOLD}

    # Tier 4: In-the-Wild test speakers (real-world internet deepfakes incl.
    # voice conversion + genuine recordings of the same celebrities).
    itw_f, itw_y = itw_splits()["test"]
    itw_fake = [f for f, y in zip(itw_f, itw_y) if y == 1]
    itw_real = [f for f, y in zip(itw_f, itw_y) if y == 0]
    report["n_itw_test_fake"] = len(itw_fake)
    report["n_itw_test_real"] = len(itw_real)

    for name, det in detectors.items():
        p_human = score_files(det, va_human)
        p_edge = score_files(det, va_ai)

        r = {
            "tier1_edge_tts_unseen_voice_detection": float((p_edge > THRESHOLD).mean()),
            "tier1_mean_ai_prob": float(p_edge.mean()),
            "human_false_positive_rate": float((p_human > THRESHOLD).mean()),
            "human_mean_ai_prob": float(p_human.mean()),
        }
        if clones:
            p_xtts = score_files(det, clones)
            r["tier23_xtts_clone_detection"] = float((p_xtts > THRESHOLD).mean())
            r["tier23_mean_ai_prob"] = float(p_xtts.mean())
        if itw_fake:
            p_if = score_files(det, itw_fake)
            p_ir = score_files(det, itw_real)
            r["tier4_itw_deepfake_detection"] = float((p_if > THRESHOLD).mean())
            r["tier4_itw_mean_ai_prob"] = float(p_if.mean())
            r["tier4_itw_human_false_positive_rate"] = float((p_ir > THRESHOLD).mean())
        hard = sorted((ROOT / "hard_samples").glob("*.wav"))
        if hard:
            r["hard_samples"] = {f.name: round(float(det.score_file(str(f))[0]), 3)
                                 for f in hard}

        report[name] = r
        print(f"\n=== {name} ===")
        print(f"  tier 1  edge-tts unseen voices : {r['tier1_edge_tts_unseen_voice_detection']:.1%} detected "
              f"(mean p={r['tier1_mean_ai_prob']:.3f})")
        if clones:
            print(f"  tier2+3 XTTS-v2 speaker clones : {r['tier23_xtts_clone_detection']:.1%} detected "
                  f"(mean p={r['tier23_mean_ai_prob']:.3f})")
        print(f"  humans (held-out speakers)     : {r['human_false_positive_rate']:.1%} false positives "
              f"(mean p={r['human_mean_ai_prob']:.3f})")
        if itw_fake:
            print(f"  tier 4  In-the-Wild deepfakes  : {r['tier4_itw_deepfake_detection']:.1%} detected "
                  f"(mean p={r['tier4_itw_mean_ai_prob']:.3f}); "
                  f"ITW real humans FPR {r['tier4_itw_human_false_positive_rate']:.1%}")

    out = ROOT / "voice_trends" / "deepfake_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

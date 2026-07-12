"""Pack the In-the-Wild clips we actually train on into a small upload bundle.

The full ITW release is 8.7 GB of 32-bit float WAV. Training only touches the
train+val speaker buckets, and 16-bit FLAC is lossless for our purposes — so the
bundle is a fraction of the size and uploads to a cloud notebook quickly.

    python -m voice_classifier.pack_itw_subset            # train+val (for training)
    python -m voice_classifier.pack_itw_subset --with-test  # + tier-4 eval split

On the AMD box:
    mkdir -p data_raw && tar xzf itw_subset.tar.gz -C data_raw
"""

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from .splits import itw_splits, ITW_DIR

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-test", action="store_true",
                    help="also include the tier-4 evaluation split (+~1 GB)")
    ap.add_argument("--out", default=str(ROOT / "itw_subset.tar.gz"))
    args = ap.parse_args()

    splits = itw_splits()
    keep = ["train", "val"] + (["test"] if args.with_test else [])
    wanted = {Path(f).name for k in keep for f in splits[k][0]}
    print(f"packing {len(wanted)} clips from splits: {', '.join(keep)}")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "release_in_the_wild"
        stage.mkdir(parents=True)

        for name in tqdm(sorted(wanted), desc="wav -> flac"):
            src = ITW_DIR / name
            data, sr = sf.read(src, dtype="float32")
            sf.write(stage / (Path(name).stem + ".flac"), data, sr,
                     format="FLAC", subtype="PCM_16")

        # meta.csv keeps the original .wav names; the loader falls back to .flac.
        rows = list(csv.DictReader((ITW_DIR / "meta.csv").open()))
        kept = [r for r in rows if r["file"] in wanted]
        with (stage / "meta.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(kept)

        out = Path(args.out)
        subprocess.run(["tar", "czf", str(out), "-C", str(Path(tmp)),
                        "release_in_the_wild"], check=True)

    print(f"DONE  {out}  ({out.stat().st_size / 1e9:.2f} GB)")
    print("On the AMD box:  mkdir -p data_raw && tar xzf itw_subset.tar.gz -C data_raw")


if __name__ == "__main__":
    main()

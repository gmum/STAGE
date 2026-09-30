# Samples the caption corpus for K0: 30,000 captions of the ObjaverseXL (Sketchfab) part of
# TRELLIS-500K, one per line.
#
#   python unlearning_methods/k0/download_captions.py
import argparse
import json
import os
import random

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def flatten_captions(df: pd.DataFrame) -> list[str]:
    # Each row stores a JSON list of captions for one asset.
    out: list[str] = []
    for cell in df["captions"].dropna():
        try:
            items = json.loads(cell)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(items, list):
            out.extend(str(x).strip() for x in items if str(x).strip())
        elif isinstance(items, str) and items.strip():
            out.append(items.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "trellis500k_captions.txt"))
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    captions = flatten_captions(df)
    random.Random(args.seed).shuffle(captions)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(captions[: args.n]))
    print(f"{len(df)} assets, {len(captions)} captions, saved {min(args.n, len(captions))} to {args.out}")


if __name__ == "__main__":
    main()

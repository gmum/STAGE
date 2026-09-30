# K0 = E[c c^T] over the token states of the TRELLIS-500K caption sample, from the same CLIP
# text encoder the edits use. Padding positions are excluded.
#
#   python unlearning_methods/k0/compute_k0.py
import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from unlearning_methods.embeddings import CONCEPT_ENCODER_MODEL


@torch.no_grad()
def compute_k0(captions: list[str], batch_size: int) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(CONCEPT_ENCODER_MODEL)
    text_encoder = CLIPTextModel.from_pretrained(CONCEPT_ENCODER_MODEL).to("cpu").eval()
    hidden_dim = text_encoder.config.hidden_size

    C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float64)
    count = 0
    for start in tqdm(range(0, len(captions), batch_size), desc="K0", unit="batch"):
        enc = tokenizer(captions[start : start + batch_size], max_length=77, padding="max_length",
                        truncation=True, return_tensors="pt")
        H = text_encoder(input_ids=enc["input_ids"]).last_hidden_state
        H = H.reshape(-1, hidden_dim)[enc["attention_mask"].reshape(-1) > 0].to(torch.float64)
        C += H.T @ H
        count += H.shape[0]
    return {"C": (C / count).to(torch.float32), "count": count}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", default=os.path.join(PROJECT_ROOT, "data", "trellis500k_captions.txt"))
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "K0_trellis_nopad.pt"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    with open(args.captions, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f if line.strip()]
    result = compute_k0(captions, args.batch_size)
    torch.save(result, args.out)
    print(f"Saved {args.out}: {len(captions)} captions, {result['count']} token states")


if __name__ == "__main__":
    main()

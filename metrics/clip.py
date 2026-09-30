# CLIP evaluator: zero-shot scoring of an asset's rendered views against text labels.
from pathlib import Path

import torch
from PIL import Image

from .parameters import CLIP_MODEL

_encoder = None


class _ClipEncoder:
    def __init__(self):
        from transformers import CLIPModel, CLIPProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)

    @torch.no_grad()
    def encode_images(self, paths: list[Path]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in paths]
        feats = self.model.get_image_features(**self.processor(images=images, return_tensors="pt").to(self.device))
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)


def _get_encoder() -> _ClipEncoder:
    global _encoder
    if _encoder is None:
        _encoder = _ClipEncoder()
    return _encoder


def predict(objects: list[list[Path]], labels: list[str]) -> list[dict[str, float]]:
    # Per asset: softmax over the labels of the view-averaged cosine similarity.
    encoder = _get_encoder()
    text_embeds = encoder.encode_texts(labels)
    results = []
    for views in objects:
        sims = (encoder.encode_images(views) @ text_embeds.T).mean(dim=0)
        results.append(dict(zip(labels, sims.softmax(dim=-1).tolist())))
    return results


def detect(objects: list[list[Path]], labels: list[str]) -> list[dict[str, bool]]:
    # A label is detected in an asset if it is the top-1 label in at least one view.
    encoder = _get_encoder()
    text_embeds = encoder.encode_texts(labels)
    results = []
    for views in objects:
        winners = {labels[i] for i in (encoder.encode_images(views) @ text_embeds.T).argmax(dim=-1).tolist()}
        results.append({label: label in winners for label in labels})
    return results


def align(objects: list[list[Path]], prompts: list[str], sibling_labels: list[str]) -> list[tuple[float, float]]:
    # Per asset: (A, A_bar), the view-averaged similarity to its own prompt and to the labels
    # of the other concepts on its axis.
    encoder = _get_encoder()
    sibling_embeds = encoder.encode_texts(sibling_labels)
    results = []
    for views, prompt in zip(objects, prompts):
        image_embeds = encoder.encode_images(views)
        own_sim = float((image_embeds @ encoder.encode_texts([prompt]).T).mean())
        sibling_sim = float((image_embeds @ sibling_embeds.T).mean())
        results.append((own_sim, sibling_sim))
    return results

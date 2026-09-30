# MINC-23 evaluator: a material classifier on an asset's rendered views, restricted to the
# five materials of the axis and renormalized, so its chance level matches the other evaluators.
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .parameters import MINC_MODEL

_classifier = None


class _MincClassifier:
    def __init__(self):
        from transformers import AutoImageProcessor, SiglipForImageClassification

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipForImageClassification.from_pretrained(MINC_MODEL).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(MINC_MODEL, use_fast=True)
        id2label = self.model.config.id2label
        self.labels = [id2label[i] for i in range(len(id2label))]

    @torch.no_grad()
    def classify(self, paths: list[Path]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in paths]
        return self.model(**self.processor(images=images, return_tensors="pt").to(self.device)).logits


def _get_classifier() -> _MincClassifier:
    global _classifier
    if _classifier is None:
        _classifier = _MincClassifier()
    return _classifier


def predict(objects: list[list[Path]], labels: list[str]) -> list[dict[str, float]]:
    # Per asset: softmax over the labels of the view-averaged logits.
    classifier = _get_classifier()
    idx = [classifier.labels.index(label) for label in labels]
    results = []
    for views in objects:
        probs = classifier.classify(views).mean(dim=0)[idx].softmax(dim=-1)
        results.append(dict(zip(labels, probs.tolist())))
    return results


def detect(objects: list[list[Path]], labels: list[str]) -> list[dict[str, bool]]:
    # A label is detected in an asset if it is the top-1 label in at least one view.
    classifier = _get_classifier()
    idx = [classifier.labels.index(label) for label in labels]
    results = []
    for views in objects:
        winners = {labels[i] for i in classifier.classify(views)[:, idx].argmax(dim=-1).tolist()}
        results.append({label: label in winners for label in labels})
    return results


def align(objects: list[list[Path]], own_labels: list[str], labels: list[str]) -> list[float]:
    # Per asset: the chance-calibrated confidence in its own material, which takes the place of
    # the contrast-corrected similarity A_hat used for CLIP and Uni3D.
    classifier = _get_classifier()
    idx = [classifier.labels.index(label) for label in labels]
    chance = 100.0 / len(labels)
    results = []
    for views, label in zip(objects, own_labels):
        probs = classifier.classify(views).mean(dim=0)[idx].softmax(dim=-1)
        conf_pct = float(probs[labels.index(label)]) * 100.0
        results.append(float(np.clip((conf_pct - chance) / (100.0 - chance), 0.0, 1.0)))
    return results

# Applies OCE to the to_kv projections of the edited stages, one embedding per prompt.
import torch
import torch.nn as nn

from .core import edit_weight
from .parameters import EDIT_WK, EDIT_WV
from ..embeddings import ConceptEncoder
from ..targets import Mode, kv_halves, kv_projections


def apply_edit(gs_model: nn.Module, gl_model: nn.Module, mode: Mode, encoder: ConceptEncoder,
               erase_prompts: list[str], anchor_prompts: list[str], retain_prompts: list[str] | None,
               K0: torch.Tensor) -> None:
    erase = encoder.concept_vectors(erase_prompts)
    anchor = encoder.concept_vectors(anchor_prompts)
    preserve = encoder.concept_vectors(retain_prompts) if retain_prompts else None
    for linear in kv_projections(gs_model, gl_model, mode):
        new_weight = linear.weight.data.detach().clone()
        for rows in kv_halves(linear, EDIT_WK, EDIT_WV):
            new_weight[rows] = edit_weight(linear.weight.data[rows], erase, anchor, preserve, K0)
        with torch.no_grad():
            linear.weight.copy_(new_weight)

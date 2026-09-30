# Fits STAGE on the edit prompts and folds it into the to_kv projections of the edited stages.
# D, N and n0 depend only on text embeddings, so one solve serves every layer.
import torch
import torch.nn as nn

from .core import build_pairs, fold, solve_map
from .parameters import (
    ALPHA, EDIT_WK, EDIT_WV, HOLD_SCALE, LAMB, NULL_SCALE, PRESERVE_GLOBAL_SCALE, RANK, TOKEN_PAIRS, VARIANT,
)
from ..embeddings import ConceptEncoder
from ..targets import Mode, kv_halves, kv_projections


def _f64(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(device="cpu", dtype=torch.float64)


def apply_edit(gs_model: nn.Module, gl_model: nn.Module, mode: Mode, encoder: ConceptEncoder,
               erase_prompts: list[str], anchor_prompts: list[str], retain_prompts: list[str] | None,
               K0: torch.Tensor) -> None:
    erase_hidden, erase_mask = encoder.concept_sequences(erase_prompts)
    anchor_hidden, anchor_mask = encoder.concept_sequences(anchor_prompts)
    x_e, x_a = build_pairs(_f64(erase_hidden), erase_mask, _f64(anchor_hidden), anchor_mask, TOKEN_PAIRS)
    d = x_e.shape[1]

    # Retained set H: the paired anchor states, all token states of the retain prompts, and all
    # 77 token states of the empty prompt (the unconditional branch of classifier-free guidance).
    holds = [x_a]
    hold_weights = [torch.ones(x_a.shape[0], dtype=torch.float64)]
    if retain_prompts:
        retain = _f64(encoder.concept_sequences(retain_prompts)[0]).reshape(-1, d)
        holds.append(retain)
        hold_weights.append(torch.ones(retain.shape[0], dtype=torch.float64))
    null = _f64(encoder.concept_sequences([""])[0]).reshape(-1, d)
    holds.append(null)
    hold_weights.append(NULL_SCALE * torch.ones(null.shape[0], dtype=torch.float64))

    D, N, n0 = solve_map(x_e, x_a, torch.cat(holds), torch.cat(hold_weights), _f64(K0), RANK,
                         HOLD_SCALE, PRESERVE_GLOBAL_SCALE, LAMB, VARIANT)
    N, n0 = ALPHA * N, ALPHA * n0

    for linear in kv_projections(gs_model, gl_model, mode):
        new_weight = linear.weight.data.detach().clone()
        new_bias = linear.bias.data.detach().clone()
        for rows in kv_halves(linear, EDIT_WK, EDIT_WV):
            W, b = fold(_f64(linear.weight.data[rows]), _f64(linear.bias.data[rows]), D, N, n0)
            new_weight[rows] = W.to(device=new_weight.device, dtype=new_weight.dtype)
            new_bias[rows] = b.to(device=new_bias.device, dtype=new_bias.dtype)
        with torch.no_grad():
            linear.weight.copy_(new_weight)
            linear.bias.copy_(new_bias)

# The cross-attention text projections an edit is folded into.
from typing import Literal

import torch.nn as nn

# S edits the structural stage G_S, L the appearance stage G_L, SL both.
Mode = Literal["S", "L", "SL"]
MODES = ("S", "L", "SL")


def kv_projections(gs_model: nn.Module, gl_model: nn.Module, mode: Mode) -> list[nn.Linear]:
    # The fused key/value projection (to_kv) of every cross-attention layer in the edited stages.
    models = {"S": [gs_model], "L": [gl_model], "SL": [gs_model, gl_model]}[mode]
    layers = [module.to_kv for model in models for module in model.modules()
              if getattr(module, "_type", None) == "cross" and isinstance(getattr(module, "to_kv", None), nn.Linear)]
    if not layers:
        raise RuntimeError("no cross-attention to_kv projections found")
    return layers


def kv_halves(linear: nn.Linear, edit_k: bool, edit_v: bool) -> list[slice]:
    # Output rows of to_kv: the key projection first, then the value projection.
    channels = linear.out_features // 2
    return ([slice(0, channels)] if edit_k else []) + ([slice(channels, 2 * channels)] if edit_v else [])

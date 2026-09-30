# STAGE hyperparameters. Defaults are the configuration reported in Table 1. Every value can
# be overridden through the environment variable named next to it, which is how the
# hyperparameter sweeps set them per SLURM job.
import os

# r: maximum dimension of the contrast subspace D.
RANK = int(os.environ.get("STAGE_RANK", 16))
# beta: weight of the retained states (anchors, null prompt, retain prompts).
HOLD_SCALE = float(os.environ.get("STAGE_HOLD_SCALE", 1.0))
# Null-prompt multiplier: extra weight on the empty prompt's states, whose drift
# classifier-free guidance amplifies.
NULL_SCALE = float(os.environ.get("STAGE_NULL_SCALE", 5.0))
# gamma: weight of the K0 term restricted to span(D).
PRESERVE_GLOBAL_SCALE = float(os.environ.get("STAGE_PRESERVE_GLOBAL_SCALE", 1.0))
# lambda: ridge on (N, n0).
LAMB = float(os.environ.get("STAGE_LAMB", 1.0))
# alpha: overshoot of the folded correction, T_alpha(c) = c + alpha (N D^T c + n0).
ALPHA = float(os.environ.get("STAGE_ALPHA", 1.2))
# Pair every aligned token position of the erase/anchor prompts; "0" pairs only the last
# content token of each prompt (Table 10, last-token pairs).
TOKEN_PAIRS = os.environ.get("STAGE_TOKEN_PAIRS", "1") != "0"

# Component ablation (Table 10): "full" is STAGE, "nosubspace" fits the affine map on the
# whole embedding space (D = I), "linear" drops the offset n0, and "rotation" replaces the
# least-squares map with an orthogonal Procrustes rotation inside span(D).
VARIANT = os.environ.get("STAGE_VARIANT", "full")
if VARIANT not in ("full", "nosubspace", "linear", "rotation"):
    raise ValueError(f"unknown STAGE_VARIANT {VARIANT!r}")

# Which halves of the fused key/value projection receive the edit (Table 13).
EDIT_WK = os.environ.get("STAGE_EDIT_WK", "1") != "0"
EDIT_WV = os.environ.get("STAGE_EDIT_WV", "1") != "0"

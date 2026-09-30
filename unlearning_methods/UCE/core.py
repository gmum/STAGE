# UCE (Gandikota et al., 2024): closed-form least-squares edit of one projection W,
#
#   min_W'  sum_i ||W' c_i - W c*||^2 + beta sum_{c in P} ||(W' - W) c||^2
#           + gamma_g tr((W' - W) K0 (W' - W)^T) + lambda ||W' - W||_F^2,
#
# with c_i the erase embeddings, c* the mean anchor embedding and P the optional preserve
# prompts. With S_P = sum_{c in P} c c^T, setting the gradient to zero gives
#   W' = W (c* sum_i c_i^T + beta S_P + gamma_g K0 + lambda I) (sum_i c_i c_i^T + beta S_P + gamma_g K0 + lambda I)^-1.
import torch

from .parameters import LAMB, PRESERVE_CONCEPT_SCALE, PRESERVE_GLOBAL_SCALE


def edit_weight(W: torch.Tensor, erase: torch.Tensor, anchor: torch.Tensor, preserve: torch.Tensor | None,
                K0: torch.Tensor) -> torch.Tensor:
    Wc = W.detach().to(dtype=torch.float64)
    f64 = dict(dtype=torch.float64, device=Wc.device)
    C_e = erase.to(**f64)
    eye = torch.eye(Wc.shape[1], **f64)

    lhs = LAMB * eye
    rhs = torch.outer(anchor.to(**f64).mean(dim=0), C_e.sum(dim=0))
    if preserve is not None:
        C_p = preserve.to(**f64)
        S_p = C_p.T @ C_p
        lhs += PRESERVE_CONCEPT_SCALE * S_p
        rhs += PRESERVE_CONCEPT_SCALE * S_p
    K0 = K0.to(**f64)
    lhs += PRESERVE_GLOBAL_SCALE * K0
    rhs += PRESERVE_GLOBAL_SCALE * K0

    W_new = Wc @ (rhs + LAMB * eye) @ torch.linalg.inv(C_e.T @ C_e + lhs)
    return W_new.to(dtype=W.dtype, device=W.device)

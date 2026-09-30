# OCE (Sun et al., 2026): the projection W is rotated, W' = R W, with R the rotation closest
# to the objective matrix
#
#   M = -s_e G (I - G*) + lambda_r W C_P C_P^T W^T + gamma_g W K0 W^T + lambda W W^T,
#
# where G and G* are the orthogonal projectors onto the spans of the erase images {W c_i}
# and the anchor images {W c*_i}, and C_P holds the optional retain prompts. R = U V^T from
# the SVD M = U S V^T, with the last column of U flipped when needed so that det R = 1.
import torch

from .parameters import ERASE_SCALE, LAMB, PRESERVE_CONCEPT_SCALE, PRESERVE_GLOBAL_SCALE


def _image_projector(W: torch.Tensor, embs: torch.Tensor) -> torch.Tensor:
    V = W @ embs.to(dtype=W.dtype, device=W.device).T
    V = V / (V.norm(dim=0, keepdim=True) + 1e-8)
    Q, _ = torch.linalg.qr(V, mode="reduced")
    return Q @ Q.T


def edit_weight(W: torch.Tensor, erase: torch.Tensor, anchor: torch.Tensor, preserve: torch.Tensor | None,
                K0: torch.Tensor) -> torch.Tensor:
    Wc = W.detach().to(dtype=torch.float64)
    G = _image_projector(Wc, erase)
    G_star = _image_projector(Wc, anchor)
    eye = torch.eye(Wc.shape[0], dtype=Wc.dtype, device=Wc.device)

    M = -ERASE_SCALE * (G @ (eye - G_star))
    if preserve is not None:
        V = Wc @ preserve.to(dtype=Wc.dtype, device=Wc.device).T
        M += PRESERVE_CONCEPT_SCALE * (V @ V.T)
    M += PRESERVE_GLOBAL_SCALE * (Wc @ K0.to(dtype=Wc.dtype, device=Wc.device) @ Wc.T)
    M += LAMB * (Wc @ Wc.T)

    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    if torch.det(U @ Vh) < 0:
        U = U.clone()
        U[:, -1] *= -1
    R = U @ Vh
    return (R @ Wc).to(dtype=W.dtype, device=W.device)

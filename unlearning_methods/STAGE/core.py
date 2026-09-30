# STAGE: an affine correction of the text embedding,
#
#     T(c) = c + N (D^T c) + n0,
#
# confined to the subspace span(D) of the paired erase-minus-anchor token states. (N, n0)
# solve the ridge least squares of equation (1): every erase state maps onto its anchor twin,
# the retained states (anchors, the null prompt, optional retain prompts) stay where they are,
# and a K0 term restricted to span(D) keeps the correction off high-variance caption directions.
# T folds exactly into a projection (W, b): W' = W + (W N) D^T, b' = b + W n0.
import torch

Tensor = torch.Tensor


def build_pairs(erase_hidden: Tensor, erase_mask: Tensor, anchor_hidden: Tensor, anchor_mask: Tensor,
                token_pairs: bool, tol: float = 1e-4) -> tuple[Tensor, Tensor]:
    # Paired (erase state, anchor state) rows of line-paired, template-parallel prompts.
    # token_pairs=False pairs only the last content token of each prompt. Otherwise prompts
    # with the same token count pair all 77 positions, and prompts whose counts differ pair the
    # positions from the later of the two last content tokens onward, where both states encode
    # the full prompt. Positions before the concept word are identical under the causal
    # encoder, so pairs whose states coincide are dropped.
    n = erase_hidden.shape[0]
    if anchor_hidden.shape[0] != n:
        raise ValueError("erase and anchor prompts must be line-paired")
    last_e = erase_mask.sum(dim=1) - 2
    last_a = anchor_mask.sum(dim=1) - 2
    if not token_pairs:
        return erase_hidden[torch.arange(n), last_e], anchor_hidden[torch.arange(n), last_a]

    rows_e, rows_a = [], []
    seq_len = erase_hidden.shape[1]
    for i in range(n):
        if int(erase_mask[i].sum()) == int(anchor_mask[i].sum()):
            sel = torch.arange(seq_len)
        else:
            sel = torch.arange(int(max(last_e[i], last_a[i])), seq_len)
        rows_e.append(erase_hidden[i, sel])
        rows_a.append(anchor_hidden[i, sel])
    x_e = torch.cat(rows_e, dim=0)
    x_a = torch.cat(rows_a, dim=0)
    keep = (x_e - x_a).norm(dim=1) > tol * x_e.norm(dim=1).clamp(min=1e-8)
    return x_e[keep], x_a[keep]


def contrast_basis(deltas: Tensor, rank: int, sv_tol: float = 1e-3) -> Tensor:
    # D: the leading right singular vectors of the paired differences, at most `rank` of them
    # and only those with singular value above sv_tol times the largest.
    _, S, Vh = torch.linalg.svd(deltas, full_matrices=False)
    k = max(1, min(rank, int((S > sv_tol * S[0]).sum())))
    return Vh[:k].T.contiguous()


def solve_map(x_e: Tensor, x_a: Tensor, holds: Tensor, hold_weights: Tensor, K0: Tensor, rank: int,
              hold_scale: float, k0_scale: float, lamb: float, variant: str) -> tuple[Tensor, Tensor, Tensor]:
    # Returns D [d, k], N [d, k] and n0 [d]. `variant` selects a component ablation
    # (see parameters.VARIANT); "full" is STAGE.
    deltas = x_e - x_a
    d = x_e.shape[1]
    D = torch.eye(d, dtype=x_e.dtype) if variant == "nosubspace" else contrast_basis(deltas, rank)
    k = D.shape[1]

    if variant == "rotation":
        # Orthogonal Procrustes inside span(D): the rotation Q best aligning D^T x_e with
        # D^T x_a, applied as T(c) = c + D (Q - I)^T D^T c.
        U, _, Vh = torch.linalg.svd((x_e @ D).T @ (x_a @ D))
        return D, D @ (U @ Vh - torch.eye(k, dtype=x_e.dtype)).T, torch.zeros(d, dtype=x_e.dtype)

    # Least squares over the features phi(c) = [D^T c; 1]; row weights enter as square roots.
    phi_e = torch.cat([x_e @ D, torch.ones(x_e.shape[0], 1, dtype=x_e.dtype)], dim=1)
    phi_h = torch.cat([holds @ D, torch.ones(holds.shape[0], 1, dtype=holds.dtype)], dim=1)
    phi_h = phi_h * hold_weights.sqrt().unsqueeze(1)
    A = phi_e.T @ phi_e + hold_scale * (phi_h.T @ phi_h)
    A[:k, :k] += k0_scale * (D.T @ K0 @ D)
    A += lamb * torch.eye(k + 1, dtype=x_e.dtype)
    B = -deltas.T @ phi_e  # T(x_e) - x_e should equal x_a - x_e

    if variant == "linear":
        N = torch.linalg.solve(A[:k, :k], B[:, :k].T).T
        return D, N, torch.zeros(d, dtype=x_e.dtype)
    M = torch.linalg.solve(A, B.T).T
    return D, M[:, :k], M[:, k]


def fold(W: Tensor, b: Tensor, D: Tensor, N: Tensor, n0: Tensor) -> tuple[Tensor, Tensor]:
    # W T(c) + b = (W + (W N) D^T) c + (b + W n0).
    return W + (W @ N) @ D.T, b + W @ n0

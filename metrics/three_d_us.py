# The 3D Unlearning Score (Section 4) computed from per-asset detections and alignments.
#
# For concept c and evaluator e:
#   E_hat_{e,c}  chance-adjusted percentage of c's assets in which e detects c,
#   F_{e,c}      = 1 - E_hat^post_{e,c},
#   P_{e,c}      = min(1, mean over the other concepts c' of A_hat^post_{e,c'} / A_hat^pre_{e,c'})  (eq. 3),
# F_c and P_c are geometric means over the axis's evaluators, and 3D-US_c is their harmonic
# mean (eq. 4). Axis scores average 3D-US_c over the axis's concepts and the overall score is
# the geometric mean of the three axis scores (metrics/aggregate_results.py).
import numpy as np

EPS = 1e-6
TAU = 0.5  # minimum E_hat^pre a concept needs under every evaluator to enter the evaluation


def chance_correct(rate_pct: float, chance_pct: float) -> float:
    return float(np.clip((rate_pct - chance_pct) / (100.0 - chance_pct), 0.0, 1.0))


def detection_rate(detected: list[bool], chance_pct: float) -> float:
    return chance_correct(100.0 * sum(detected) / len(detected), chance_pct)


def eligible(e_hat_pre_by_evaluator: dict[str, float], tau: float = TAU) -> bool:
    return all(e >= tau for e in e_hat_pre_by_evaluator.values())


def forgetting(e_hat_post: float) -> float:
    return 1.0 - e_hat_post


def contrast_correct(a: float, a_bar: float) -> float:
    # A_hat: similarity to the asset's own prompt, centered on its mean similarity to the other
    # concepts' labels.
    return float(np.clip((a - a_bar) / max(1.0 - a_bar, EPS), 0.0, 1.0))


def preservation(a_hat_pre: dict[str, float], a_hat_post: dict[str, float]) -> float:
    ratios = [a_hat_post[c] / (a_hat_pre[c] + EPS) for c in a_hat_pre]
    return min(1.0, sum(ratios) / len(ratios))


def geometric_mean(values: list[float]) -> float:
    values = [max(v, 0.0) for v in values]
    if not values or any(v == 0.0 for v in values):
        return 0.0
    return float(np.exp(np.mean(np.log(values))))


def combine_evaluators(per_evaluator: dict[str, float]) -> float:
    return geometric_mean(list(per_evaluator.values()))


def three_d_us(f_c: float, p_c: float) -> float:
    return 100.0 * 2 * f_c * p_c / (f_c + p_c + EPS)

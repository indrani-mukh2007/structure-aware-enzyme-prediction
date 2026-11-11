from dataclasses import dataclass
from typing import Dict
import numpy as np

from config import DEFAULT_CONFIG


@dataclass
class FmaxResult:
    fmax: float
    best_threshold: float
    precision_at_best: float
    recall_at_best: float
    thresholds: np.ndarray
    f_scores: np.ndarray
    precisions: np.ndarray
    recalls: np.ndarray


@dataclass
class BootstrapSummary:
    mean_fmax: float
    std_fmax: float
    ci_95_lower: float
    ci_95_upper: float
    best_threshold: float


def compute_cafa_fmax(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_thresholds: int = DEFAULT_CONFIG.fmax_thresholds
) -> FmaxResult:
    assert y_true.shape == y_prob.shape
    y_true_bin = (y_true > 0.5).astype(np.float32)

    true_counts = y_true_bin.sum(axis=1)
    valid_mask = true_counts > 0
    if not np.any(valid_mask):
        zeros = np.zeros(num_thresholds, dtype=np.float32)
        return FmaxResult(0.0, 0.5, 0.0, 0.0, np.linspace(0.0, 1.0, num_thresholds), zeros, zeros, zeros)

    y_t = y_true_bin[valid_mask]
    y_p = y_prob[valid_mask]
    t_counts = true_counts[valid_mask]

    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)
    precisions = np.zeros(num_thresholds, dtype=np.float32)
    recalls = np.zeros(num_thresholds, dtype=np.float32)
    f_scores = np.zeros(num_thresholds, dtype=np.float32)

    for idx, tau in enumerate(thresholds):
        pred_bin = (y_p >= tau).astype(np.float32)
        tp_per_protein = (pred_bin * y_t).sum(axis=1)
        pred_counts = pred_bin.sum(axis=1)

        rc_tau = float(np.mean(tp_per_protein / t_counts))

        active_pred = pred_counts > 0
        if np.any(active_pred):
            pr_tau = float(np.mean(tp_per_protein[active_pred] / pred_counts[active_pred]))
        else:
            pr_tau = 0.0

        precisions[idx] = pr_tau
        recalls[idx] = rc_tau
        if (pr_tau + rc_tau) > 0:
            f_scores[idx] = (2.0 * pr_tau * rc_tau) / (pr_tau + rc_tau)
        else:
            f_scores[idx] = 0.0

    best_idx = int(np.argmax(f_scores))
    return FmaxResult(
        fmax=float(f_scores[best_idx]),
        best_threshold=float(thresholds[best_idx]),
        precision_at_best=float(precisions[best_idx]),
        recall_at_best=float(recalls[best_idx]),
        thresholds=thresholds,
        f_scores=f_scores,
        precisions=precisions,
        recalls=recalls
    )


def compute_per_protein_f1_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float
) -> np.ndarray:
    y_t = (y_true > 0.5).astype(np.float32)
    y_p = (y_prob >= threshold).astype(np.float32)

    tp = (y_p * y_t).sum(axis=1)
    pred_cnt = y_p.sum(axis=1)
    true_cnt = y_t.sum(axis=1)

    denom = pred_cnt + true_cnt
    f1 = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1e-8), 0.0)
    return f1.astype(np.float32)


def bootstrap_fmax_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = DEFAULT_CONFIG.bootstrap_iterations,
    num_thresholds: int = DEFAULT_CONFIG.fmax_thresholds,
    seed: int = 42
) -> BootstrapSummary:
    base_res = compute_cafa_fmax(y_true, y_prob, num_thresholds=num_thresholds)
    rng = np.random.default_rng(seed)
    n_samples = y_true.shape[0]

    boot_fmax = np.zeros(n_bootstrap, dtype=np.float32)
    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)
    y_t = (y_true > 0.5).astype(np.float32)
    t_counts = np.maximum(y_t.sum(axis=1, keepdims=True), 1.0)

    tp_mat = np.zeros((n_samples, num_thresholds), dtype=np.float32)
    pred_cnt_mat = np.zeros((n_samples, num_thresholds), dtype=np.float32)
    for t_idx, tau in enumerate(thresholds):
        p_bin = (y_prob >= tau).astype(np.float32)
        tp_mat[:, t_idx] = (p_bin * y_t).sum(axis=1)
        pred_cnt_mat[:, t_idx] = p_bin.sum(axis=1)

    rc_mat = tp_mat / t_counts
    active_mat = pred_cnt_mat > 0
    pr_mat = np.where(active_mat, tp_mat / np.maximum(pred_cnt_mat, 1.0), 0.0)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)
        rc_b = rc_mat[idx].mean(axis=0)
        act_cnt_b = active_mat[idx].sum(axis=0)
        pr_sum_b = pr_mat[idx].sum(axis=0)
        pr_b = np.where(act_cnt_b > 0, pr_sum_b / np.maximum(act_cnt_b, 1), 0.0)
        denom_b = pr_b + rc_b
        f_b = np.where(denom_b > 0, (2.0 * pr_b * rc_b) / np.maximum(denom_b, 1e-8), 0.0)
        boot_fmax[b] = np.max(f_b)

    ci_low, ci_high = np.percentile(boot_fmax, [2.5, 97.5])
    return BootstrapSummary(
        mean_fmax=float(base_res.fmax),
        std_fmax=float(np.std(boot_fmax, ddof=1)),
        ci_95_lower=float(ci_low),
        ci_95_upper=float(ci_high),
        best_threshold=float(base_res.best_threshold)
    )


def paired_bootstrap_fmax_difference_test(
    y_true: np.ndarray,
    y_prob_model_a: np.ndarray,
    y_prob_model_b: np.ndarray,
    n_bootstrap: int = DEFAULT_CONFIG.bootstrap_iterations,
    num_thresholds: int = DEFAULT_CONFIG.fmax_thresholds,
    seed: int = 42
) -> Dict[str, float]:
    res_a = compute_cafa_fmax(y_true, y_prob_model_a, num_thresholds)
    res_b = compute_cafa_fmax(y_true, y_prob_model_b, num_thresholds)
    observed_delta = res_a.fmax - res_b.fmax

    rng = np.random.default_rng(seed)
    n_samples = y_true.shape[0]
    thresholds = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float32)
    y_t = (y_true > 0.5).astype(np.float32)
    t_counts = np.maximum(y_t.sum(axis=1, keepdims=True), 1.0)

    def _precompute_matrices(y_p: np.ndarray):
        tp_m = np.zeros((n_samples, num_thresholds), dtype=np.float32)
        pc_m = np.zeros((n_samples, num_thresholds), dtype=np.float32)
        for t_idx, tau in enumerate(thresholds):
            pb = (y_p >= tau).astype(np.float32)
            tp_m[:, t_idx] = (pb * y_t).sum(axis=1)
            pc_m[:, t_idx] = pb.sum(axis=1)
        rc_m = tp_m / t_counts
        act_m = pc_m > 0
        pr_m = np.where(act_m, tp_m / np.maximum(pc_m, 1.0), 0.0)
        return rc_m, pr_m, act_m

    rc_a, pr_a, act_a = _precompute_matrices(y_prob_model_a)
    rc_b, pr_b, act_b = _precompute_matrices(y_prob_model_b)

    deltas = np.zeros(n_bootstrap, dtype=np.float32)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_samples, size=n_samples)

        rc_ai = rc_a[idx].mean(axis=0)
        cnt_ai = act_a[idx].sum(axis=0)
        pr_ai = np.where(cnt_ai > 0, pr_a[idx].sum(axis=0) / np.maximum(cnt_ai, 1), 0.0)
        f_ai = np.where((pr_ai + rc_ai) > 0, 2.0 * pr_ai * rc_ai / np.maximum(pr_ai + rc_ai, 1e-8), 0.0)

        rc_bi = rc_b[idx].mean(axis=0)
        cnt_bi = act_b[idx].sum(axis=0)
        pr_bi = np.where(cnt_bi > 0, pr_b[idx].sum(axis=0) / np.maximum(cnt_bi, 1), 0.0)
        f_bi = np.where((pr_bi + rc_bi) > 0, 2.0 * pr_bi * rc_bi / np.maximum(pr_bi + rc_bi, 1e-8), 0.0)

        deltas[i] = np.max(f_ai) - np.max(f_bi)

    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    p_val_one_sided = float(np.mean(deltas <= 0.0))

    return {
        "fmax_a": float(res_a.fmax),
        "fmax_b": float(res_b.fmax),
        "observed_delta": float(observed_delta),
        "delta_std": float(np.std(deltas, ddof=1)),
        "delta_ci_95_lower": float(ci_low),
        "delta_ci_95_upper": float(ci_high),
        "p_value_one_sided": p_val_one_sided
    }

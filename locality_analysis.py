from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

from config import DEFAULT_CONFIG
from evaluation import compute_cafa_fmax, paired_bootstrap_fmax_difference_test


@dataclass
class LocalitySubsetReport:
    subset_name: str
    n_proteins: int
    fmax_esm2_alone: float
    fmax_esm2_structure: float
    fmax_gain: float
    delta_ci_95_lower: float
    delta_ci_95_upper: float
    p_value: float


def evaluate_active_site_locality_subsets(
    y_true: np.ndarray,
    y_prob_esm2_alone: np.ndarray,
    y_prob_esm2_structure: np.ndarray,
    is_non_local_mask: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42
) -> Dict[str, LocalitySubsetReport]:
    non_local_mask = is_non_local_mask.astype(bool)
    local_mask = ~non_local_mask

    subsets = [
        ("Non-local active sites", non_local_mask),
        ("Local active sites", local_mask),
        ("All test enzymes", np.ones_like(non_local_mask, dtype=bool))
    ]

    reports: Dict[str, LocalitySubsetReport] = {}
    for name, mask in subsets:
        n_sub = int(np.sum(mask))
        if n_sub == 0:
            continue
        stats = paired_bootstrap_fmax_difference_test(
            y_true=y_true[mask],
            y_prob_model_a=y_prob_esm2_structure[mask],
            y_prob_model_b=y_prob_esm2_alone[mask],
            n_bootstrap=n_bootstrap,
            seed=seed
        )
        reports[name] = LocalitySubsetReport(
            subset_name=name,
            n_proteins=n_sub,
            fmax_esm2_alone=stats["fmax_b"],
            fmax_esm2_structure=stats["fmax_a"],
            fmax_gain=stats["observed_delta"],
            delta_ci_95_lower=stats["delta_ci_95_lower"],
            delta_ci_95_upper=stats["delta_ci_95_upper"],
            p_value=stats["p_value_one_sided"]
        )

    return reports


def run_stratified_length_matching_control(
    y_true: np.ndarray,
    y_prob_esm2_alone: np.ndarray,
    y_prob_esm2_structure: np.ndarray,
    sequence_lengths: np.ndarray,
    is_non_local_mask: np.ndarray,
    bin_size: int = DEFAULT_CONFIG.length_matching_bin_size,
    n_repeats: int = 100,
    seed: int = 42
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    non_local_idx = np.where(is_non_local_mask)[0]
    local_idx = np.where(~is_non_local_mask)[0]

    bins = sequence_lengths // bin_size
    unique_bins = np.unique(bins[non_local_idx])

    matched_bins = []
    for b in unique_bins:
        nl_in_bin = non_local_idx[bins[non_local_idx] == b]
        loc_in_bin = local_idx[bins[local_idx] == b]
        if len(nl_in_bin) == 0 or len(loc_in_bin) == 0:
            continue
        k_match = min(len(nl_in_bin), len(loc_in_bin))
        matched_bins.append((nl_in_bin, loc_in_bin, k_match))

    nl_base_all = np.concatenate([nl_b[:k_m] for nl_b, _, k_m in matched_bins])
    base_nl_gain = (
        compute_cafa_fmax(y_true[nl_base_all], y_prob_esm2_structure[nl_base_all]).fmax
        - compute_cafa_fmax(y_true[nl_base_all], y_prob_esm2_alone[nl_base_all]).fmax
    )

    nl_boot_gains = []
    loc_matched_gains = []
    matched_counts = len(nl_base_all)

    for _ in range(n_repeats):
        nl_sel = []
        loc_sel = []
        for nl_in_bin, loc_in_bin, k_match in matched_bins:
            nl_choice = rng.choice(nl_in_bin, size=k_match, replace=True)
            loc_choice = rng.choice(loc_in_bin, size=k_match, replace=False)
            nl_sel.append(nl_choice)
            loc_sel.append(loc_choice)

        nl_all = np.concatenate(nl_sel)
        loc_all = np.concatenate(loc_sel)

        fmax_nl_struct = compute_cafa_fmax(y_true[nl_all], y_prob_esm2_structure[nl_all]).fmax
        fmax_nl_seq = compute_cafa_fmax(y_true[nl_all], y_prob_esm2_alone[nl_all]).fmax
        nl_boot_gains.append(fmax_nl_struct - fmax_nl_seq)

        fmax_loc_struct = compute_cafa_fmax(y_true[loc_all], y_prob_esm2_structure[loc_all]).fmax
        fmax_loc_seq = compute_cafa_fmax(y_true[loc_all], y_prob_esm2_alone[loc_all]).fmax
        loc_matched_gains.append(fmax_loc_struct - fmax_loc_seq)

    return {
        "matched_pairs_per_draw": float(matched_counts),
        "non_local_length_matched_gain_mean": float(base_nl_gain),
        "non_local_length_matched_gain_std": float(np.std(nl_boot_gains, ddof=1)),
        "local_length_matched_gain_mean": float(np.mean(loc_matched_gains)),
        "local_length_matched_gain_std": float(np.std(loc_matched_gains, ddof=1)),
        "net_locality_advantage_after_length_matching": float(base_nl_gain - np.mean(loc_matched_gains))
    }


def run_sequence_separation_threshold_sweep(
    y_true: np.ndarray,
    y_prob_esm2_alone: np.ndarray,
    y_prob_esm2_structure: np.ndarray,
    max_catalytic_seq_seps: np.ndarray,
    sweep_thresholds: Tuple[int, ...] = DEFAULT_CONFIG.non_local_sweep_thresholds
) -> List[Dict[str, float]]:
    sweep_results = []
    for k in sweep_thresholds:
        nl_mask = max_catalytic_seq_seps >= k
        loc_mask = ~nl_mask

        n_nl = int(np.sum(nl_mask))
        n_loc = int(np.sum(loc_mask))

        if n_nl > 0:
            f_nl_struct = compute_cafa_fmax(y_true[nl_mask], y_prob_esm2_structure[nl_mask]).fmax
            f_nl_seq = compute_cafa_fmax(y_true[nl_mask], y_prob_esm2_alone[nl_mask]).fmax
            gain_nl = f_nl_struct - f_nl_seq
        else:
            f_nl_struct, f_nl_seq, gain_nl = 0.0, 0.0, 0.0

        if n_loc > 0:
            f_loc_struct = compute_cafa_fmax(y_true[loc_mask], y_prob_esm2_structure[loc_mask]).fmax
            f_loc_seq = compute_cafa_fmax(y_true[loc_mask], y_prob_esm2_alone[loc_mask]).fmax
            gain_loc = f_loc_struct - f_loc_seq
        else:
            f_loc_struct, f_loc_seq, gain_loc = 0.0, 0.0, 0.0

        sweep_results.append({
            "seq_sep_threshold_k": int(k),
            "n_non_local": n_nl,
            "n_local": n_loc,
            "fmax_non_local_seq": float(f_nl_seq),
            "fmax_non_local_struct": float(f_nl_struct),
            "fmax_gain_non_local": float(gain_nl),
            "fmax_local_seq": float(f_loc_seq),
            "fmax_local_struct": float(f_loc_struct),
            "fmax_gain_local": float(gain_loc)
        })

    return sweep_results

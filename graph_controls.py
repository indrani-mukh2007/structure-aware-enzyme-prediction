from typing import Optional, Tuple
import numpy as np

from config import ExperimentConfig, DEFAULT_CONFIG
from data_pipeline import compute_edge_features


def degree_preserving_shuffle_graph(
    edge_index: np.ndarray,
    num_nodes: int,
    dist_matrix: np.ndarray,
    num_swaps_multiplier: int = DEFAULT_CONFIG.rewiring_swap_multiplier,
    seed: int = 42,
    cfg: ExperimentConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] == 0:
        return edge_index.copy(), np.zeros((0, cfg.edge_rbf_kernels + cfg.edge_seq_pos_dim), dtype=np.float32)

    rng = np.random.default_rng(seed)

    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    undirected_edges = np.stack([src[mask], dst[mask]], axis=1)
    num_undirected = len(undirected_edges)

    if num_undirected < 2:
        return edge_index.copy(), compute_edge_features(edge_index, dist_matrix, cfg)

    adj_set = set(map(tuple, undirected_edges.tolist()))
    num_attempts = num_undirected * num_swaps_multiplier

    for _ in range(num_attempts):
        idx1, idx2 = rng.choice(num_undirected, size=2, replace=False)
        u, v = int(undirected_edges[idx1, 0]), int(undirected_edges[idx1, 1])
        x, y = int(undirected_edges[idx2, 0]), int(undirected_edges[idx2, 1])

        if len({u, v, x, y}) < 4:
            continue

        if rng.random() < 0.5:
            e1_new = (min(u, y), max(u, y))
            e2_new = (min(x, v), max(x, v))
        else:
            e1_new = (min(u, x), max(u, x))
            e2_new = (min(v, y), max(v, y))

        if e1_new[0] == e1_new[1] or e2_new[0] == e2_new[1]:
            continue
        if e1_new in adj_set or e2_new in adj_set or e1_new == e2_new:
            continue

        adj_set.remove((u, v))
        adj_set.remove((x, y))
        adj_set.add(e1_new)
        adj_set.add(e2_new)

        undirected_edges[idx1] = e1_new
        undirected_edges[idx2] = e2_new

    u_arr = undirected_edges[:, 0]
    v_arr = undirected_edges[:, 1]
    shuffled_src = np.concatenate([u_arr, v_arr])
    shuffled_dst = np.concatenate([v_arr, u_arr])
    shuffled_edge_index = np.stack([shuffled_src, shuffled_dst], axis=0).astype(np.int64)

    upper_rand = rng.uniform(3.8, cfg.ca_distance_cutoff_angstrom, size=dist_matrix.shape).astype(np.float32)
    sym_fake_dist = np.triu(upper_rand, k=1)
    sym_fake_dist = sym_fake_dist + sym_fake_dist.T
    shuffled_edge_attr = compute_edge_features(shuffled_edge_index, sym_fake_dist, cfg)

    return shuffled_edge_index, shuffled_edge_attr


def build_sequence_window_graph(
    num_nodes: int,
    plddt_scores: np.ndarray,
    target_edge_count: Optional[int] = None,
    default_window_k: int = DEFAULT_CONFIG.default_seq_window_k,
    seed: int = 42,
    cfg: ExperimentConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, np.ndarray]:
    high_conf = plddt_scores >= cfg.plddt_confidence_threshold
    rng = np.random.default_rng(seed)

    undirected_pairs = []
    if target_edge_count is not None and num_nodes > 1:
        target_undirected = max(0, target_edge_count // 2)
        for step in range(1, num_nodes):
            if len(undirected_pairs) >= target_undirected:
                break
            shell_pairs = [
                (i, i + step)
                for i in range(num_nodes - step)
                if high_conf[i] and high_conf[i + step]
            ]
            if not shell_pairs:
                continue
            remaining = target_undirected - len(undirected_pairs)
            if len(shell_pairs) <= remaining:
                undirected_pairs.extend(shell_pairs)
            else:
                chosen_idx = rng.choice(len(shell_pairs), size=remaining, replace=False)
                for c_idx in sorted(chosen_idx.tolist()):
                    undirected_pairs.append(shell_pairs[c_idx])
                break
    else:
        for i in range(num_nodes):
            if not high_conf[i]:
                continue
            for offset in range(1, default_window_k + 1):
                j = i + offset
                if j < num_nodes and high_conf[j]:
                    undirected_pairs.append((i, j))

    if not undirected_pairs:
        empty_idx = np.zeros((2, 0), dtype=np.int64)
        empty_attr = np.zeros((0, cfg.edge_rbf_kernels + cfg.edge_seq_pos_dim), dtype=np.float32)
        return empty_idx, empty_attr

    pairs_arr = np.asarray(undirected_pairs, dtype=np.int64)
    u_arr, v_arr = pairs_arr[:, 0], pairs_arr[:, 1]
    seq_src = np.concatenate([u_arr, v_arr])
    seq_dst = np.concatenate([v_arr, u_arr])
    seq_edge_index = np.stack([seq_src, seq_dst], axis=0).astype(np.int64)

    idx_grid = np.arange(num_nodes)
    seq_dist_matrix = np.minimum(
        cfg.ca_distance_cutoff_angstrom - 0.1,
        np.abs(idx_grid[:, None] - idx_grid[None, :]) * 1.8 + 2.0
    ).astype(np.float32)
    np.fill_diagonal(seq_dist_matrix, 0.0)

    seq_edge_attr = compute_edge_features(seq_edge_index, seq_dist_matrix, cfg)
    return seq_edge_index, seq_edge_attr

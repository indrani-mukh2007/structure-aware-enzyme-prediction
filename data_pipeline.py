from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist

from config import ExperimentConfig, DEFAULT_CONFIG


@dataclass
class EnzymeGraphRecord:
    uniprot_id: str
    cluster_id: str
    sequence_length: int
    esm2_embeddings: np.ndarray
    ca_coords: np.ndarray
    plddt_scores: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    ec_labels: np.ndarray
    catalytic_residues: List[int]
    max_catalytic_seq_sep: int
    is_non_local_active_site: bool


AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}


def parse_alphafold_pdb_ca(pdb_filepath: str) -> Tuple[np.ndarray, np.ndarray, str]:
    coords, plddts, seq_chars = [], [], []
    with open(pdb_filepath, "r") as f:
        for line in f:
            if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                res_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                b_factor = float(line[60:66].strip())
                coords.append([x, y, z])
                plddts.append(b_factor)
                seq_chars.append(AA_MAP.get(res_name, "X"))

    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(plddts, dtype=np.float32),
        "".join(seq_chars)
    )


def compute_edge_features(
    edge_index: np.ndarray,
    dist_matrix: np.ndarray,
    cfg: ExperimentConfig = DEFAULT_CONFIG
) -> np.ndarray:
    if edge_index.shape[1] == 0:
        return np.zeros((0, cfg.edge_rbf_kernels + cfg.edge_seq_pos_dim), dtype=np.float32)

    src, dst = edge_index[0], edge_index[1]
    distances = dist_matrix[src, dst]

    centers = np.linspace(0.0, cfg.ca_distance_cutoff_angstrom, cfg.edge_rbf_kernels, dtype=np.float32)
    step = (centers[1] - centers[0]) if cfg.edge_rbf_kernels > 1 else cfg.ca_distance_cutoff_angstrom
    gamma = 1.0 / (step ** 2 + 1e-8)
    rbf_feats = np.exp(-gamma * (distances[:, None] - centers[None, :]) ** 2)

    seq_sep = np.clip(src - dst, -32, 32).astype(np.float32)
    div_term = np.exp(
        np.arange(0, cfg.edge_seq_pos_dim, 2, dtype=np.float32) * -(np.log(100.0) / max(cfg.edge_seq_pos_dim, 1))
    )
    pos_feats = np.zeros((edge_index.shape[1], cfg.edge_seq_pos_dim), dtype=np.float32)
    pos_feats[:, 0::2] = np.sin(seq_sep[:, None] * div_term[None, :])
    pos_feats[:, 1::2] = np.cos(seq_sep[:, None] * div_term[None, :])

    return np.concatenate([rbf_feats, pos_feats], axis=1).astype(np.float32)


def build_confidence_masked_contact_graph(
    ca_coords: np.ndarray,
    plddt_scores: np.ndarray,
    cfg: ExperimentConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist_matrix = cdist(ca_coords, ca_coords, metric="euclidean").astype(np.float32)
    high_conf_mask = plddt_scores >= cfg.plddt_confidence_threshold
    joint_conf_mask = np.outer(high_conf_mask, high_conf_mask)

    adj_matrix = (
        (dist_matrix < cfg.ca_distance_cutoff_angstrom)
        & joint_conf_mask
        & (~np.eye(len(plddt_scores), dtype=bool))
    )

    src, dst = np.nonzero(adj_matrix)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)
    edge_attr = compute_edge_features(edge_index, dist_matrix, cfg)
    return edge_index, edge_attr, dist_matrix


def compute_active_site_locality(
    catalytic_residues: List[int],
    ca_coords: np.ndarray,
    seq_sep_threshold: int = DEFAULT_CONFIG.non_local_seq_sep_threshold,
    spatial_cluster_cutoff: float = 10.0
) -> Tuple[int, bool]:
    if len(catalytic_residues) < 2:
        return 0, False

    cat_idx = np.asarray(catalytic_residues, dtype=np.int64)
    cat_coords = ca_coords[cat_idx]
    cat_dists = cdist(cat_coords, cat_coords, metric="euclidean")

    max_sep = 0
    num_cat = len(cat_idx)
    for a in range(num_cat):
        for b in range(a + 1, num_cat):
            if cat_dists[a, b] <= spatial_cluster_cutoff:
                sep = abs(int(cat_idx[a]) - int(cat_idx[b]))
                if sep > max_sep:
                    max_sep = sep

    return max_sep, (max_sep >= seq_sep_threshold)


def generate_mmseqs2_cluster_command(fasta_input: str, output_prefix: str, tmp_dir: str = "tmp_mmseqs") -> str:
    return (
        f"mmseqs easy-cluster {fasta_input} {output_prefix} {tmp_dir} "
        f"--min-seq-id 0.3 -c 0.8 --cov-mode 0"
    )


def create_synthetic_enzyme_record(
    uniprot_id: str,
    cluster_id: str,
    seq_len: int = 160,
    force_non_local: Optional[bool] = None,
    embedding_dim: int = 64,
    num_classes: int = 32,
    rng: Optional[np.random.Generator] = None,
    cfg: ExperimentConfig = DEFAULT_CONFIG
) -> EnzymeGraphRecord:
    if rng is None:
        rng = np.random.default_rng(42)

    t = np.linspace(0, 4 * np.pi, seq_len)
    ca_coords = np.stack([
        12.0 * np.cos(t) + rng.normal(0, 1.2, size=seq_len),
        12.0 * np.sin(t) + rng.normal(0, 1.2, size=seq_len),
        2.5 * np.sin(2 * t) + rng.normal(0, 1.2, size=seq_len),
    ], axis=1).astype(np.float32)

    plddt_scores = rng.uniform(76.0, 98.0, size=seq_len).astype(np.float32)
    low_conf_indices = rng.choice(seq_len, size=max(4, int(0.12 * seq_len)), replace=False)
    plddt_scores[low_conf_indices] = rng.uniform(40.0, 68.0, size=len(low_conf_indices)).astype(np.float32)

    high_conf_candidates = np.where(plddt_scores >= cfg.plddt_confidence_threshold)[0]
    anchor = int(high_conf_candidates[len(high_conf_candidates) // 4])

    if force_non_local is True:
        partner = min(seq_len - 2, anchor + int(rng.integers(25, 65)))
        plddt_scores[partner] = 92.0
        ca_coords[partner] = ca_coords[anchor] + rng.uniform(-2.8, 2.8, size=3).astype(np.float32)
        catalytic_residues = sorted([anchor, min(seq_len - 1, anchor + 3), int(partner)])
    elif force_non_local is False:
        partner_1 = min(seq_len - 3, anchor + 4)
        partner_2 = min(seq_len - 2, anchor + 9)
        plddt_scores[[partner_1, partner_2]] = 91.0
        ca_coords[partner_1] = ca_coords[anchor] + np.array([3.2, 1.0, -1.0], dtype=np.float32)
        ca_coords[partner_2] = ca_coords[anchor] + np.array([-2.2, 2.8, 1.2], dtype=np.float32)
        catalytic_residues = sorted([anchor, int(partner_1), int(partner_2)])
    else:
        catalytic_residues = sorted(rng.choice(high_conf_candidates, size=3, replace=False).tolist())

    edge_index, edge_attr, _ = build_confidence_masked_contact_graph(ca_coords, plddt_scores, cfg)
    max_sep, is_non_local = compute_active_site_locality(
        catalytic_residues, ca_coords, seq_sep_threshold=cfg.non_local_seq_sep_threshold
    )

    esm2_embeddings = rng.standard_normal((seq_len, embedding_dim)).astype(np.float32)
    ec_labels = np.zeros(num_classes, dtype=np.float32)
    active_classes = rng.choice(num_classes, size=3, replace=False)
    ec_labels[active_classes] = 1.0

    return EnzymeGraphRecord(
        uniprot_id=uniprot_id,
        cluster_id=cluster_id,
        sequence_length=seq_len,
        esm2_embeddings=esm2_embeddings,
        ca_coords=ca_coords,
        plddt_scores=plddt_scores,
        edge_index=edge_index,
        edge_attr=edge_attr,
        ec_labels=ec_labels,
        catalytic_residues=catalytic_residues,
        max_catalytic_seq_sep=max_sep,
        is_non_local_active_site=is_non_local,
    )

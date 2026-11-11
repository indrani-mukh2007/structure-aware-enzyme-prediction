from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ExperimentConfig:
    total_swissprot_enzymes: int = 12400
    mmseqs2_min_seq_id: float = 0.30
    mmseqs2_coverage: float = 0.80
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    test_non_local_count: int = 273
    test_local_count: int = 967
    non_local_seq_sep_threshold: int = 20
    non_local_sweep_thresholds: Tuple[int, ...] = (10, 15, 20, 30, 50)
    length_matching_bin_size: int = 50

    esm2_model_name: str = "facebook/esm2_t33_650M_UR50D"
    esm2_embedding_dim: int = 1280
    freeze_esm2: bool = True

    ca_distance_cutoff_angstrom: float = 8.0
    plddt_confidence_threshold: float = 70.0
    edge_rbf_kernels: int = 16
    edge_seq_pos_dim: int = 16

    default_seq_window_k: int = 3
    rewiring_swap_multiplier: int = 10

    num_ec_classes: int = 480
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 3
    gnn_num_heads: int = 4
    gnn_dropout: float = 0.20
    classifier_hidden_dim: int = 512

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 50
    early_stopping_patience: int = 8

    fmax_thresholds: int = 101
    random_seeds: Tuple[int, ...] = (42, 1337, 2026)
    bootstrap_iterations: int = 1000


DEFAULT_CONFIG = ExperimentConfig()

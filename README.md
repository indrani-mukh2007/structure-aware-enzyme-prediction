# Structure-Aware Enzyme Function Prediction

Code for evaluating whether 3D residue contact graphs (AlphaFold Cα contacts < 8.0 Å with pLDDT >= 70 masking) on top of frozen ESM-2 (`esm2_t33_650M_UR50D`) embeddings improve enzyme function prediction under a strict 30% sequence-identity split (MMseqs2).

## Overview

We compare:
- **ESM-2 + Structure (GNN)**: 3-layer edge-aware GATv2 over confidence-masked residue contact graphs.
- **DIAMOND Homology**: `blastp` bit-score weighted annotation transfer across the 30% identity split.
- **ESM-2 Alone**: Mean + max pooled frozen residue representations with an identical MLP classifier head.
- **Shuffled-Graph Control**: Degree-preserving Maslov-Sneppen edge rewiring.
- **Sequence-Window Control**: Density-matched local 1D sequence neighbor edges.

## Results Summary

| Model | CAFA Fmax |
|---|---|
| ESM-2 + structure (GNN) | 0.72 |
| DIAMOND homology | 0.71 |
| ESM-2 alone | 0.70 |
| Shuffled-graph control | 0.70 |
| Sequence-window control | 0.70 |

### Active-Site Locality (Test Set, n = 1,240)
- **Non-local active sites** (`|i - j| >= 20`, `n = 273`): **+0.08** Fmax gain from structure.
- **Local active sites** (`|i - j| < 20`, `n = 967`): **+0.01** Fmax gain from structure.

## Usage

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the evaluation and control suite:
```bash
python run_experiments.py
```

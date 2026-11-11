from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

from config import DEFAULT_CONFIG
from graph_controls import degree_preserving_shuffle_graph, build_sequence_window_graph


def global_mean_max_pool(
    node_feats: torch.Tensor,
    batch_index: Optional[torch.Tensor] = None,
    num_graphs: Optional[int] = None
) -> torch.Tensor:
    if node_feats.size(0) == 0:
        out_graphs = num_graphs if num_graphs is not None else 1
        return torch.zeros((out_graphs, 2 * node_feats.size(-1)), device=node_feats.device, dtype=node_feats.dtype)

    if batch_index is None:
        mean_pool = node_feats.mean(dim=0, keepdim=True)
        max_pool = node_feats.max(dim=0, keepdim=True).values
        return torch.cat([mean_pool, max_pool], dim=-1)

    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1

    device = node_feats.device
    dtype = node_feats.dtype
    feat_dim = node_feats.size(-1)

    sum_pool = torch.zeros((num_graphs, feat_dim), device=device, dtype=dtype)
    counts = torch.zeros((num_graphs, 1), device=device, dtype=dtype)
    sum_pool.index_add_(0, batch_index, node_feats)
    ones = torch.ones((node_feats.size(0), 1), device=device, dtype=dtype)
    counts.index_add_(0, batch_index, ones)
    mean_pool = sum_pool / counts.clamp(min=1.0)

    max_pool = torch.full((num_graphs, feat_dim), -1e9, device=device, dtype=dtype)
    if hasattr(max_pool, "scatter_reduce_"):
        idx_expanded = batch_index.unsqueeze(-1).expand(-1, feat_dim)
        max_pool.scatter_reduce_(0, idx_expanded, node_feats, reduce="amax", include_self=False)
        empty_graphs = (counts.squeeze(-1) == 0)
        if empty_graphs.any():
            max_pool[empty_graphs] = 0.0
    else:
        for g in range(num_graphs):
            mask = (batch_index == g)
            if mask.any():
                max_pool[g] = node_feats[mask].max(dim=0).values
            else:
                max_pool[g] = 0.0

    return torch.cat([mean_pool, max_pool], dim=-1)


class MultiLabelECClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = DEFAULT_CONFIG.classifier_hidden_dim,
        num_classes: int = DEFAULT_CONFIG.num_ec_classes,
        dropout: float = DEFAULT_CONFIG.gnn_dropout
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, pooled_repr: torch.Tensor) -> torch.Tensor:
        return self.net(pooled_repr)


class ESM2SequenceBaseline(nn.Module):
    def __init__(
        self,
        esm_dim: int = DEFAULT_CONFIG.esm2_embedding_dim,
        hidden_dim: int = DEFAULT_CONFIG.gnn_hidden_dim,
        classifier_dim: int = DEFAULT_CONFIG.classifier_hidden_dim,
        num_classes: int = DEFAULT_CONFIG.num_ec_classes,
        dropout: float = DEFAULT_CONFIG.gnn_dropout
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = MultiLabelECClassifierHead(
            input_dim=2 * hidden_dim,
            hidden_dim=classifier_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None
    ) -> torch.Tensor:
        h = self.input_proj(x)
        pooled = global_mean_max_pool(h, batch_index=batch_index, num_graphs=num_graphs)
        return self.classifier(pooled)


class EdgeAwareGATv2Layer(nn.Module):
    def __init__(
        self,
        hidden_dim: int = DEFAULT_CONFIG.gnn_hidden_dim,
        edge_dim: int = DEFAULT_CONFIG.edge_rbf_kernels + DEFAULT_CONFIG.edge_seq_pos_dim,
        num_heads: int = DEFAULT_CONFIG.gnn_num_heads,
        dropout: float = DEFAULT_CONFIG.gnn_dropout
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.w_src = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_dst = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_edge = nn.Linear(edge_dim, hidden_dim, bias=False)

        self.attn_vec = nn.Parameter(torch.empty(1, num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        num_nodes = h.size(0)
        if edge_index.numel() == 0:
            return self.norm(h)

        src, dst = edge_index[0], edge_index[1]

        h_src = self.w_src(h).view(num_nodes, self.num_heads, self.head_dim)
        h_dst = self.w_dst(h).view(num_nodes, self.num_heads, self.head_dim)
        e_proj = self.w_edge(edge_attr).view(-1, self.num_heads, self.head_dim)

        edge_feat = F.leaky_relu(h_src[src] + h_dst[dst] + e_proj, negative_slope=0.2)
        attn_logits = (edge_feat * self.attn_vec).sum(dim=-1)

        max_logits = torch.full((num_nodes, self.num_heads), -1e9, device=h.device, dtype=h.dtype)
        if hasattr(max_logits, "scatter_reduce_"):
            dst_expanded = dst.unsqueeze(-1).expand(-1, self.num_heads)
            max_logits.scatter_reduce_(0, dst_expanded, attn_logits, reduce="amax", include_self=False)
        else:
            max_logits.index_copy_(0, dst, attn_logits)

        exp_logits = torch.exp(attn_logits - max_logits[dst].detach())
        denom = torch.zeros((num_nodes, self.num_heads), device=h.device, dtype=h.dtype)
        denom.index_add_(0, dst, exp_logits)
        alpha = exp_logits / (denom[dst] + 1e-9)
        alpha = self.dropout(alpha)

        messages = (h_src[src] + e_proj) * alpha.unsqueeze(-1)
        agg = torch.zeros((num_nodes, self.num_heads, self.head_dim), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, messages)

        agg_flat = agg.reshape(num_nodes, self.hidden_dim)
        out = self.out_proj(F.gelu(agg_flat))
        return self.norm(h + self.dropout(out))


class ESM2StructureGNN(nn.Module):
    def __init__(
        self,
        esm_dim: int = DEFAULT_CONFIG.esm2_embedding_dim,
        hidden_dim: int = DEFAULT_CONFIG.gnn_hidden_dim,
        edge_dim: int = DEFAULT_CONFIG.edge_rbf_kernels + DEFAULT_CONFIG.edge_seq_pos_dim,
        num_layers: int = DEFAULT_CONFIG.gnn_num_layers,
        num_heads: int = DEFAULT_CONFIG.gnn_num_heads,
        classifier_dim: int = DEFAULT_CONFIG.classifier_hidden_dim,
        num_classes: int = DEFAULT_CONFIG.num_ec_classes,
        dropout: float = DEFAULT_CONFIG.gnn_dropout
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.gnn_layers = nn.ModuleList([
            EdgeAwareGATv2Layer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.classifier = MultiLabelECClassifierHead(
            input_dim=2 * hidden_dim,
            hidden_dim=classifier_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
        num_graphs: Optional[int] = None
    ) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.gnn_layers:
            h = layer(h, edge_index, edge_attr)
        pooled = global_mean_max_pool(h, batch_index=batch_index, num_graphs=num_graphs)
        return self.classifier(pooled)


def collate_enzyme_graphs(records: List, graph_mode: str = "native", seed: int = 42) -> dict:
    x_list, edge_index_list, edge_attr_list, y_list, batch_idx_list = [], [], [], [], []
    node_offset = 0

    for g_idx, rec in enumerate(records):
        num_nodes = rec.sequence_length
        x_list.append(torch.from_numpy(rec.esm2_embeddings))
        y_list.append(torch.from_numpy(rec.ec_labels))
        batch_idx_list.append(torch.full((num_nodes,), g_idx, dtype=torch.long))

        if graph_mode == "native":
            ei, ea = rec.edge_index, rec.edge_attr
        elif graph_mode == "shuffled":
            dm = cdist(rec.ca_coords, rec.ca_coords, metric="euclidean").astype(np.float32)
            ei, ea = degree_preserving_shuffle_graph(rec.edge_index, num_nodes, dm, seed=seed + g_idx)
        elif graph_mode == "sequence_window":
            ei, ea = build_sequence_window_graph(
                num_nodes, rec.plddt_scores, target_edge_count=rec.edge_index.shape[1], seed=seed + g_idx
            )
        else:
            raise ValueError(f"Invalid graph_mode: {graph_mode}")

        if ei.shape[1] > 0:
            shifted_edges = torch.from_numpy(ei) + node_offset
            edge_index_list.append(shifted_edges)
            edge_attr_list.append(torch.from_numpy(ea))

        node_offset += num_nodes

    x_batch = torch.cat(x_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    batch_index = torch.cat(batch_idx_list, dim=0)

    if edge_index_list:
        edge_index_batch = torch.cat(edge_index_list, dim=1)
        edge_attr_batch = torch.cat(edge_attr_list, dim=0)
    else:
        edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_batch = torch.zeros((0, 32), dtype=torch.float32)

    return {
        "x": x_batch,
        "edge_index": edge_index_batch,
        "edge_attr": edge_attr_batch,
        "y": y_batch,
        "batch_index": batch_index,
        "num_graphs": len(records)
    }

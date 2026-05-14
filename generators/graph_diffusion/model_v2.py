"""F9 architecture variants: extends GraphDenoiser with MLP-on-pair and
GAT-style edge heads, while keeping the original bilinear head as the
default for backwards compatibility with final_v2.pt.

The design pattern: a single GraphDenoiserV2 class with a config flag
(`edge_head: 'bilinear' | 'mlp' | 'gat'`) that swaps the head module
without touching the backbone. Existing checkpoints load by setting
edge_head='bilinear' (the default).

Usage:
    from generators.graph_diffusion.model_v2 import (
        GraphDenoiserV2, GraphDiffusionConfigV2,
    )
    cfg = GraphDiffusionConfigV2(edge_head='mlp')
    model = GraphDenoiserV2(cfg)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from generators.graph_diffusion.model import (
    cosine_alpha_bar, edge_flip_rate, RasterEncoder, TimeEmbedding,
)


@dataclass
class GraphDiffusionConfigV2:
    n_nodes: int = 200
    coord_dim: int = 2
    embed_dim: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    raster_channels: int = 6
    timesteps: int = 200
    no_raster: bool = False
    edge_head: str = "bilinear"  # 'bilinear' | 'mlp' | 'gat'
    # MLP edge-head depth + width.
    edge_mlp_hidden: int = 256
    edge_mlp_layers: int = 2
    # GAT edge-head: number of attention heads (k=4 in the plan).
    gat_heads: int = 4


class _BilinearEdgeHead(nn.Module):
    """Original final_v2 head: e_ij = (W h_i)^T (W h_j) / sqrt(E)."""

    def __init__(self, E: int):
        super().__init__()
        self.norm = nn.LayerNorm(E)
        self.proj = nn.Linear(E, E)
        self.E = E

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_n = self.norm(h)
        h_p = self.proj(h_n)
        logits = torch.einsum("bne,bme->bnm", h_p, h_p) / math.sqrt(self.E)
        return 0.5 * (logits + logits.transpose(-1, -2))


class _MLPPairEdgeHead(nn.Module):
    """MLP-on-pair head: e_ij = MLP(concat(h_i, h_j) + concat(h_j, h_i)) / 2.

    Strictly more expressive than bilinear: can model non-multiplicative
    interactions between node embeddings. Symmetrised by averaging
    (i,j) and (j,i) inputs through the same network.
    """

    def __init__(self, E: int, hidden: int, n_layers: int):
        super().__init__()
        self.norm = nn.LayerNorm(E)
        layers: list[nn.Module] = []
        in_dim = 2 * E
        for k in range(n_layers):
            out_dim = hidden if k < n_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if k < n_layers - 1:
                layers.append(nn.GELU())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_n = self.norm(h)
        B, N, E = h_n.shape
        # Pairwise concat: (B, N, N, 2E)
        h_i = h_n.unsqueeze(2).expand(B, N, N, E)
        h_j = h_n.unsqueeze(1).expand(B, N, N, E)
        ij = torch.cat([h_i, h_j], dim=-1)
        ji = torch.cat([h_j, h_i], dim=-1)
        # Symmetrise by averaging two passes.
        l_ij = self.mlp(ij).squeeze(-1)
        l_ji = self.mlp(ji).squeeze(-1)
        return 0.5 * (l_ij + l_ji)


class _GATEdgeHead(nn.Module):
    """Single-layer GAT-style head producing pairwise edge logits.

    For each pair (i, j) the logit is

        e_ij = LeakyReLU( a^T [W h_i || W h_j] )

    averaged over k attention heads (Veličković et al. 2018), then
    symmetrised. Adds ~0.6M params for E=256, k=4.
    """

    def __init__(self, E: int, k_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(E)
        self.W = nn.Linear(E, E, bias=False)
        # Per-head attention vector a of shape (2E,)
        self.a = nn.Parameter(torch.randn(k_heads, 2 * E) * 0.02)
        self.k_heads = k_heads
        self.E = E

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_n = self.norm(h)
        Wh = self.W(h_n)                                  # (B, N, E)
        B, N, E = Wh.shape
        h_i = Wh.unsqueeze(2).expand(B, N, N, E)
        h_j = Wh.unsqueeze(1).expand(B, N, N, E)
        cat = torch.cat([h_i, h_j], dim=-1)                # (B, N, N, 2E)
        # Multi-head: (B, N, N, k)
        scores = torch.einsum("bnme,ke->bnmk", cat, self.a)
        scores = F.leaky_relu(scores, negative_slope=0.2)
        logits = scores.mean(dim=-1)                       # (B, N, N)
        return 0.5 * (logits + logits.transpose(-1, -2))


class GraphDenoiserV2(nn.Module):
    """Graph denoiser with swappable edge head.

    Backbone is identical to GraphDenoiser (model.py). Only the edge
    head module differs based on cfg.edge_head.
    """

    def __init__(self, cfg: GraphDiffusionConfigV2):
        super().__init__()
        self.cfg = cfg
        E = cfg.embed_dim

        self.node_in = nn.Linear(cfg.coord_dim + 1, E)
        self.node_pos = nn.Parameter(torch.randn(cfg.n_nodes, E) * 0.02)
        self.time_embed = TimeEmbedding(E)
        self.raster = RasterEncoder(cfg.raster_channels, E)

        layer = nn.TransformerDecoderLayer(
            d_model=E, nhead=cfg.n_heads,
            dim_feedforward=E * 4,
            dropout=cfg.dropout,
            batch_first=True, norm_first=True,
        )
        self.tx = nn.TransformerDecoder(layer, num_layers=cfg.n_layers)

        self.coord_head = nn.Sequential(
            nn.LayerNorm(E), nn.Linear(E, cfg.coord_dim),
        )

        if cfg.edge_head == "bilinear":
            self.edge_head_module = _BilinearEdgeHead(E)
        elif cfg.edge_head == "mlp":
            self.edge_head_module = _MLPPairEdgeHead(
                E, cfg.edge_mlp_hidden, cfg.edge_mlp_layers,
            )
        elif cfg.edge_head == "gat":
            self.edge_head_module = _GATEdgeHead(E, cfg.gat_heads)
        else:
            raise ValueError(
                f"Unknown edge_head: {cfg.edge_head!r}. "
                f"Expected 'bilinear' | 'mlp' | 'gat'."
            )

    def forward(self, coords_t, mask, t_norm, raster):
        node_in = torch.cat([coords_t, mask.unsqueeze(-1)], dim=-1)
        h = self.node_in(node_in) + self.node_pos.unsqueeze(0)
        t_emb = self.time_embed(t_norm)
        h = h + t_emb.unsqueeze(1)

        memory = self.raster(raster)
        if self.cfg.no_raster:
            memory = torch.zeros_like(memory)
        key_padding = (mask < 0.5)
        h = self.tx(h, memory, tgt_key_padding_mask=key_padding)

        coords_pred = self.coord_head(h)
        edge_logits = self.edge_head_module(h)
        return coords_pred, edge_logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "GraphDiffusionConfigV2",
    "GraphDenoiserV2",
    "count_params",
    "cosine_alpha_bar",
    "edge_flip_rate",
]

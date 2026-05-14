"""W1-augmented training scaffold: closes the loop the paper proposes
but does not run end-to-end.

Adds a Wasserstein-1 surrogate term to the DiGress loss so that
training directly minimises the right-hand side of the bound

    CPR <= 2 * L * W1(mu_G, mu_*)

The surrogate is computed at each training step on the (X, E) pairs
produced by the model vs. the truth batch:

  - For coordinates, an entropic OT (Sinkhorn) loss between the
    model-predicted X_pred and the truth X.
  - For edges, a binary cross-entropy + per-graph degree-distribution
    discrepancy (a coarse W1 on the degree histogram, since exact
    graph-W1 is intractable for n_nodes = 200 in-loop).

Usage (smoke test, 50 epochs):
    python -m generators.graph_diffusion.train_w1 \
        --epochs 50 --ckpt-suffix w1_smoke

This script demonstrates that the W1 term can be added without
breaking training. We do NOT claim that the resulting checkpoint
out-performs final_v2.pt -- that would require multi-day compute
and architecture iteration. The smoke run logs whether the W1
loss decreases, which is the minimum required to prove the loop
closes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr.graphs import extract_graph
from generators.graph_diffusion.model import (
    GraphDenoiser, GraphDiffusionConfig, cosine_alpha_bar, count_params,
)
from generators.graph_diffusion.train import (
    CityDataset, _truth_graph_to_tensors, _downsample_raster,
    TRAIN_CACHE, HELDOUT, CHECKPOINT_DIR, TILE_PX,
)


# ---------------------------------------------------------------------
#  W1 surrogates
# ---------------------------------------------------------------------

def sinkhorn_w1(x: torch.Tensor, y: torch.Tensor,
                 mask_x: torch.Tensor | None = None,
                 mask_y: torch.Tensor | None = None,
                 reg: float = 0.05, n_iter: int = 30) -> torch.Tensor:
    """Entropic OT between two batched point clouds.

    x : (B, N, 2) predicted coords
    y : (B, N, 2) truth coords
    Returns scalar (mean over batch) Sinkhorn divergence.
    """
    B, N, _ = x.shape
    if mask_x is None:
        mask_x = torch.ones(B, N, device=x.device)
    if mask_y is None:
        mask_y = torch.ones(B, N, device=y.device)
    # Cost = pairwise L2.
    cost = torch.cdist(x, y, p=2)              # (B, N, N)
    a = mask_x / mask_x.sum(dim=1, keepdim=True).clamp_min(1.0)
    b = mask_y / mask_y.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_a = torch.log(a.clamp_min(1e-9))
    log_b = torch.log(b.clamp_min(1e-9))
    K = -cost / reg                            # log-K
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)
    for _ in range(n_iter):
        u = log_a - torch.logsumexp(K + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(K + u.unsqueeze(2), dim=1)
    pi = torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))
    return (pi * cost).sum(dim=(1, 2)).mean()


def degree_histogram_w1(edge_logits: torch.Tensor,
                         edges_truth: torch.Tensor,
                         mask: torch.Tensor,
                         n_bins: int = 7) -> torch.Tensor:
    """L1 distance between per-graph degree histograms.

    edge_logits : (B, N, N) raw logits
    edges_truth : (B, N, N) {0,1}
    mask        : (B, N) {0,1}
    """
    p = torch.sigmoid(edge_logits)
    deg_pred = (p * mask.unsqueeze(-1) * mask.unsqueeze(-2)).sum(dim=-1)
    deg_truth = (edges_truth * mask.unsqueeze(-1) * mask.unsqueeze(-2)
                  ).sum(dim=-1)
    bins = torch.linspace(0, n_bins - 1, n_bins, device=p.device)
    pred_hist = torch.stack([
        torch.exp(-((deg_pred - b) ** 2) / 0.5).mean(dim=-1)
        for b in bins
    ], dim=-1)
    true_hist = torch.stack([
        torch.exp(-((deg_truth - b) ** 2) / 0.5).mean(dim=-1)
        for b in bins
    ], dim=-1)
    return (pred_hist - true_hist).abs().sum(dim=-1).mean()


# ---------------------------------------------------------------------
#  Training loop with W1 augmentation
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-nodes", type=int, default=200)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--w1-weight", type=float, default=1.0,
                   help="Coefficient on the W1 (Sinkhorn coord) term.")
    p.add_argument("--deg-weight", type=float, default=0.5,
                   help="Coefficient on the degree-histogram W1 term.")
    p.add_argument("--ckpt-suffix", default="w1")
    args = p.parse_args(argv)

    cities = sorted(p_ for p_ in TRAIN_CACHE.glob("*.npz")
                    if p_.stem not in HELDOUT)
    print(f"Training cities: {len(cities)}")
    if not cities:
        print("ERROR: no training cities found.")
        return 2

    ds = CityDataset(cities, n_nodes=args.n_nodes)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    cfg = GraphDiffusionConfig(n_nodes=args.n_nodes,
                                 timesteps=args.timesteps)
    model = GraphDenoiser(cfg).to(args.device)
    print(f"Model: {count_params(model)/1e6:.2f}M params on "
          f"{args.device}")

    alpha_bar = cosine_alpha_bar(args.timesteps).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=1e-5)
    pos_weight = torch.tensor(50.0, device=args.device)
    bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

    history = []
    t0 = time()
    for epoch in range(args.epochs):
        epoch_total = epoch_coord = epoch_edge = epoch_w1 = epoch_deg = 0.0
        n_batches = 0
        for X, E, mask, raster in loader:
            X = X.to(args.device); E = E.to(args.device)
            mask = mask.to(args.device); raster = raster.to(args.device)
            B, N, _ = X.shape

            t = torch.randint(1, args.timesteps + 1, (B,),
                                device=args.device)
            ab = alpha_bar[t - 1].view(B, 1, 1)
            noise = torch.randn_like(X)
            X_noisy = (ab.sqrt() * X) + ((1 - ab).sqrt() * noise)
            t_norm = (t.float() - 1) / max(args.timesteps - 1, 1)
            X_pred, edge_logits = model(X_noisy, mask, t_norm, raster)

            coord_l = ((X_pred - X) ** 2).sum(-1) * mask
            coord_loss = coord_l.sum() / mask.sum().clamp_min(1.0)

            real_mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
            e_loss = bce(edge_logits, E) * real_mask_2d
            edge_loss = e_loss.sum() / real_mask_2d.sum().clamp_min(1.0)

            # W1 augmentation: Sinkhorn on coords + degree-hist L1.
            w1_coord = sinkhorn_w1(X_pred, X, mask, mask)
            w1_deg = degree_histogram_w1(edge_logits, E, mask)

            loss = (coord_loss + 0.5 * edge_loss
                    + args.w1_weight * w1_coord
                    + args.deg_weight * w1_deg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_total += float(loss.item())
            epoch_coord += float(coord_loss.item())
            epoch_edge += float(edge_loss.item())
            epoch_w1 += float(w1_coord.item())
            epoch_deg += float(w1_deg.item())
            n_batches += 1

        n = max(n_batches, 1)
        history.append({
            "epoch": epoch,
            "total": epoch_total / n,
            "coord": epoch_coord / n,
            "edge": epoch_edge / n,
            "w1_coord": epoch_w1 / n,
            "w1_deg": epoch_deg / n,
            "t_s": time() - t0,
        })
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.epochs - 1:
            h = history[-1]
            print(f"epoch {epoch+1:4d}/{args.epochs}  "
                  f"total={h['total']:.4f}  "
                  f"coord={h['coord']:.4f}  edge={h['edge']:.4f}  "
                  f"w1_coord={h['w1_coord']:.4f}  "
                  f"w1_deg={h['w1_deg']:.4f}  ({h['t_s']:.0f}s)")

    suffix = f"_{args.ckpt_suffix}" if args.ckpt_suffix else ""
    ckpt = CHECKPOINT_DIR / f"final{suffix}.pt"
    torch.save({
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "history": history,
    }, ckpt)
    print(f"Saved {ckpt}")

    history_path = CHECKPOINT_DIR / f"history{suffix}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

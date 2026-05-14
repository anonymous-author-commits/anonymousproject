"""F9 training driver with all sweep knobs:

  - edge head: bilinear | mlp | gat (model_v2)
  - MST-edge loss : BCE on the truth graph's MST edge mask
  - focal-style edge loss : standard focal BCE with gamma=2
  - curriculum : three-phase schedule that warms up coord+edge,
    then phases in MST-edge, then phases in W1-coord + focal
  - W1-coord (Sinkhorn) + degree-histogram-W1 surrogates (carried from
    train_w1.py)

Usage examples:
    python -m generators.graph_diffusion.train_v2 \
        --epochs 8000 --edge-head mlp --mst-loss-weight 5.0 \
        --focal-loss --ckpt-suffix B4

    python -m generators.graph_diffusion.train_v2 \
        --epochs 16000 --edge-head gat --mst-loss-weight 5.0 \
        --focal-loss --curriculum --ckpt-suffix E2

The script writes:
    checkpoints/final_<suffix>.pt
    checkpoints/history_<suffix>.json
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generators.graph_diffusion.model_v2 import (
    GraphDenoiserV2, GraphDiffusionConfigV2, cosine_alpha_bar, count_params,
)
from generators.graph_diffusion.train import (
    CityDataset, TRAIN_CACHE, HELDOUT, CHECKPOINT_DIR,
)
from generators.graph_diffusion.train_w1 import (
    sinkhorn_w1, degree_histogram_w1,
)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def _truth_mst_mask(E: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """For each batch element, compute the MST edge mask of the truth
    adjacency E among real (mask>0.5) nodes.

    E    : (B, N, N) {0,1} adjacency
    mask : (B, N) {0,1}
    Returns (B, N, N) {0,1} mask whose 1-entries are the truth-graph
    MST edges (using uniform unit weights since edge lengths aren't
    available at this point in the loop).
    """
    B, N, _ = E.shape
    out = torch.zeros_like(E)
    for b in range(B):
        real = (mask[b] > 0.5)
        idx = torch.nonzero(real, as_tuple=False).squeeze(-1)
        if idx.numel() < 2:
            continue
        sub = E[b][idx][:, idx].cpu().numpy()
        # Build MST via Kruskal on unit weights, breaking ties by
        # numpy's stable argsort. Weight = 1 if edge present, else
        # large penalty (excluded).
        n = sub.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if sub[i, j] > 0.5:
                    edges.append((1.0, i, j))
        if not edges:
            continue
        # Union-find.
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        edges.sort(key=lambda e: e[0])
        mst_pairs: list[tuple[int, int]] = []
        for w, i, j in edges:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                mst_pairs.append((i, j))
                if len(mst_pairs) == n - 1:
                    break
        for i_local, j_local in mst_pairs:
            i_g = idx[i_local].item()
            j_g = idx[j_local].item()
            out[b, i_g, j_g] = 1.0
            out[b, j_g, i_g] = 1.0
    return out


def mst_edge_loss(edge_logits: torch.Tensor, E: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """BCE-on-MST-edges loss.

    Penalises the model for low predicted probability on edges that ARE
    in the truth graph's MST. Uses pos_weight=10 to overcome the
    sparsity of MST edges relative to all node pairs.
    """
    mst_mask = _truth_mst_mask(E, mask)
    real_mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    pos_weight = torch.tensor(10.0, device=edge_logits.device)
    bce = F.binary_cross_entropy_with_logits(
        edge_logits, mst_mask, pos_weight=pos_weight, reduction="none",
    )
    # Only penalise on real-node pairs.
    weighted = bce * real_mask_2d
    return weighted.sum() / real_mask_2d.sum().clamp_min(1.0)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def focal_edge_loss(edge_logits: torch.Tensor, E: torch.Tensor,
                     mask: torch.Tensor, gamma: float = 2.0,
                     pos_weight: float = 50.0) -> torch.Tensor:
    """Focal-BCE on edges: BCE * (1 - p_t)^gamma.

    Pushes the model to focus on hard examples (uncertain edges)
    rather than already-correct ones. Pos-weight retained at 50x as
    in the baseline trainer.
    """
    real_mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
    p = torch.sigmoid(edge_logits)
    p_t = E * p + (1 - E) * (1 - p)
    focal_factor = (1 - p_t).clamp(min=1e-6) ** gamma
    bce = F.binary_cross_entropy_with_logits(
        edge_logits, E,
        pos_weight=torch.tensor(pos_weight, device=edge_logits.device),
        reduction="none",
    )
    weighted = bce * focal_factor * real_mask_2d
    return weighted.sum() / real_mask_2d.sum().clamp_min(1.0)


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

def curriculum_weights(epoch: int, total_epochs: int,
                        base_mst: float, base_w1_coord: float,
                        base_focal: bool) -> dict:
    """Three-phase curriculum.

      Phase 1 (0 -> 0.25 * total):     coord + edge_BCE only
      Phase 2 (0.25 -> 0.75 * total):  MST loss linearly phased in
      Phase 3 (0.75 -> total):         W1-coord + focal phased in
    """
    p1 = total_epochs * 0.25
    p2 = total_epochs * 0.75
    if epoch < p1:
        return {"mst": 0.0, "w1_coord": 0.0, "focal_active": False}
    elif epoch < p2:
        ramp = (epoch - p1) / max(p2 - p1, 1)
        return {"mst": base_mst * ramp,
                "w1_coord": 0.0,
                "focal_active": False}
    else:
        ramp = (epoch - p2) / max(total_epochs - p2, 1)
        return {"mst": base_mst,
                "w1_coord": base_w1_coord * ramp,
                "focal_active": base_focal and ramp >= 1.0}


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-nodes", type=int, default=200)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--edge-head", default="bilinear",
                   choices=["bilinear", "mlp", "gat"])
    p.add_argument("--mst-loss-weight", type=float, default=0.0,
                   help="Weight on the MST-edge BCE term (F9.2). 0 disables.")
    p.add_argument("--w1-coord-weight", type=float, default=1.0,
                   help="Weight on the Sinkhorn coord-W1 term.")
    p.add_argument("--deg-weight", type=float, default=0.5,
                   help="Weight on the degree-histogram W1 term.")
    p.add_argument("--focal-loss", action="store_true",
                   help="Replace the edge BCE with focal-BCE (gamma=2). F9.3.")
    p.add_argument("--curriculum", action="store_true",
                   help="Enable the three-phase curriculum schedule. F9.3b.")
    p.add_argument("--pos-weight", type=float, default=50.0)
    p.add_argument("--ckpt-suffix", default="v2")
    p.add_argument("--checkpoint-every", type=int, default=2000)
    args = p.parse_args(argv)

    # Optional override via env var (used by F11.3 transfer-learning).
    import os
    override = os.environ.get("CPR_TRAIN_CITIES", "").strip()
    if override:
        wanted = {c.strip() for c in override.split(",") if c.strip()}
        cities = sorted(p_ for p_ in TRAIN_CACHE.glob("*.npz")
                        if p_.stem in wanted)
        print(f"Override CPR_TRAIN_CITIES: {len(cities)} of "
              f"{len(wanted)} requested found in cache", flush=True)
    else:
        cities = sorted(p_ for p_ in TRAIN_CACHE.glob("*.npz")
                        if p_.stem not in HELDOUT)
    print(f"Training cities: {len(cities)}", flush=True)
    if not cities:
        print("ERROR: no training cities found.")
        return 2

    ds = CityDataset(cities, n_nodes=args.n_nodes)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
    )
    cfg = GraphDiffusionConfigV2(
        n_nodes=args.n_nodes, timesteps=args.timesteps,
        edge_head=args.edge_head,
    )
    model = GraphDenoiserV2(cfg).to(args.device)
    print(f"Model: {count_params(model)/1e6:.2f}M params, "
          f"edge_head={cfg.edge_head}, on {args.device}", flush=True)

    alpha_bar = cosine_alpha_bar(args.timesteps).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=1e-5)

    history = []
    t0 = time()
    for epoch in range(args.epochs):
        cur = curriculum_weights(
            epoch, args.epochs,
            base_mst=args.mst_loss_weight,
            base_w1_coord=args.w1_coord_weight,
            base_focal=args.focal_loss,
        ) if args.curriculum else {
            "mst": args.mst_loss_weight,
            "w1_coord": args.w1_coord_weight,
            "focal_active": args.focal_loss,
        }

        epoch_total = epoch_coord = epoch_edge = 0.0
        epoch_mst = epoch_w1 = epoch_deg = 0.0
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

            if cur["focal_active"]:
                edge_loss = focal_edge_loss(
                    edge_logits, E, mask,
                    gamma=2.0, pos_weight=args.pos_weight,
                )
            else:
                real_mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)
                bce = F.binary_cross_entropy_with_logits(
                    edge_logits, E,
                    pos_weight=torch.tensor(args.pos_weight,
                                              device=args.device),
                    reduction="none",
                )
                edge_loss = (bce * real_mask_2d).sum() / \
                            real_mask_2d.sum().clamp_min(1.0)

            mst_l = (mst_edge_loss(edge_logits, E, mask)
                     if cur["mst"] > 0 else
                     torch.zeros((), device=args.device))
            w1_coord_l = (sinkhorn_w1(X_pred, X, mask, mask)
                          if cur["w1_coord"] > 0 else
                          torch.zeros((), device=args.device))
            deg_l = degree_histogram_w1(edge_logits, E, mask)

            loss = (coord_loss + 0.5 * edge_loss
                    + cur["mst"] * mst_l
                    + cur["w1_coord"] * w1_coord_l
                    + args.deg_weight * deg_l)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_total += float(loss.item())
            epoch_coord += float(coord_loss.item())
            epoch_edge += float(edge_loss.item())
            epoch_mst += float(mst_l.item())
            epoch_w1 += float(w1_coord_l.item())
            epoch_deg += float(deg_l.item())
            n_batches += 1

        n = max(n_batches, 1)
        history.append({
            "epoch": epoch,
            "total": epoch_total / n,
            "coord": epoch_coord / n,
            "edge": epoch_edge / n,
            "mst": epoch_mst / n,
            "w1_coord": epoch_w1 / n,
            "w1_deg": epoch_deg / n,
            "cur_mst": cur["mst"],
            "cur_w1_coord": cur["w1_coord"],
            "cur_focal": cur["focal_active"],
            "t_s": time() - t0,
        })
        if (epoch + 1) % 200 == 0 or epoch == 0 or epoch == args.epochs - 1:
            h = history[-1]
            print(f"epoch {epoch+1:5d}/{args.epochs}  "
                  f"total={h['total']:.4f}  coord={h['coord']:.4f}  "
                  f"edge={h['edge']:.4f}  mst={h['mst']:.4f}  "
                  f"w1c={h['w1_coord']:.4f}  deg={h['w1_deg']:.4f}  "
                  f"({h['t_s']:.0f}s)", flush=True)

    suffix = f"_{args.ckpt_suffix}" if args.ckpt_suffix else ""
    ckpt = CHECKPOINT_DIR / f"final{suffix}.pt"
    torch.save({
        "epoch": args.epochs,
        "state_dict": model.state_dict(),
        "config": cfg.__dict__,
        "history": history,
        "args": vars(args),
    }, ckpt)
    print(f"Saved {ckpt}", flush=True)

    history_path = CHECKPOINT_DIR / f"history{suffix}.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

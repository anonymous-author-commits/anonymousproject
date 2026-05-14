"""Inference / sampling for the adapted graph-diffusion generator.

Loads a trained checkpoint, runs the reverse process from random
init conditioned on a held-out city's raster, and produces a
NetworkX MultiGraph compatible with the rest of the CPR pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generators.graph_diffusion.model import (  # noqa: E402
    GraphDenoiser, GraphDiffusionConfig, cosine_alpha_bar,
)
from generators.graph_diffusion.train import _downsample_raster  # noqa: E402

REPO_ROOT = ROOT.parent
CACHE_DIR = REPO_ROOT / "DUPT" / "data" / "cache_hv_v2"
CHECKPOINT_DIR = ROOT / "generators" / "graph_diffusion" / "checkpoints"

TILE_PX = 1536


def load_model(ckpt_path: Path, device: str = "cpu"):
    """Load a checkpoint and rebuild the model.

    Auto-detects whether the checkpoint was saved by train.py
    (GraphDiffusionConfig / GraphDenoiser) or train_v2.py
    (GraphDiffusionConfigV2 / GraphDenoiserV2) by inspecting the
    config dict for V2-only keys (edge_head, edge_mlp_hidden, etc.).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]
    if "edge_head" in cfg_dict:
        from generators.graph_diffusion.model_v2 import (
            GraphDenoiserV2, GraphDiffusionConfigV2,
        )
        cfg = GraphDiffusionConfigV2(**cfg_dict)
        model = GraphDenoiserV2(cfg).to(device)
    else:
        cfg = GraphDiffusionConfig(**cfg_dict)
        model = GraphDenoiser(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def sample_edge_logits(
    model: GraphDenoiser,
    raster: torch.Tensor,
    *,
    n_active_nodes: int = 100,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run the reverse process and return (coords, edge_p) without
    binarising — used for per-city threshold calibration.
    """
    cfg = model.cfg
    rng_torch = torch.Generator(device=device).manual_seed(seed)
    T = cfg.timesteps
    alpha_bar = cosine_alpha_bar(T).to(device)
    x_t = torch.randn(
        1, cfg.n_nodes, cfg.coord_dim,
        generator=rng_torch, device=device,
    )
    mask = torch.zeros(1, cfg.n_nodes, device=device)
    mask[0, :n_active_nodes] = 1.0
    for t_step in reversed(range(T)):
        ab = alpha_bar[t_step]
        t_norm = torch.full((1,), t_step / max(T - 1, 1), device=device)
        x_pred, edge_logits = model(x_t, mask, t_norm, raster)
        if t_step > 0:
            ab_prev = alpha_bar[t_step - 1]
            x_t = (
                ab_prev.sqrt() * x_pred
                + (1 - ab_prev).sqrt() * (x_t - ab.sqrt() * x_pred)
                / (1 - ab).sqrt().clamp_min(1e-6)
            )
        else:
            x_t = x_pred
    coords = x_t[0, :n_active_nodes].cpu().numpy()
    edge_p = torch.sigmoid(edge_logits[0, :n_active_nodes, :n_active_nodes])
    edge_p = edge_p.cpu().numpy()
    np.fill_diagonal(edge_p, 0.0)
    return coords, edge_p


def calibrate_threshold(edge_p: np.ndarray, target_n_edges: int) -> float:
    """Pick the edge-probability threshold that yields target_n_edges
    in the upper triangle. Returns 0.5 if target_n_edges <= 0.
    """
    if target_n_edges <= 0:
        return 0.5
    n = edge_p.shape[0]
    iu = np.triu_indices(n, k=1)
    probs = np.sort(edge_p[iu])[::-1]
    if target_n_edges >= len(probs):
        return float(probs[-1])
    return float(probs[target_n_edges])


def _truth_edge_count(city: str) -> int:
    """Approximate truth edge count (from graphs/_truth/{city}.graphml)."""
    p = ROOT / "results" / "graphs" / "_truth" / f"{city}.graphml"
    if not p.exists():
        return 100
    g = nx.read_graphml(str(p))
    return g.number_of_edges()


@torch.no_grad()
def sample_graph(
    model: GraphDenoiser,
    raster: torch.Tensor,
    *,
    n_active_nodes: int = 100,
    edge_threshold: float = 0.3,
    seed: int = 0,
    device: str = "cpu",
) -> nx.MultiGraph:
    """Single sample from the diffusion model's posterior over multigraphs.

    raster: (1, 6, H, W) tensor.
    Returns: a NetworkX MultiGraph with `px` (row, col) node attributes
    and `length_px` edge attribute, comparable to the rest of the
    pipeline.
    """
    cfg = model.cfg
    rng_torch = torch.Generator(device=device).manual_seed(seed)
    T = cfg.timesteps
    alpha_bar = cosine_alpha_bar(T).to(device)

    # Initialise from the prior.
    x_t = torch.randn(
        1, cfg.n_nodes, cfg.coord_dim,
        generator=rng_torch, device=device,
    )
    mask = torch.zeros(1, cfg.n_nodes, device=device)
    mask[0, :n_active_nodes] = 1.0

    # Reverse process. Use the model's predicted x0 to step, with no
    # auxiliary noise schedule -- this is the deterministic-DDIM-style
    # variant which is fast and adequate for the planning use case.
    for t_step in reversed(range(T)):
        ab = alpha_bar[t_step]
        t_norm = torch.full((1,), t_step / max(T - 1, 1), device=device)
        x_pred, edge_logits = model(x_t, mask, t_norm, raster)
        if t_step > 0:
            ab_prev = alpha_bar[t_step - 1]
            x_t = (
                ab_prev.sqrt() * x_pred
                + (1 - ab_prev).sqrt() * (x_t - ab.sqrt() * x_pred)
                / (1 - ab).sqrt().clamp_min(1e-6)
            )
        else:
            x_t = x_pred

    coords = x_t[0, :n_active_nodes].cpu().numpy()  # in [0, 1]
    edge_p = torch.sigmoid(edge_logits[0, :n_active_nodes, :n_active_nodes])
    edge_p = edge_p.cpu().numpy()
    np.fill_diagonal(edge_p, 0.0)

    # Build NetworkX MultiGraph.
    g = nx.MultiGraph()
    pixel_coords = (coords * TILE_PX).clip(0, TILE_PX - 1).astype(int)
    for i, (r, c) in enumerate(pixel_coords):
        key = (int(r), int(c))
        g.add_node(key, px=key)

    for i in range(n_active_nodes):
        for j in range(i + 1, n_active_nodes):
            if edge_p[i, j] >= edge_threshold:
                u_key = (int(pixel_coords[i, 0]), int(pixel_coords[i, 1]))
                v_key = (int(pixel_coords[j, 0]), int(pixel_coords[j, 1]))
                if u_key == v_key:
                    continue
                length = float(np.linalg.norm(coords[i] - coords[j]) * TILE_PX)
                g.add_edge(u_key, v_key, length_px=length)
    return g


def sample_for_city(
    model: GraphDenoiser,
    city: str,
    *,
    n_samples: int = 8,
    device: str = "cpu",
    n_active_nodes: int = 100,
    edge_threshold: float = 0.3,
    calibrate_to_truth: bool = False,
) -> list[nx.MultiGraph]:
    """Sample n_samples graphs for a held-out city's conditioning raster.

        If calibrate_to_truth is True, edge_threshold is overridden per-sample
        to match the truth-graph edge count for that city.

    """
    npz = CACHE_DIR / f"{city}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"No cache for city {city}: {npz}")
    cond = np.load(npz)["cond"].astype(np.float32)
    raster_np = _downsample_raster(cond, target=384)
    raster = torch.from_numpy(raster_np).unsqueeze(0).to(device)

    target_edges = _truth_edge_count(city) if calibrate_to_truth else 0

    samples = []
    for s in range(n_samples):
        if calibrate_to_truth:
            coords, edge_p = sample_edge_logits(
                model, raster,
                n_active_nodes=n_active_nodes, seed=s, device=device,
            )
            thr = calibrate_threshold(edge_p, target_n_edges=target_edges)
            g = _build_graph_from_arrays(coords, edge_p, thr, n_active_nodes)
        else:
            g = sample_graph(
                model, raster, n_active_nodes=n_active_nodes,
                edge_threshold=edge_threshold, seed=s, device=device,
            )
        samples.append(g)
    return samples


def _build_graph_from_arrays(coords: np.ndarray, edge_p: np.ndarray,
                              edge_threshold: float, n_active_nodes: int
                              ) -> nx.MultiGraph:
    g = nx.MultiGraph()
    pixel_coords = (coords * TILE_PX).clip(0, TILE_PX - 1).astype(int)
    for i, (r, c) in enumerate(pixel_coords):
        key = (int(r), int(c))
        g.add_node(key, px=key)
    for i in range(n_active_nodes):
        for j in range(i + 1, n_active_nodes):
            if edge_p[i, j] >= edge_threshold:
                u_key = (int(pixel_coords[i, 0]), int(pixel_coords[i, 1]))
                v_key = (int(pixel_coords[j, 0]), int(pixel_coords[j, 1]))
                if u_key == v_key:
                    continue
                length = float(np.linalg.norm(coords[i] - coords[j]) * TILE_PX)
                g.add_edge(u_key, v_key, length_px=length)
    return g


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=str(CHECKPOINT_DIR / "final.pt"))
    parser.add_argument("--cities", nargs="*",
                       default=["zurich", "berlin", "chicago", "bangkok", "sao_paulo"])
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-active-nodes", type=int, default=100)
    parser.add_argument("--edge-threshold", type=float, default=0.3)
    parser.add_argument("--calibrate-to-truth", action="store_true",
                       help="F6.3: pick per-sample threshold to match "
                            "truth-graph edge count.")
    parser.add_argument("--out-suffix", default="",
                       help="Subfolder suffix under graphs_mc (e.g. "
                            "'_v2' creates graphs_mc/digress_v1_v2/).")
    args = parser.parse_args(argv)

    print(f"Loading checkpoint: {args.ckpt}")
    model = load_model(Path(args.ckpt), device=args.device)

    folder = f"digress_v1{args.out_suffix}" if args.out_suffix else "digress_v1"
    out_root = ROOT / "results" / "graphs_mc" / folder
    out_root.mkdir(parents=True, exist_ok=True)
    for city in args.cities:
        samples = sample_for_city(
            model, city,
            n_samples=args.n_samples,
            device=args.device,
            n_active_nodes=args.n_active_nodes,
            edge_threshold=args.edge_threshold,
            calibrate_to_truth=args.calibrate_to_truth,
        )
        for s, g in enumerate(samples):
            from cpr.graphs import save_graphml
            save_graphml(g, out_root / f"{city}_{s:02d}.graphml")
        print(f"[digress_v1/{city}] wrote {len(samples)} samples "
              f"(typical |V|={samples[0].number_of_nodes()}, "
              f"|E|={samples[0].number_of_edges()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

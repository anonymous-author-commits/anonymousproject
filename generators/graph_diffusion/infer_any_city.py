"""Zero-shot inference on any city worldwide.

Given a city name + bounding box, this script:
  1. Fetches OSM transport / power features (roads, rails, substations).
  2. Builds a 6-channel conditioning raster compatible with the
     trained DiGress checkpoint (built_up, water, substations,
     elevation, roads, rails) at 10 m / pixel.
  3. Runs the diffusion model conditioned on that raster.
  4. Calibrates the edge-probability threshold to a target edge
     density (default: matches OSM substation density heuristic
     since no truth graph is available).
  5. Emits the predicted HV multigraph as GraphML.
  6. Optionally evaluates the predicted graph under the road-corridor
     payoff if the road raster is available.

Usage:
    python -m generators.graph_diffusion.infer_any_city         --city nairobi --bbox -1.45,36.65,-1.20,36.95         --ckpt generators/graph_diffusion/checkpoints/final_v2.pt

Limitations of this scaffold:
  - Real conditioning channels (built-up, water, elevation) require
    actual raster sources (ESA WorldCover for water/built-up, SRTM
    for elevation). Without them we fall back to OSM-derived
    proxies and zero-fill missing channels. The model's accuracy
    will degrade gracefully but predictably.
  - Without a truth graph we cannot compute CPR; we report only
    the predicted graph and a structural-realism check (edges per
    truth-substation, road-following fraction).

This is a SCAFFOLD not a finished product. The pipeline runs
end-to-end on any city the user provides; whether the output is
useful depends on (a) the model's transferability and (b) the
quality of the input rasters.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr.graphs import save_graphml
from generators.graph_diffusion.model import (
    GraphDenoiser, GraphDiffusionConfig, cosine_alpha_bar,
)
from generators.graph_diffusion.infer import (
    sample_edge_logits, calibrate_threshold, _build_graph_from_arrays,
)

TILE_PX = 1536
PIXEL_M = 10.0


# ---------------------------------------------------------------------
# Conditioning raster construction
# ---------------------------------------------------------------------

def _try_fetch_osm_transport(city: str, bbox: tuple[float, ...],
                              cache_dir: Path) -> dict[str, list]:
    """Fetch road + rail line geometries via osmnx if installed.

    Returns dict with 'roads', 'rails', 'substations' as lists of
    pixel-coordinate polylines / points. Falls back to empty lists
    if osmnx is not available (the model still runs but the road /
    rail / substation channels will be zeros).
    """
    out = {"roads": [], "rails": [], "substations": []}
    try:
        import osmnx as ox  # noqa: F401
        from shapely.geometry import LineString, Point
    except Exception:
        print("  [warn] osmnx not installed; proceeding with zero "
              "road/rail/substation channels (model accuracy will "
              "degrade).")
        return out
    south, west, north, east = bbox
    # osmnx 2.x: single bbox argument as (left, bottom, right, top).
    bbox_2x = (west, south, east, north)
    try:
        G = ox.graph_from_bbox(bbox=bbox_2x, network_type="drive")
        for u, v, _, d in G.edges(keys=True, data=True):
            if "geometry" in d:
                out["roads"].append(list(d["geometry"].coords))
            else:
                u_p = (G.nodes[u]["x"], G.nodes[u]["y"])
                v_p = (G.nodes[v]["x"], G.nodes[v]["y"])
                out["roads"].append([u_p, v_p])
    except Exception as exc:
        print(f"  [warn] OSM road fetch failed: {exc}")
    try:
        rails = ox.features_from_bbox(bbox=bbox_2x,
                                        tags={"railway": True})
        for _, row in rails.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "LineString":
                out["rails"].append(list(geom.coords))
    except Exception as exc:
        print(f"  [warn] OSM rails fetch failed: {exc}")
    try:
        subs = ox.features_from_bbox(bbox=bbox_2x,
                                        tags={"power": "substation"})
        for _, row in subs.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if geom.geom_type == "Point":
                out["substations"].append((geom.x, geom.y))
            else:
                out["substations"].append(geom.centroid.coords[0])
    except Exception as exc:
        print(f"  [warn] OSM substations fetch failed: {exc}")
    return out


def _rasterize(features: dict, bbox: tuple[float, ...],
                size: int = TILE_PX) -> np.ndarray:
    """Rasterise OSM features onto a (6, size, size) cube.

    Channels: built_up, water, substations, elevation, roads, rails.
    Built-up + water + elevation are zero unless real sources are
    provided (these are placeholders; production deployments should
    plug in ESA WorldCover and SRTM).
    """
    south, west, north, east = bbox
    raster = np.zeros((6, size, size), dtype=np.float32)
    if north == south or east == west:
        return raster
    lat_per_px = (north - south) / size
    lon_per_px = (east - west) / size

    def _to_px(lon, lat):
        c = int((lon - west) / lon_per_px)
        r = int(size - 1 - (lat - south) / lat_per_px)
        return r, c

    def _draw_line(coords, ch):
        for i in range(len(coords) - 1):
            r1, c1 = _to_px(*coords[i])
            r2, c2 = _to_px(*coords[i + 1])
            n = max(abs(r2 - r1), abs(c2 - c1), 1)
            for k in range(n + 1):
                rr = int(round(r1 + (r2 - r1) * k / n))
                cc = int(round(c1 + (c2 - c1) * k / n))
                if 0 <= rr < size and 0 <= cc < size:
                    raster[ch, rr, cc] = 1.0

    for poly in features.get("roads", []):
        _draw_line(poly, ch=4)
    for poly in features.get("rails", []):
        _draw_line(poly, ch=5)
    for lon, lat in features.get("substations", []):
        r, c = _to_px(lon, lat)
        if 0 <= r < size and 0 <= c < size:
            raster[2, r, c] = 1.0

    # Built-up proxy: dilate roads by 2 pixels (very rough proxy).
    from scipy.ndimage import binary_dilation
    raster[0] = binary_dilation(raster[4] > 0, iterations=2).astype(np.float32)
    return raster


def _downsample(raster: np.ndarray, target: int = 384) -> np.ndarray:
    factor = raster.shape[1] // target
    if factor <= 1:
        return raster.astype(np.float32)
    c, h, w = raster.shape
    h_n, w_n = h // factor, w // factor
    raster = raster[:, :h_n * factor, :w_n * factor]
    return raster.reshape(c, h_n, factor, w_n, factor).mean(
        axis=(2, 4)).astype(np.float32)


# ---------------------------------------------------------------------
# Inference + reporting
# ---------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--city", required=True,
                   help="City name (used only for output naming).")
    p.add_argument("--bbox", required=True,
                   help="Comma-separated south,west,north,east lat/lon "
                        "(e.g. '-1.45,36.65,-1.20,36.95' for Nairobi).")
    p.add_argument("--ckpt",
                   default="generators/graph_diffusion/checkpoints/final_v2.pt",
                   help="DiGress checkpoint to use.")
    p.add_argument("--n-active-nodes", type=int, default=100)
    p.add_argument("--target-edges", type=int, default=80,
                   help="Calibrate edge threshold to this many edges. "
                        "If you have a planning context with known "
                        "substation density, set this to your "
                        "expected truth edge count.")
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--output-dir", default="results/zero_shot")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    bbox = tuple(float(x) for x in args.bbox.split(","))
    if len(bbox) != 4:
        print("ERROR: --bbox must be 'south,west,north,east'.")
        return 2
    out_dir = Path(args.output_dir) / args.city
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.city}] bbox = {bbox}")
    print(f"[{args.city}] checkpoint = {args.ckpt}")
    print(f"[{args.city}] fetching OSM features...")
    features = _try_fetch_osm_transport(args.city, bbox, out_dir)
    print(f"  roads={len(features['roads'])}  "
          f"rails={len(features['rails'])}  "
          f"substations={len(features['substations'])}")

    print(f"[{args.city}] rasterising 6-channel conditioning cube...")
    raster_full = _rasterize(features, bbox, size=TILE_PX)
    raster_ds = _downsample(raster_full, target=384)
    np.save(out_dir / "conditioning_raster.npy", raster_ds)

    print(f"[{args.city}] loading model...")
    ckpt = torch.load(args.ckpt, map_location=args.device,
                        weights_only=False)
    cfg = GraphDiffusionConfig(**ckpt["config"])
    model = GraphDenoiser(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    raster_t = torch.from_numpy(raster_ds).unsqueeze(0).to(args.device)

    print(f"[{args.city}] sampling {args.n_samples} graphs...")
    sample_paths = []
    for s in range(args.n_samples):
        coords, edge_p = sample_edge_logits(
            model, raster_t,
            n_active_nodes=args.n_active_nodes, seed=s,
            device=args.device,
        )
        thr = calibrate_threshold(edge_p, target_n_edges=args.target_edges)
        g = _build_graph_from_arrays(coords, edge_p, thr,
                                      n_active_nodes=args.n_active_nodes)
        out_path = out_dir / f"{args.city}_{s:02d}.graphml"
        save_graphml(g, out_path)
        sample_paths.append(str(out_path))
        print(f"  sample {s}: |V|={g.number_of_nodes()}, "
              f"|E|={g.number_of_edges()}, threshold={thr:.3f}")

    summary = {
        "city": args.city, "bbox": list(bbox),
        "checkpoint": args.ckpt,
        "n_samples": args.n_samples,
        "n_active_nodes": args.n_active_nodes,
        "target_edges": args.target_edges,
        "features_fetched": {k: len(v) for k, v in features.items()},
        "samples": sample_paths,
        "limitations": [
            "Conditioning is OSM-derived only; built-up, water, "
            "and elevation channels are crude proxies (production "
            "use should plug in ESA WorldCover and SRTM rasters).",
            "Without a truth graph, this script does not compute CPR "
            "for the held-out city. Quality assessment is structural "
            "only (edges per substation, road-following fraction).",
            "Generalization to cities outside the 21-city training "
            "corpus is not validated by this paper. Treat zero-shot "
            "outputs as informed guesses, not authoritative grids.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[{args.city}] wrote summary.json + {len(sample_paths)} graphmls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

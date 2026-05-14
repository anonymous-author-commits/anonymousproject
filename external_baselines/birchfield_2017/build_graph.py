"""Birchfield 2017 synthetic-grid reproduction.

Reference
---------
A.B. Birchfield, T. Xu, K.M. Gegner, K.S. Shetye, T.J. Overbye.
"Grid Structural Characteristics as Validation Criteria for Synthetic
Networks." IEEE Transactions on Power Systems 32(4):3258-3265 (2017).

Algorithm (paraphrased from sec III of the paper)
--------------------------------------------------
1. Substation siting.  Given a load-density raster L(r, c) (population
   or built-up-area proxy), place N substations at the centroids of a
   weighted k-means clustering of pixels-with-load. The number N is
   set to match the truth-graph cardinality for a fair comparison
   (Birchfield's algorithm scales N with system load; we calibrate N
   per-city for evaluation parity).

2. Voltage-class assignment.  Each substation is assigned to one of
   {HV, MV, LV} by the total load served (Birchfield Table 2):
       load > 100 MVA   -> HV
       5 < load <= 100  -> MV
       load <= 5        -> LV
   We use the per-cluster sum of the load proxy as the load.

3. Line construction.  Connect each substation to its k = 3 nearest
   neighbours by Euclidean pixel distance. Birchfield prunes edges
   that exceed a "load-balance" threshold (their Eq. 4); we apply a
   simplified pruning: drop edges whose length exceeds the median
   of the kept edges multiplied by ALPHA_PRUNE = 2.5 (a conservative
   substitute for the per-edge thermal-limit check, which would
   require an explicit DC OPF run).

4. Output.  NetworkX MultiGraph with ``px_r`` and ``px_c`` integer
   pixel-coordinate node attributes, matching the GraphML schema
   used by every other generator in the panel.

This reproduction is deterministic given a seed and is independent of
the DUPT training corpus and DiGress checkpoints; it does not see any
of the panel-specific training data. It is therefore the panel's
externally-defined comparison point.

The built-up-area channel of the conditioning cube (ESA WorldCover
classification at 10 m/px, scaled to [-1, +1]) is used as the load
proxy. Pixels with built-up value above zero are considered candidate
load nodes; this matches Birchfield's "where the load lives" input
without requiring proprietary population data.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np


# Tuning knobs from the paper + reasonable defaults.
K_NEAREST_NEIGHBOURS: int = 3   # Birchfield sec III-B
ALPHA_PRUNE: float = 2.5         # length-pruning threshold multiplier
KMEANS_ITERS: int = 50
LOAD_THRESHOLD: float = 0.0      # built-up channel > 0 = load present


def _seed_kmeans_pp(
    points: np.ndarray, k: int, *, rng: np.random.Generator,
) -> np.ndarray:
    """k-means++ initialisation; returns initial (k, 2) centres."""
    n = points.shape[0]
    centres = np.empty((k, points.shape[1]), dtype=np.float64)
    centres[0] = points[rng.integers(0, n)]
    for i in range(1, k):
        d2 = ((points[:, None, :] - centres[None, :i, :]) ** 2).sum(axis=-1)
        d2 = d2.min(axis=1)
        if d2.sum() <= 0:
            centres[i] = points[rng.integers(0, n)]
        else:
            probs = d2 / d2.sum()
            centres[i] = points[rng.choice(n, p=probs)]
    return centres


def _weighted_kmeans(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    n_iter: int = KMEANS_ITERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Lloyd's k-means.

    Returns
    -------
    centres : (k, 2) float64 — final cluster centroids in pixel space.
    labels  : (n,) int64 — per-point cluster assignment.
    masses  : (k,) float64 — sum of weights per cluster (= "load").
    """
    rng = np.random.default_rng(seed)
    if points.shape[0] == 0:
        return (
            np.zeros((0, 2)),
            np.zeros(0, dtype=np.int64),
            np.zeros(0),
        )
    k = min(k, points.shape[0])
    centres = _seed_kmeans_pp(points, k, rng=rng)
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(n_iter):
        d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                w = weights[mask]
                w_sum = float(w.sum())
                if w_sum > 0:
                    centres[j] = (points[mask] * w[:, None]).sum(axis=0) / w_sum
                else:
                    centres[j] = points[mask].mean(axis=0)
    masses = np.array(
        [weights[labels == j].sum() for j in range(k)],
        dtype=np.float64,
    )
    return centres, labels, masses


def _voltage_class(load: float, *, total_load: float) -> str:
    """Birchfield Table 2 thresholds, rescaled to the per-city load total."""
    # Express the load as percentage of city-total so the thresholds
    # transfer between cities of very different sizes.
    pct = (load / total_load) * 100.0 if total_load > 0 else 0.0
    if pct >= 10.0:
        return "HV"
    if pct >= 1.0:
        return "MV"
    return "LV"


def _knn_edges(
    coords: np.ndarray, k: int,
) -> list[tuple[int, int, float]]:
    """k-nearest-neighbour edges (undirected, deduplicated) with lengths."""
    n = coords.shape[0]
    if n == 0:
        return []
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    edges: set[tuple[int, int, float]] = set()
    actual_k = min(k, n - 1)
    if actual_k <= 0:
        return []
    nn = np.argsort(d2, axis=1)[:, :actual_k]
    for i in range(n):
        for j in nn[i]:
            if j == i:
                continue
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b, float(np.sqrt(d2[a, b]))))
    return sorted(edges)


def _prune_long_edges(
    edges: list[tuple[int, int, float]],
    *,
    alpha: float = ALPHA_PRUNE,
) -> list[tuple[int, int, float]]:
    """Drop edges whose length exceeds alpha * median(kept lengths).

    The pruning loop is single-pass: take the median of the full edge
    set, then drop edges above alpha * median. This is the simplified
    surrogate for Birchfield's per-edge thermal-limit check; the
    resulting graph remains connected with high probability for
    K_NEAREST_NEIGHBOURS = 3 because each node retains its shortest
    neighbour.
    """
    if not edges:
        return []
    lengths = np.array([e[2] for e in edges])
    threshold = alpha * float(np.median(lengths))
    return [e for e in edges if e[2] <= threshold]


def build_birchfield_graph(
    cond_cube: np.ndarray,
    n_substations: int,
    *,
    seed: int = 0,
    built_up_channel: int = 0,
) -> nx.MultiGraph:
    """Construct a single Birchfield 2017 synthetic-grid sample.

    Parameters
    ----------
    cond_cube
        Conditioning array of shape (C, H, W). The built-up channel is
        used as the load-density proxy.
    n_substations
        Calibration target: place this many substations. Set equal to
        the truth-graph cardinality for the city for fair comparison
        with other panel generators.
    seed
        Reproducibility seed for the k-means initialisation.
    built_up_channel
        Index of the built-up channel (default 0 = ESA WorldCover).

    Returns
    -------
    networkx.MultiGraph
        Nodes carry ``px_r``, ``px_c`` integer pixel attributes plus
        ``voltage_class`` and ``load_share``; edges carry ``length_px``.
    """
    bu = cond_cube[built_up_channel]
    H, W = bu.shape
    mask = bu > LOAD_THRESHOLD
    rows, cols = np.where(mask)
    if rows.size == 0:
        # No load detected; return empty.
        return nx.MultiGraph()

    points = np.column_stack([rows, cols]).astype(np.float64)
    weights = np.ones(rows.size, dtype=np.float64)  # built-up = present/absent

    n_target = max(2, min(n_substations, int(rows.size)))
    centres, labels, masses = _weighted_kmeans(
        points, weights, n_target, seed=seed,
    )
    total_load = float(masses.sum())

    g = nx.MultiGraph()
    for i, (cr, cc) in enumerate(centres):
        cls = _voltage_class(float(masses[i]), total_load=total_load)
        g.add_node(
            i,
            px_r=int(round(cr)),
            px_c=int(round(cc)),
            voltage_class=cls,
            load_share=float(masses[i] / total_load) if total_load > 0 else 0.0,
        )

    raw_edges = _knn_edges(centres, K_NEAREST_NEIGHBOURS)
    edges = _prune_long_edges(raw_edges, alpha=ALPHA_PRUNE)
    for a, b, length in edges:
        g.add_edge(a, b, length_px=float(length))

    return g


def write_panel_graphs(
    cities: list[str],
    cache_dir: Path,
    truth_dir: Path,
    out_dir: Path,
    *,
    seed: int = 0,
    samples_per_city: int = 1,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run Birchfield on each city in ``cities`` and write GraphML samples.

    For each city, reads ``{cache_dir}/{city}.npz`` for the conditioning
    cube and ``{truth_dir}/{city}.graphml`` for the truth-graph
    cardinality (used as the n_substations calibration). Writes
    ``{out_dir}/{city}_{idx:02d}.graphml`` for idx in 0..samples_per_city-1.

    Birchfield's algorithm is deterministic up to the k-means
    initialisation; multiple samples use different seeds to capture
    that single source of stochasticity.

    Returns a per-city diagnostics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: dict[str, dict] = {}
    for city in cities:
        cache_path = cache_dir / f"{city}.npz"
        truth_path = truth_dir / f"{city}.graphml"
        if not cache_path.exists():
            if verbose:
                print(f"  [skip] {city}: missing {cache_path.name}")
            continue
        if not truth_path.exists():
            if verbose:
                print(f"  [skip] {city}: missing {truth_path.name}")
            continue
        truth = nx.read_graphml(truth_path)
        n_truth = truth.number_of_nodes()
        if n_truth < 10:
            if verbose:
                print(f"  [skip] {city}: truth too small ({n_truth} nodes)")
            continue
        z = np.load(cache_path)
        cond = z["cond"].astype(np.float32)

        diags_per_sample: list[dict] = []
        for idx in range(samples_per_city):
            g = build_birchfield_graph(
                cond, n_truth, seed=seed + idx,
            )
            out_path = out_dir / f"{city}_{idx:02d}.graphml"
            nx.write_graphml(g, out_path)
            diags_per_sample.append({
                "sample_idx": idx,
                "nodes": g.number_of_nodes(),
                "edges": g.number_of_edges(),
                "components": nx.number_connected_components(g),
            })
        diagnostics[city] = {
            "n_truth": n_truth,
            "samples": diags_per_sample,
        }
        if verbose:
            mean_n = np.mean([s["nodes"] for s in diags_per_sample])
            mean_e = np.mean([s["edges"] for s in diags_per_sample])
            mean_c = np.mean([s["components"] for s in diags_per_sample])
            print(
                f"  [{city:>15s}]  truth |V|={n_truth:4d}  "
                f"birchfield mean(|V|, |E|, components) = "
                f"({mean_n:.0f}, {mean_e:.0f}, {mean_c:.1f})"
            )
    return diagnostics

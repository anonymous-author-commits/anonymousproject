"""Mask → NetworkX multigraph extraction.

Pipeline:
    binary HV mask  →  morphological skeletonisation  →  pixel-graph
                    →  contract degree-2 chains       →  multigraph
                    →  snap nodes to OSM substations  →  μ_G sample

The returned graph carries node attributes (px coords, lon/lat if
provided) and edge attributes (length in pixels, polyline geometry).

Used in W2 (build μ_G samples per checkpoint × city) and W4 (compute
W₁ between μ_G and μ_*).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import networkx as nx

# skimage is the canonical skeletoniser; falls back to a tiny fixed-iters
# routine if scikit-image is not available.
try:
    from skimage.morphology import skeletonize as _skimage_skeletonize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def _skeletonize_fallback(mask: np.ndarray, iters: int = 16) -> np.ndarray:
    """Lewis-Zhang-Suen-style thinning, deterministic but slower.

    Used only when scikit-image is unavailable. The skimage path is
    preferred; this is here so the package remains import-safe in
    minimal environments.
    """
    m = (mask > 0).astype(np.uint8)
    for _ in range(iters):
        prev = m.copy()
        # 4-neighbour erosion: remove a foreground pixel if it has any
        # 0-valued 4-neighbour (boundary peel).
        n = np.zeros_like(m)
        n[1:, :] += m[:-1, :]
        n[:-1, :] += m[1:, :]
        n[:, 1:] += m[:, :-1]
        n[:, :-1] += m[:, 1:]
        keep = (m == 1) & (n == 4)  # interior pixel — keep
        m = keep.astype(np.uint8) | (m & ~((m == 1) & (n < 4)).astype(np.uint8))
        if (m == prev).all():
            break
    return m


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """Binary skeletonisation. mask: 2-D array, any numeric type.

    Returns a uint8 array of the same shape, 1 on the skeleton, 0 off.
    """
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if _HAS_SKIMAGE:
        return _skimage_skeletonize(binary).astype(np.uint8)
    return _skeletonize_fallback(binary)


# ── Pixel graph → simplified multigraph ────────────────────────────────


_NEIGH8 = [(-1, -1), (-1, 0), (-1, 1),
           ( 0, -1),          ( 0, 1),
           ( 1, -1), ( 1, 0), ( 1, 1)]


def _pixel_graph(skel: np.ndarray) -> nx.Graph:
    """Build an 8-connected graph on the skeleton pixels."""
    g = nx.Graph()
    rows, cols = np.where(skel > 0)
    coords = list(zip(rows.tolist(), cols.tolist()))
    coord_set = set(coords)
    for r, c in coords:
        g.add_node((r, c), px=(r, c))
    for r, c in coords:
        for dr, dc in _NEIGH8:
            nb = (r + dr, c + dc)
            if nb in coord_set and nb > (r, c):  # avoid duplicate edges
                # diagonal length √2 in pixel units
                length = float(np.sqrt(dr * dr + dc * dc))
                g.add_edge((r, c), nb, weight=length)
    return g


def _node_degrees(g: nx.Graph) -> dict:
    return dict(g.degree())


def _contract_chains(g: nx.Graph) -> nx.MultiGraph:
    """Collapse chains of degree-2 pixels into single edges.

    Endpoints (degree 1) and junctions (degree ≥ 3) are kept; the
    intermediate degree-2 pixels become the edge polyline (stored as
    ``geom`` on the multigraph edge), and the edge ``length_px`` sums
    the original pixel weights along the chain.
    """
    deg = _node_degrees(g)
    keep = {n for n, d in deg.items() if d != 2}

    multi = nx.MultiGraph()
    for n in keep:
        multi.add_node(n, px=g.nodes[n]["px"])

    visited_edges: set = set()
    for u in keep:
        for v in g.neighbors(u):
            edge_key = tuple(sorted((u, v)))
            if edge_key in visited_edges:
                continue
            # Walk the chain starting from u → v.
            path = [u, v]
            length = g[u][v]["weight"]
            prev, curr = u, v
            while curr not in keep:
                # curr is degree-2; pick the neighbour that isn't prev.
                nbrs = [x for x in g.neighbors(curr) if x != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                length += g[curr][nxt]["weight"]
                path.append(nxt)
                prev, curr = curr, nxt
            # Mark every consecutive pair in the path as visited.
            for a, b in zip(path[:-1], path[1:]):
                visited_edges.add(tuple(sorted((a, b))))
            if curr in keep:
                multi.add_edge(u, curr, length_px=float(length), geom=path)
    return multi


@dataclass
class GraphExtractionResult:
    graph: nx.MultiGraph
    n_nodes: int
    n_edges: int
    total_length_px: float
    n_components: int


def extract_graph(mask: np.ndarray, *, prune_below_px: float = 4.0) -> GraphExtractionResult:
    """Mask → simplified multigraph.

    Parameters
    ----------
    mask
        2-D binary array (HV positive).
    prune_below_px
        Drop edges shorter than this many pixels — these are usually
        skeletonisation hairs at junctions.
    """
    skel = skeletonize(mask)
    pg = _pixel_graph(skel)
    multi = _contract_chains(pg)

    if prune_below_px > 0:
        to_drop = [
            (u, v, k) for u, v, k, d in multi.edges(keys=True, data=True)
            if d.get("length_px", 0.0) < prune_below_px
        ]
        for u, v, k in to_drop:
            multi.remove_edge(u, v, key=k)
        # Drop any node now isolated.
        isolated = [n for n in multi.nodes if multi.degree(n) == 0]
        multi.remove_nodes_from(isolated)

    total_len = sum(d.get("length_px", 0.0) for _, _, d in multi.edges(data=True))
    n_comp = nx.number_connected_components(multi)
    return GraphExtractionResult(
        graph=multi,
        n_nodes=multi.number_of_nodes(),
        n_edges=multi.number_of_edges(),
        total_length_px=float(total_len),
        n_components=int(n_comp),
    )


def graph_signature(g: nx.MultiGraph, *, n_bins: int = 16,
                     scale_invariant: bool = False) -> np.ndarray:
    """Fixed-length feature vector for W₁ embedding.

    Concatenation of:
      - log-spaced edge-length histogram (n_bins)
      - degree histogram for degrees 1..6 (6)
      - scalars: n_nodes, n_edges, n_components, mean_edge_length,
                 total_length, largest_component_fraction (6)

    Total dimension = n_bins + 12.

    Parameters
    ----------
    scale_invariant
        If True, replace the absolute-magnitude scalars (node/edge
        counts, total length) with scale-invariant ratios and
        log-rescaled magnitudes. This matters when the panel includes
        graphs of wildly different sizes (DiGress at ~30-50 nodes vs.
        truth ~100 nodes vs. some baselines at ~1000 nodes); the
        default scaling tunes the constants to truth scale and hence
        treats large generators as if they were enormous outliers in
        signature space, which overstates their cross-city variance.
        Used by F5.6 to make the truth-vs-generator memorisation
        diagnostic commensurate.
    """
    if g.number_of_edges() == 0:
        return np.zeros(n_bins + (13 if scale_invariant else 12),
                         dtype=np.float32)

    edge_lengths = np.array(
        [d.get("length_px", 0.0) for _, _, d in g.edges(data=True)],
        dtype=np.float64,
    )
    edge_lengths = edge_lengths[edge_lengths > 0]

    log_lo, log_hi = 0.0, max(np.log10(edge_lengths.max() + 1e-9), 4.0)
    edges = np.logspace(log_lo, log_hi, n_bins + 1)
    hist, _ = np.histogram(edge_lengths, bins=edges)
    hist = hist.astype(np.float32) / max(hist.sum(), 1)

    deg_hist = np.zeros(6, dtype=np.float32)
    for _, d in g.degree():
        if 1 <= d <= 6:
            deg_hist[d - 1] += 1
    deg_hist /= max(deg_hist.sum(), 1)

    components = list(nx.connected_components(g))
    largest = max((len(c) for c in components), default=0)
    largest_frac = largest / max(g.number_of_nodes(), 1)

    if scale_invariant:
        # Scale-invariant scalars: log-magnitudes plus pure ratios.
        # log-rescaling compresses the dynamic range; ratios are
        # already O(1). This block keeps the truth-vs-generator
        # memorisation diagnostic commensurate across orders-of-
        # magnitude differences in graph size.
        n_nodes = g.number_of_nodes()
        n_edges = g.number_of_edges()
        edges_per_node = n_edges / max(n_nodes, 1)
        comp_frac = len(components) / max(n_nodes, 1)
        scalars = np.array([
            np.log10(max(n_nodes, 1)) / 4.0,
            np.log10(max(n_edges, 1)) / 4.0,
            comp_frac,
            np.log10(max(float(edge_lengths.mean()), 1.0)) / 3.0,
            np.log10(max(float(edge_lengths.sum()), 1.0)) / 6.0,
            largest_frac,
            edges_per_node / 4.0,
        ], dtype=np.float32)
        return np.concatenate([hist, deg_hist, scalars])

    # Scalars normalised so each component is order-1 and commensurate
    # with the (already-normalised) histogram bins. The constants here
    # are typical scales for HV graphs at 1536² @ 10 m/px (truth: ~100
    # nodes, ~100 edges, ~10 km total length).
    scalars = np.array([
        g.number_of_nodes() / 100.0,
        g.number_of_edges() / 100.0,
        len(components) / 50.0,
        float(edge_lengths.mean()) / 100.0,
        float(edge_lengths.sum()) / 10000.0,
        largest_frac,
    ], dtype=np.float32)

    return np.concatenate([hist, deg_hist, scalars])


def save_graphml(g: nx.MultiGraph, path: Path) -> None:
    """Write a multigraph to GraphML, stringifying complex node ids."""
    h = nx.MultiGraph()
    id_map = {n: f"{i}" for i, n in enumerate(g.nodes)}
    for n, d in g.nodes(data=True):
        h.add_node(id_map[n], px_r=int(d["px"][0]), px_c=int(d["px"][1]))
    for u, v, d in g.edges(data=True):
        h.add_edge(
            id_map[u], id_map[v],
            length_px=float(d.get("length_px", 0.0)),
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(h, str(path))

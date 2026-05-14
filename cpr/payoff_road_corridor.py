"""W6 -- Road-corridor-constrained routing payoff.

Realistic transmission-line construction follows existing road
corridors (rights-of-way already exist; permitting is faster;
construction is cheaper). A generator that produces grid lines
following road corridors is more deployable than one that draws
straight lines through unpermitted terrain.

This payoff penalises edges that fail to follow roads:

    pi_road(a, theta) = - sum_{e in MST(theta) + taps}
                          length(e) * (1 + beta * (1 - frac_on_road(e)))

with beta = 2.0 (an off-road edge costs 3x an on-road edge of the
same length). frac_on_road(e) is computed by sampling N_SAMPLE = 10
points along edge e and looking up the road raster at each.

The road raster is channel index 4 of the 6-channel conditioning
cube (built_up, water, substations, elevation, roads, rails) in
DUPT/data/cache_hv_v2/{city}.npz.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import networkx as nx

from .payoff import PayoffConfig
from .payoff_routing import precompute_components

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
ROAD_CACHE: dict[str, np.ndarray] = {}
ROAD_CHANNEL_INDEX = 4   # built_up, water, subs, elev, roads, rails
N_SAMPLE = 10
BETA_OFF_ROAD = 2.0


def _load_road_raster(city: str) -> np.ndarray:
    if city in ROAD_CACHE:
        return ROAD_CACHE[city]
    p = REPO_ROOT / "DUPT" / "data" / "cache_hv_v2" / f"{city}.npz"
    if not p.exists():
        ROAD_CACHE[city] = np.zeros((1536, 1536), dtype=np.float32)
        return ROAD_CACHE[city]
    cond = np.load(p)["cond"].astype(np.float32)
    if cond.shape[0] >= ROAD_CHANNEL_INDEX + 1:
        roads = cond[ROAD_CHANNEL_INDEX]
    else:
        roads = np.zeros(cond.shape[1:], dtype=np.float32)
    # Normalise: cache may store roads in [-1, 1] -> shift to [0, 1]
    if roads.min() < 0:
        roads = (roads + 1.0) * 0.5
    ROAD_CACHE[city] = roads
    return roads


def _frac_on_road(coord_a: np.ndarray, coord_b: np.ndarray,
                   roads: np.ndarray) -> float:
    """Fraction of N_SAMPLE points along (a,b) that hit road > 0.5."""
    H, W = roads.shape
    ts = np.linspace(0.0, 1.0, N_SAMPLE)
    rs = coord_a[0] * (1 - ts) + coord_b[0] * ts
    cs = coord_a[1] * (1 - ts) + coord_b[1] * ts
    rs = np.clip(rs.astype(int), 0, H - 1)
    cs = np.clip(cs.astype(int), 0, W - 1)
    on = float((roads[rs, cs] > 0.5).mean())
    return on


def _component_mst_road_weighted_cost(
    component_graph: nx.Graph, roads: np.ndarray,
    beta: float = BETA_OFF_ROAD,
) -> float:
    """Sum of MST edge lengths weighted by road-following discount."""
    if component_graph.number_of_edges() == 0:
        return 0.0
    # Weight each edge with road-aware cost.
    weighted = nx.Graph()
    for u, v, d in component_graph.edges(data=True):
        coord_u = np.asarray(u, dtype=np.float64)
        coord_v = np.asarray(v, dtype=np.float64)
        base_len = float(d.get("length_px",
                                np.linalg.norm(coord_u - coord_v)))
        on = _frac_on_road(coord_u, coord_v, roads)
        weighted_len = base_len * (1.0 + beta * (1.0 - on))
        if weighted.has_edge(u, v):
            existing = weighted[u][v].get("weight", float("inf"))
            if weighted_len < existing:
                weighted[u][v]["weight"] = weighted_len
        else:
            weighted.add_edge(u, v, weight=weighted_len)
    if weighted.number_of_edges() == 0:
        return 0.0
    mst = nx.minimum_spanning_tree(weighted, weight="weight")
    return sum(d["weight"] for _, _, d in mst.edges(data=True))


def road_corridor_payoff(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    city: str,
    cfg: PayoffConfig = PayoffConfig(),
    cache: dict | None = None,
) -> float:
    """Negative road-aware routing cost (MST + tap), per node."""
    if theta.number_of_nodes() == 0:
        return -cfg.beta * cfg.unreachable_distance

    sites = np.asarray([np.asarray(s, dtype=np.float64) for s in action])
    if sites.size == 0:
        return -cfg.beta * float(theta.number_of_nodes()) \
                * cfg.unreachable_distance / 1000.0
    if sites.ndim == 1:
        sites = sites.reshape(1, -1)

    info = cache if cache is not None else precompute_components(theta)
    roads = _load_road_raster(city)
    components = list(nx.connected_components(theta))
    total = 0.0
    for comp_nodes in components:
        sub = theta.subgraph(comp_nodes)
        # Use the simple-graph projection so MST sees one edge per pair.
        proj = nx.Graph()
        for u, v, d in sub.edges(data=True):
            le = float(d.get("length_px", 0.0))
            if proj.has_edge(u, v):
                if le < proj[u][v]["weight"]:
                    proj[u][v]["weight"] = le
                    proj[u][v]["length_px"] = le
            else:
                proj.add_edge(u, v, weight=le, length_px=le)
        total += _component_mst_road_weighted_cost(proj, roads)
        # Tap distance: nearest action site to component (off-road
        # by definition; no road discount).
        coords = np.asarray(list(comp_nodes), dtype=np.float64)
        diffs = coords[:, None, :] - sites[None, :, :]
        d2 = np.einsum("nki,nki->nk", diffs, diffs)
        tap_d = float(np.sqrt(d2.min()))
        total += tap_d * (1.0 + BETA_OFF_ROAD)  # tap is fully off-road

    if cfg.normalize and theta.number_of_nodes() > 0:
        total /= theta.number_of_nodes()
    return -total

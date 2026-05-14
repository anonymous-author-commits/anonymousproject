"""Allocation payoff π(a, θ) for the K-substation siting problem.

The decision problem (paper §3):
    Action  a ∈ A : choose K nodes from V(θ) as substation sites.
    State   θ     : the realised graph topology + per-node demand.
    Payoff  π(a, θ) = − ( wire-length-cost(a, θ) + β · unserved-load(a, θ) )

where:
  - wire-length-cost(a, θ) = sum over demand nodes v of the
    shortest-path graph distance from v to the nearest substation in a.
  - unserved-load(a, θ) = sum of demand of nodes in components of θ
    that contain no substation in a (i.e., physically unreachable).

This is a deliberately simple, Lipschitz-continuous payoff: small
perturbations of θ (edge added/removed, weights perturbed) shift π
boundedly. The Lipschitz constant L is verified numerically in
``lipschitz_constant``.

Note: paper Theorem 1 requires Lipschitzness of π in θ for the
W₁ bound to apply. We use the graph-signature embedding from
``graphs.graph_signature`` as the metric on θ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import networkx as nx

from .graphs import graph_signature


# Default unserved-load penalty multiplier.
DEFAULT_BETA = 100.0


@dataclass(frozen=True)
class PayoffConfig:
    K: int = 4                     # number of substations to site
    beta: float = DEFAULT_BETA     # unserved-load penalty
    demand_uniform: bool = True    # uniform per-node demand if True
    unreachable_distance: float = 1e6
    # Per-node maintenance cost on the realised topology. Charges
    # alpha for every node in θ regardless of action, preventing
    # degenerate "all-foreground" generators from gaming the
    # wire-length payoff with cheap nearest-node-distance.
    alpha_node: float = 0.0
    # If True, divide the wire-cost sum by |V(θ)| (mean-distance
    # form). The mean-distance payoff is 1-Lipschitz in the node-W1
    # metric; the unnormalised form is |V|-Lipschitz.
    normalize: bool = False


def _node_list(g: nx.MultiGraph) -> list:
    return list(g.nodes())


def _shortest_distance(g: nx.MultiGraph, source: object, weight: str = "length_px") -> dict:
    """Single-source shortest distances; unreachable → +inf."""
    try:
        return nx.single_source_dijkstra_path_length(g, source, weight=weight)
    except nx.NodeNotFound:
        return {}


def _distance_to_set(
    g: nx.MultiGraph,
    sources: Sequence,
    weight: str = "length_px",
) -> dict:
    """Multi-source shortest distances from any node in ``sources``.

    Returns dict node → min distance to ``sources``; unreachable nodes
    are absent from the dict.
    """
    if not sources:
        return {}
    # Use multi-source Dijkstra: insert a virtual super-source with
    # zero-weight edges to each source.
    virtual = ("__src__",)
    h = g.copy()
    h.add_node(virtual)
    for s in sources:
        h.add_edge(virtual, s, length_px=0.0)
    dists = nx.single_source_dijkstra_path_length(h, virtual, weight=weight)
    dists.pop(virtual, None)
    return dists


def allocation_payoff(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
) -> float:
    """Compute π(a, θ) for a *coordinate-based* action space.

    The action ``a`` is a sequence of (row, col) pixel coordinates
    designating K substation sites. The payoff is

        π(a, θ) = − Σ_{v ∈ V(θ)} dist(v, nearest_site_in_a)

    in pixel units. This is θ-dependent (the node positions of θ
    determine the cost), Lipschitz in θ under the
    ``graph_signature`` embedding (the sum bounds with the number of
    nodes and the diameter of the tile, both of which are signature
    components), and well-defined for any graph including empty
    ones.

    Parameters
    ----------
    action
        Sequence of (row, col) coordinates. Length K ≤ cfg.K is OK.
    theta
        Graph with `px` node attribute set to (row, col).
    cfg
        Payoff hyperparameters.
    """
    if theta.number_of_nodes() == 0:
        # Empty graph → zero-node sum, but penalise via cfg.beta
        # to discourage degenerate generators.
        return -cfg.beta * cfg.unreachable_distance

    sites = np.asarray([np.asarray(s, dtype=np.float64) for s in action])
    if sites.size == 0:
        # No sites placed → all demand unserved.
        return -cfg.beta * float(theta.number_of_nodes()) * cfg.unreachable_distance / 1000.0

    nodes = np.asarray([
        np.asarray(theta.nodes[n].get("px", n), dtype=np.float64)
        for n in theta.nodes()
    ])
    if sites.ndim == 1:
        sites = sites.reshape(1, -1)

    # Pairwise distance (N, K), then min over sites per node.
    diffs = nodes[:, None, :] - sites[None, :, :]
    d2 = np.einsum("nki,nki->nk", diffs, diffs)
    nearest = np.sqrt(d2.min(axis=1))
    if cfg.normalize:
        # Mean-distance form: 1-Lipschitz in node-W1.
        wire_cost = float(nearest.mean())
    else:
        wire_cost = float(nearest.sum())
    maintenance = cfg.alpha_node * float(theta.number_of_nodes())
    return -(wire_cost + maintenance)


def lipschitz_constant(
    payoff_fn,
    theta_pairs: Iterable[tuple[nx.MultiGraph, nx.MultiGraph]],
    *,
    actions_per_pair: int = 16,
    action_sampler=None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Empirical Lipschitz estimate of π in the graph-signature metric.

        L̂ = max_{θ, θ', a} |π(a, θ) − π(a, θ')| / d_emb(θ, θ')

    Parameters
    ----------
    payoff_fn
        Callable (action, theta) → float.
    theta_pairs
        Iterable of (theta_a, theta_b) graph pairs.
    actions_per_pair
        Random actions to sample per pair.
    action_sampler
        Callable (rng, theta_a, theta_b) → list[action]. Defaults to a
        coordinate-based grid sampler.
    rng
        Numpy generator for reproducibility.
    """
    rng = rng or np.random.default_rng(0)
    if action_sampler is None:
        action_sampler = _default_action_sampler
    ratios: list[float] = []

    for g1, g2 in theta_pairs:
        sig1 = graph_signature(g1)
        sig2 = graph_signature(g2)
        d_emb = float(np.linalg.norm(sig1 - sig2))
        if d_emb < 1e-9:
            continue

        for _ in range(actions_per_pair):
            action = action_sampler(rng, g1, g2)
            if not action:
                continue
            p1 = payoff_fn(action, g1)
            p2 = payoff_fn(action, g2)
            ratios.append(abs(p1 - p2) / d_emb)

    if not ratios:
        return {"L_max": float("nan"), "L_p95": float("nan"), "ratios": []}
    arr = np.asarray(ratios)
    return {
        "L_max": float(arr.max()),
        "L_p95": float(np.percentile(arr, 95)),
        "L_mean": float(arr.mean()),
        "ratios": arr.tolist(),
        "n_pairs": int(len(arr)),
    }


def _default_action_sampler(
    rng: np.random.Generator,
    g1: nx.MultiGraph,
    g2: nx.MultiGraph,
    *,
    K: int = 4,
    grid_size: int = 16,
    tile_size: int = 1536,
) -> list[tuple[float, float]]:
    """Sample a K-subset from a grid_size × grid_size grid of cell centres."""
    cell = tile_size / grid_size
    cells = [
        ((i + 0.5) * cell, (j + 0.5) * cell)
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    idx = rng.choice(len(cells), size=K, replace=False)
    return [cells[i] for i in idx]


def grid_action_set(
    *,
    K: int = 4,
    grid_size: int = 16,
    tile_size: int = 1536,
    n_actions: int = 200,
    seed: int = 0,
) -> list[list[tuple[float, float]]]:
    """Pre-sample a deterministic action set for CPR experiments."""
    rng = np.random.default_rng(seed)
    cell = tile_size / grid_size
    cells = [
        ((i + 0.5) * cell, (j + 0.5) * cell)
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    seen: set = set()
    out: list[list[tuple[float, float]]] = []
    while len(out) < n_actions:
        idx = tuple(sorted(rng.choice(len(cells), size=K, replace=False)))
        if idx in seen:
            continue
        seen.add(idx)
        out.append([cells[i] for i in idx])
    return out

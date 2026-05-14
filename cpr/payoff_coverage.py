"""Demand-coverage payoff -- canonical-panel third payoff.

Where wire-cost payoff sums per-graph-node nearest-distances and
routing payoff uses theta's MST + tap distance, the demand-
coverage payoff measures how well the chosen substation sites
serve a fixed grid of *demand* points, modulated by whether the
realised topology actually reaches those points.

Formally, for a fixed grid of demand points D = {d_1,..., d_M}
covering the city tile uniformly,

    pi_cov(a, theta) = -sum_{m=1}^{M}
                          [ d(d_m, a)
                          + lambda * d(d_m, V(theta)) ]

where the first term penalises demand points far from any chosen
site, and the second penalises demand points far from any
realised-graph node (so a sparse theta is penalised even if the
sites themselves are well-placed).

This is qualitatively different from both wire-cost and routing:
  - Wire-cost: how well sites cover graph nodes.
  - Routing: how well theta's edges connect graph nodes via sites.
  - Coverage: how well sites serve a fixed demand grid given theta.

Lipschitzness in node-W1: 1-Lipschitz on the second term only;
the first term is independent of theta. Constant-summed
contributions cancel in CPR (analogous to the alpha_node observation
in payoff.py), so the effective Lipschitz constant is
``lambda`` exactly. Set ``lambda`` to the second term's coefficient
and the bound applies directly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import networkx as nx

from .payoff import PayoffConfig


def _demand_grid(grid_size: int = 12, tile_size: int = 1536) -> np.ndarray:
    """Return a (M, 2) array of demand-point coordinates."""
    cell = tile_size / grid_size
    pts = []
    for i in range(grid_size):
        for j in range(grid_size):
            pts.append(((i + 0.5) * cell, (j + 0.5) * cell))
    return np.array(pts, dtype=np.float64)


def coverage_payoff(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    lambda_topology: float = 1.0,
    grid_size: int = 12,
    tile_size: int = 1536,
) -> float:
    """Compute the negative demand-coverage cost.

    Parameters
    ----------
    action : sequence of K (row, col) coordinates -- substation sites.
    theta : the realised graph.
    lambda_topology : weight on the topology-aware second term.
    grid_size : 1-D resolution of the fixed demand grid (M = grid_size^2).
    tile_size : pixel extent of the city tile.
    """
    D = _demand_grid(grid_size=grid_size, tile_size=tile_size)
    if not action:
        return -cfg.beta * cfg.unreachable_distance

    sites = np.asarray(
        [np.asarray(s, dtype=np.float64) for s in action]
    )
    if sites.ndim == 1:
        sites = sites.reshape(1, -1)

    # First term: demand-to-site distances.
    diffs_a = D[:, None, :] - sites[None, :, :]
    d2_a = np.einsum("nki,nki->nk", diffs_a, diffs_a)
    site_dist = np.sqrt(d2_a.min(axis=1))

    # Second term: demand-to-theta-node distances. If theta is empty,
    # add a large penalty proportional to grid size and tile.
    if theta.number_of_nodes() == 0:
        topology_dist = np.full(len(D), tile_size, dtype=np.float64)
    else:
        nodes = np.asarray(
            [np.asarray(theta.nodes[n].get("px", n), dtype=np.float64)
             for n in theta.nodes()]
        )
        diffs_t = D[:, None, :] - nodes[None, :, :]
        d2_t = np.einsum("nki,nki->nk", diffs_t, diffs_t)
        topology_dist = np.sqrt(d2_t.min(axis=1))

    cost_per_demand = site_dist + lambda_topology * topology_dist
    if cfg.normalize:
        total = float(cost_per_demand.mean())
    else:
        total = float(cost_per_demand.sum())
    return -total


def coverage_payoff_interactive(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    grid_size: int = 12,
    tile_size: int = 1536,
) -> float:
    """Interactive coverage payoff -- graph nodes act as auxiliary sites.

    The additive variant ``coverage_payoff`` decomposes as
    f(a) + g(theta), making the optimal action independent of theta
    and CPR identically zero. This interactive form removes that
    pathology by treating action sites and graph nodes as members of
    a single 'service set' S = a U V(theta):

        pi_int(a, theta) = -mean_{d in D} min_{p in S} ||d - p||_2.

    A demand point is served by whichever is closest -- a built
    substation or a graph node. A model that produces a topology
    with useful nodes near demand points pays a smaller wire-cost
    even before any sites are placed; conversely, a sparse model
    must rely entirely on the K sites. Hence the optimal a* now
    *depends* on theta, and CPR is generically non-zero.

    Lipschitzness in node-W1: 1-Lipschitz, by the same min-of-
    distances argument as the wire-cost payoff. The Theorem-1
    bound applies with L = 1.
    """
    from .payoff_coverage import _demand_grid

    D = _demand_grid(grid_size=grid_size, tile_size=tile_size)
    sites = np.asarray([np.asarray(s, dtype=np.float64) for s in action])
    if sites.ndim == 1:
        sites = sites.reshape(1, -1)

    if theta.number_of_nodes() > 0:
        nodes = np.asarray(
            [np.asarray(theta.nodes[n].get("px", n), dtype=np.float64)
             for n in theta.nodes()]
        )
        service = np.vstack([sites, nodes]) if sites.size else nodes
    else:
        service = sites

    if service.size == 0:
        # Degenerate: no service points at all.
        return -cfg.beta * cfg.unreachable_distance

    diffs = D[:, None, :] - service[None, :, :]
    d2 = np.einsum("nki,nki->nk", diffs, diffs)
    nearest = np.sqrt(d2.min(axis=1))
    if cfg.normalize:
        total = float(nearest.mean())
    else:
        total = float(nearest.sum())
    return -total

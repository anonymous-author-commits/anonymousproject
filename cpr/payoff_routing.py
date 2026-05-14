"""Line-routing payoff: minimum-spanning-forest cost over (theta + a).

Where the wire-cost payoff treats each graph node independently
(each contributes its nearest-site distance), the routing payoff
charges for *connecting* the realised topology to the substation
sites: each connected component of theta pays the MST cost over
the component itself, plus the cost of one tap edge from the
component to its nearest substation site.

Formally, for action a = (s_1,..., s_K) and graph theta with
components {C_1,..., C_M}, the routing payoff is

    pi(a, theta) = - sum_{m=1}^{M}
                       [ MST(C_m).total_length
                       + min_{v in V(C_m), k} ||v - s_k||_2 ].

The payoff uses theta's existing edges (the lines DUPT predicted)
to amortise routing across nearby nodes, so it is genuinely
sensitive to topology rather than just node positions. This
distinguishes it from the wire-cost payoff and makes the
"different payoffs give different rankings" claim non-vacuous.

Lipschitzness: in the node-W1 metric, the routing payoff is
K + bounded-by-MST-degree-Lipschitz (per-component MST is stable
to coordinate perturbations under the 1-Lipschitz tap-distance
plus the per-edge length sum). For the empirical analysis we
treat the routing-payoff Lipschitz constant L_route as a
quantity to be estimated from data, not analytical.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import networkx as nx

from .payoff import PayoffConfig


def precompute_components(theta: nx.MultiGraph) -> dict:
    """Precompute MST length and node-coords per component.

    Routing payoff invokes per-action evaluation many times on the
    same theta; the MST is action-independent and can be computed
    once. Returns a dict with:
      total_mst_len (sum across components),
      n_nodes (total),
      comp_coords (list of (N_i, 2) arrays of component node coords).
    """
    total_mst = 0.0
    comp_coords: list[np.ndarray] = []
    for comp in nx.connected_components(theta):
        sub = nx.Graph(theta.subgraph(comp))
        try:
            mst = nx.minimum_spanning_tree(sub, weight="length_px")
            total_mst += sum(
                float(d.get("length_px", 0.0)) for _, _, d in mst.edges(data=True)
            )
        except Exception:
            pass
        coords = np.asarray(
            [np.asarray(theta.nodes[n].get("px", n), dtype=np.float64)
             for n in comp]
        )
        if coords.size > 0:
            comp_coords.append(coords)
    return {
        "total_mst_len": total_mst,
        "n_nodes": theta.number_of_nodes(),
        "comp_coords": comp_coords,
    }


def routing_mst_payoff(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    cache: dict | None = None,
) -> float:
    """Compute the negative MST + tap-distance routing cost.

    Pass ``cache`` from ``precompute_components(theta)`` if calling
    multiple times with the same theta to avoid the per-call MST
    recomputation (which dominates runtime on dense graphs).
    """
    if theta.number_of_nodes() == 0:
        return -cfg.beta * cfg.unreachable_distance

    sites = np.asarray([np.asarray(s, dtype=np.float64) for s in action])
    if sites.size == 0:
        return -cfg.beta * float(theta.number_of_nodes()) * cfg.unreachable_distance / 1000.0
    if sites.ndim == 1:
        sites = sites.reshape(1, -1)

    info = cache if cache is not None else precompute_components(theta)
    total = info["total_mst_len"]

    for coords in info["comp_coords"]:
        diffs = coords[:, None, :] - sites[None, :, :]
        d2 = np.einsum("nki,nki->nk", diffs, diffs)
        tap_d = float(np.sqrt(d2.min()))
        total += tap_d

    if cfg.normalize and info["n_nodes"] > 0:
        total /= info["n_nodes"]

    return -total

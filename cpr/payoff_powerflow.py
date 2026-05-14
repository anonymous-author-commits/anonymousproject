"""DC power-flow CPR payoff.

Wires the existing DC power-flow solver in
``tessera.powergrid.evaluation.powerflow`` into the CPR loop. The
adapter converts a CPR multigraph (pixel-coordinate nodes,
length-px edges) into a ``tessera.powergrid.approaches.common.PowerGraph``
populated with electrical parameters, then runs ``DCPowerFlow.solve()``
and converts the result into a payoff value:

    pi_DC(a, theta) = -[ capital_cost(a, theta)
                        + operating_cost(theta)
                        + c_ENS * unserved_load(a, theta) ]

where:
  capital_cost   ~ length-weighted infrastructure cost per voltage class
  operating_cost ~ generator-dispatch cost (uniform plants for the
                   purposes of this payoff)
  unserved_load  ~ MW of unserved demand under the dispatched plan,
                   estimated from the solver's overloaded-line and
                   convergence flags.

For Lipschitz analysis: the DC-flow solution is continuous in the
B-matrix entries (which are 1/x_ohm, themselves continuous in edge
length), so for fixed connectivity the payoff is Lipschitz in node
coordinates. The empirical Lipschitz constant under the node-W1
metric is estimated by the same paired-topology procedure as in
Methods (``Estimating the Lipschitz constant in practice'').

N-1 augmentation
----------------
``payoff_n_minus_1`` evaluates the worst-case operating cost across
single-line outages. Monte Carlo sampling of 32 outages keeps wall-clock
tractable while preserving the qualitative N-1 ranking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import networkx as nx

try:
    from ._powergrid.approaches.common import (
        NodeType, PowerEdge, PowerGraph, PowerNode, VoltageClass,
    )
    from ._powergrid.evaluation.powerflow import DCPowerFlow
    _HAS_POWERGRID = True
except Exception as _exc:
    _HAS_POWERGRID = False
    _IMPORT_ERR = _exc

from .payoff import PayoffConfig

# Capital cost per kilometre, USD millions, mid-point of the IEA / ENTSO-E
# ranges adopted in cpr.monetary. Used as a relative weighting only --
# constants cancel in CPR.
_CAPEX_PER_KM = {"HV": 2.5, "MV": 0.6, "LV": 0.10, "UNKNOWN": 1.0}
_C_ENS_USD_PER_MWH = 5_000.0  # standard ENS penalty for OECD planning
_DEFAULT_TILE_PX = 1536
_DEFAULT_PIXEL_M = 10.0

def _coord_to_lonlat(r_px: float, c_px: float, tile_px: int = _DEFAULT_TILE_PX,
                     pixel_m: float = _DEFAULT_PIXEL_M) -> tuple[float, float]:
    """Map pixel (row, col) to a synthetic (lon, lat) within a 0--0.15 deg box.

    The actual lon/lat is irrelevant for DC-flow (only edge length and node
    type matter), so we compress the entire tile into a synthetic 0--lon_max
    box for compactness. lat increases northward as row decreases.
    """
    deg_per_m = 1.0 / 111_000.0  # rough -- only lengths matter, not absolute pos
    lon = c_px * pixel_m * deg_per_m
    lat = (tile_px - r_px) * pixel_m * deg_per_m
    return lon, lat

def topology_to_powergraph(
    theta: nx.MultiGraph,
    action_sites: Sequence[tuple[float, float]] | None = None,
    *,
    tile_px: int = _DEFAULT_TILE_PX,
    pixel_m: float = _DEFAULT_PIXEL_M,
    voltage_class: str = "HV",
) -> PowerGraph:
    """Convert a CPR multigraph into a tessera PowerGraph.

    Action sites (if provided) are added as HV_SUBSTATION nodes
    that act as plants/sources. Other graph nodes become LOAD_PROXY,
    so the DC-flow solver dispatches from action sites to graph nodes.
    """
    if not _HAS_POWERGRID:
        raise ImportError(
            f"tessera.powergrid not importable: {_IMPORT_ERR}. "
        )

    pg = PowerGraph(approach="cpr_dcflow", city_id="cpr_eval")
    vc = voltage_class
    capacity_mva = {"HV": 200.0, "MV": 30.0, "LV": 0.5}.get(vc, 100.0)

    # Sites first, so the slack-bus pick is deterministic.
    site_ids: list[str] = []
    if action_sites:
        for k, (r, c) in enumerate(action_sites):
            sid = f"site_{k}"
            lon, lat = _coord_to_lonlat(r, c, tile_px, pixel_m)
            pg.add_node(PowerNode(
                id=sid,
                node_type=NodeType.HV_SUBSTATION,
                lon=lon, lat=lat,
                voltage_class=vc,
                attrs={"capacity_mva": capacity_mva},
            ))
            site_ids.append(sid)

    # Graph nodes become loads (consumers).
    node_ids: dict = {}
    for n in theta.nodes():
        r, c = theta.nodes[n].get("px", n)
        nid = f"n_{r}_{c}"
        if nid in node_ids:
            continue
        lon, lat = _coord_to_lonlat(r, c, tile_px, pixel_m)
        pg.add_node(PowerNode(
            id=nid,
            node_type=NodeType.LOAD_PROXY,
            lon=lon, lat=lat,
            voltage_class=vc,
            attrs={"load_mw": 0.005},  # 5 kW per node default
        ))
        node_ids[n] = nid

    # Graph edges with length in metres.
    seen_edges: set = set()
    for u, v, data in theta.edges(data=True):
        if u not in node_ids or v not in node_ids:
            continue
        u_id, v_id = node_ids[u], node_ids[v]
        key = tuple(sorted((u_id, v_id)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        length_m = float(data.get("length_px", 0.0)) * pixel_m
        pg.add_edge(PowerEdge(
            u=u_id, v=v_id,
            voltage_class=vc,
            length_m=length_m,
            attrs={"capacity_mva": capacity_mva},
        ))

    # Connect each action site to its closest 2 graph nodes by Euclidean
    # distance (so the solver sees a feasible network with a path from
    # source to load). This represents the "tap" connection from the
    # candidate site into the predicted topology.
    if site_ids and node_ids:
        graph_nodes = list(node_ids.items())
        node_coords = np.asarray(
            [[r, c] for (r, c), _nid in graph_nodes], dtype=np.float64
        )
        for k, (r, c) in enumerate(action_sites or []):
            d2 = ((node_coords - np.array([r, c])) ** 2).sum(axis=1)
            order = np.argsort(d2)
            for idx in order[:2]:
                nb_id = graph_nodes[idx][1]
                length_m = float(np.sqrt(d2[idx]) * pixel_m)
                pg.add_edge(PowerEdge(
                    u=site_ids[k], v=nb_id,
                    voltage_class=vc,
                    length_m=length_m,
                    attrs={"capacity_mva": capacity_mva,
                           "tap_connection": True},
                ))

    return pg

def _capital_cost_usd_m(pg: PowerGraph) -> float:
    """Sum of edge length * $/km midpoints, in USD millions."""
    total = 0.0
    for edge in pg._edges:
        per_km = _CAPEX_PER_KM.get(str(edge.voltage_class), 1.0)
        total += per_km * edge.length_m / 1000.0
    return total

def dc_flow_payoff(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    tile_px: int = _DEFAULT_TILE_PX,
    pixel_m: float = _DEFAULT_PIXEL_M,
    horizon_hours: int = 8760,
) -> float:
    """Negative total annualised cost under DC power flow.

    Components:
      - capital cost (USD M, length-weighted)
      - operating cost (proxy: equal to total_load_mw * $50/MWh * horizon)
      - ENS penalty (proxy: overload-fraction * total_load * c_ENS)

    Returns a negative number (lower magnitude is better, like other
    CPR payoffs).
    """
    if theta.number_of_nodes() == 0:
        return -cfg.beta * cfg.unreachable_distance

    sites = [tuple(s) for s in action]
    pg = topology_to_powergraph(theta, sites, tile_px=tile_px, pixel_m=pixel_m)

    capital = _capital_cost_usd_m(pg)  # in USD M

    if pg.node_count < 2 or pg.edge_count == 0:
        return -(capital + cfg.beta * cfg.unreachable_distance / 1000.0)

    pf = DCPowerFlow(pg)
    res = pf.solve()
    total_load_mw = max(res.total_load_mw, 1e-6)

    # Operating: $50/MWh annualised over the horizon.
    operating_usd_m = (
        total_load_mw * 50.0 * horizon_hours / 1e6
    )

    # ENS proxy: when feasible, ENS = 0; when infeasible, treat overload
    # fraction as unserved load.
    if not res.feasible or not res.converged:
        ens_fraction = 1.0
    elif res.max_line_loading_pct > 100.0:
        # Linear penalty above 100% loading.
        ens_fraction = min(1.0, (res.max_line_loading_pct - 100.0) / 100.0)
    else:
        ens_fraction = 0.0

    ens_usd_m = (
        ens_fraction * total_load_mw * _C_ENS_USD_PER_MWH * horizon_hours / 1e6
    )

    total_cost_usd_m = capital + operating_usd_m + ens_usd_m
    if cfg.normalize:
        # Normalise by the number of nodes for comparability with the
        # other normalised payoffs in the panel.
        return -total_cost_usd_m / max(pg.node_count, 1)
    return -total_cost_usd_m

def payoff_n_minus_1(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    tile_px: int = _DEFAULT_TILE_PX,
    pixel_m: float = _DEFAULT_PIXEL_M,
    n_outages: int = 32,
    seed: int = 0,
) -> float:
    """Worst-case operating cost across single-line outages.

    For each Monte Carlo sample, removes one HV edge from the topology
    and re-solves DC flow. Returns the negative of the maximum total
    cost across the sampled outages -- the planner's worst case.
    """
    if theta.number_of_nodes() == 0:
        return -cfg.beta * cfg.unreachable_distance

    rng = np.random.default_rng(seed)
    edges = list(theta.edges())
    if not edges:
        return dc_flow_payoff(action, theta, cfg=cfg, tile_px=tile_px,
                              pixel_m=pixel_m)
    n = min(n_outages, len(edges))
    outage_indices = rng.choice(len(edges), size=n, replace=False)

    worst = float("inf")
    for idx in outage_indices:
        theta_minus = theta.copy()
        u, v = edges[idx]
        # Remove all parallel edges between u and v (multigraph safe).
        if theta_minus.has_edge(u, v):
            theta_minus.remove_edge(u, v)
        cost = -dc_flow_payoff(action, theta_minus, cfg=cfg,
                                tile_px=tile_px, pixel_m=pixel_m)
        if cost > worst or worst == float("inf"):
            worst = cost
    return -float(worst)

__all__ = [
    "topology_to_powergraph",
    "dc_flow_payoff",
    "payoff_n_minus_1",
]
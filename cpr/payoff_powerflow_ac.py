"""AC-OPF planning payoff.

Mirrors `cpr.payoff_powerflow.dc_flow_payoff` but uses pandapower's
nonlinear AC power flow (Newton-Raphson) instead of the linearised
DC flow. Convergence + overload feasibility under AC tells us
whether the DC ranking generalises to the realistic operating
regime.

The payoff structure is intentionally identical to the DC version
so the two are directly comparable:

    pi_AC(a, theta) = - (capex + op_cost + ENS_penalty)

where the AC-specific bits are:

    - bus voltage limits at 0.95-1.05 pu (reasonable for an HV
      planning study; tighter than the IEEE relaxation but enough
      to let pandapower converge on most cells)
    - reactive power limits at +/- 0.4 * MVA capacity (rule of
      thumb for resistive-dominated load)
    - line thermal limits same as DC (200 MVA HV)
    - non-convergence -> ENS penalty proportional to total load

Convergence rate is reported by `ac_flow_payoff(..., return_meta=True)`.
"""
from __future__ import annotations

from typing import Sequence

import networkx as nx
import numpy as np

try:
    import pandapower as pp
    _HAS_PP = True
except Exception as _exc:
    _HAS_PP = False
    _IMPORT_ERR = repr(_exc)

from .payoff import PayoffConfig

_DEFAULT_TILE_PX = 1536
_DEFAULT_PIXEL_M = 10.0
_C_ENS_USD_PER_MWH = 6_000.0
_C_OP_USD_PER_MWH = 50.0
_HORIZON_HOURS = 8760.0


def _build_pp_net(
    theta: nx.MultiGraph,
    action_sites: Sequence[tuple[float, float]],
    *,
    pixel_m: float = _DEFAULT_PIXEL_M,
    voltage_class: str = "HV",
):
    """Build a pandapower Network from a CPR multigraph + action sites.

    HV is modelled at 110 kV (typical first-tier HV); thermal limit
    converted to current rating via S = sqrt(3)*V*I -> I_max =
    capacity_mva * 1e6 / (sqrt(3)*V_kv*1e3) amperes.
    """
    if not _HAS_PP:
        raise ImportError(f"pandapower not available: {_IMPORT_ERR}")

    capacity_mva = {"HV": 200.0, "MV": 30.0, "LV": 0.5}.get(
        voltage_class, 100.0)
    vn_kv = {"HV": 110.0, "MV": 22.0, "LV": 0.4}.get(voltage_class, 110.0)
    i_ka = capacity_mva / (np.sqrt(3.0) * vn_kv)

    net = pp.create_empty_network()

    # Action sites become buses with a generator at the first; rest are
    # synchronous generators.
    site_buses: list[int] = []
    for k, (r, c) in enumerate(action_sites):
        b = pp.create_bus(net, vn_kv=vn_kv, name=f"site_{k}",
                            min_vm_pu=0.95, max_vm_pu=1.05)
        site_buses.append(b)
    if not site_buses:
        return net, [], {}

    # Graph nodes -> buses + loads.
    node_bus: dict = {}
    total_load_mw = 0.0
    p_per_load = 0.005  # 5 kW per node
    q_per_load = p_per_load * 0.2  # 5:1 P:Q ratio
    for n in theta.nodes():
        if n in node_bus:
            continue
        b = pp.create_bus(net, vn_kv=vn_kv, name=f"node_{n}",
                            min_vm_pu=0.95, max_vm_pu=1.05)
        pp.create_load(net, bus=b, p_mw=p_per_load, q_mvar=q_per_load,
                        name=f"load_{n}")
        node_bus[n] = b
        total_load_mw += p_per_load

    if not node_bus:
        return net, site_buses, node_bus

    # Distribute the total load across the action sites as generators.
    # The first becomes the slack (ext_grid); others are PV gens.
    pp.create_ext_grid(net, bus=site_buses[0], vm_pu=1.0,
                        name=f"ext_grid_site_0")
    per_site_gen_mw = total_load_mw / max(len(site_buses), 1)
    for k, b in enumerate(site_buses):
        cap = capacity_mva
        pp.create_gen(
            net, bus=b, p_mw=per_site_gen_mw,
            vm_pu=1.0, name=f"gen_site_{k}",
            min_p_mw=0.0, max_p_mw=cap,
            min_q_mvar=-0.4 * cap, max_q_mvar=0.4 * cap,
            slack_weight=1.0 if k == 0 else 0.0,
        )

    # Graph edges -> AC lines. Use a generic 110 kV line type.
    seen_edges: set = set()
    edge_count = 0
    for u, v, data in theta.edges(data=True):
        if u not in node_bus or v not in node_bus:
            continue
        if u == v:
            continue
        key = tuple(sorted((node_bus[u], node_bus[v])))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        length_km = max(float(data.get("length_px", 0.0)) * pixel_m
                          / 1000.0, 0.001)
        pp.create_line_from_parameters(
            net, from_bus=node_bus[u], to_bus=node_bus[v],
            length_km=length_km,
            r_ohm_per_km=0.06,    # typical 110 kV ACSR
            x_ohm_per_km=0.30,    # typical 110 kV ACSR
            c_nf_per_km=10.0,
            max_i_ka=i_ka,
            name=f"line_{edge_count}",
        )
        edge_count += 1

    # Tap edges from each action site to its 2 nearest graph nodes.
    if site_buses and node_bus:
        action_arr = np.asarray(action_sites, dtype=np.float64)
        node_coords = np.asarray(
            [list(n) if isinstance(n, tuple) else [0, 0]
             for n in node_bus.keys()], dtype=np.float64,
        )
        node_keys = list(node_bus.keys())
        for k, (r, c) in enumerate(action_sites):
            d2 = ((node_coords - np.array([r, c])) ** 2).sum(axis=1)
            order = np.argsort(d2)
            for idx in order[:2]:
                length_km = max(float(np.sqrt(d2[idx])) * pixel_m / 1000.0,
                                  0.001)
                pp.create_line_from_parameters(
                    net,
                    from_bus=site_buses[k],
                    to_bus=node_bus[node_keys[idx]],
                    length_km=length_km,
                    r_ohm_per_km=0.06,
                    x_ohm_per_km=0.30,
                    c_nf_per_km=10.0,
                    max_i_ka=i_ka,
                    name=f"tap_{k}_{int(idx)}",
                )

    return net, site_buses, node_bus


def ac_flow_payoff(
    action: Sequence,
    theta: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    return_meta: bool = False,
):
    """AC-OPF planning payoff.

    Returns -(capex + op_cost + ENS_penalty) in millions of USD,
    or NaN if the AC flow fails to converge.

    Parameters mirror dc_flow_payoff so the two are interchangeable
    in run_cpr_panel.py.
    """
    if not _HAS_PP:
        return float("nan") if not return_meta else (
            float("nan"), {"converged": False, "reason": "no_pandapower"})
    if theta.number_of_nodes() == 0:
        return -cfg.beta * cfg.unreachable_distance

    sites = [tuple(s) for s in action]
    if not sites:
        return -cfg.beta * float(theta.number_of_nodes()) \
                * cfg.unreachable_distance / 1000.0

    net, site_buses, node_bus = _build_pp_net(theta, sites)
    if not site_buses or not node_bus:
        return float("nan") if not return_meta else (
            float("nan"), {"converged": False,
                            "reason": "empty_network"})

    converged = False
    overloaded = False
    max_loading = 0.0
    try:
        # max_iteration kept tight (5) so non-convergent cells fail
        # fast instead of burning the budget. numba-JIT is opportunistic.
        try:
            pp.runpp(net, max_iteration=5, tolerance_mva=1e-3,
                      numba=True, calculate_voltage_angles=False,
                      init="flat")
        except TypeError:
            pp.runpp(net, max_iteration=5, tolerance_mva=1e-3,
                      numba=False, calculate_voltage_angles=False,
                      init="flat")
        converged = bool(net.converged)
        if converged and not net.res_line.empty:
            max_loading = float(net.res_line["loading_percent"].max())
            overloaded = max_loading > 100.0
    except Exception:
        converged = False

    # Capex: total line length × cost/km.
    total_line_km = float(net.line["length_km"].sum())
    capex_M = total_line_km * 2.5  # HV midpoint USD M / km

    total_load_mw = float(net.load["p_mw"].sum()) if not net.load.empty else 0.0

    if converged and not overloaded:
        op_cost_M = (total_load_mw * _C_OP_USD_PER_MWH * _HORIZON_HOURS
                     / 1e6)
        ens_M = 0.0
    elif converged and overloaded:
        ens_fraction = min(1.0, (max_loading - 100.0) / 100.0)
        op_cost_M = (total_load_mw * _C_OP_USD_PER_MWH * _HORIZON_HOURS
                     / 1e6)
        ens_M = (ens_fraction * total_load_mw * _C_ENS_USD_PER_MWH
                  * _HORIZON_HOURS / 1e6)
    else:
        # Non-convergence: treat as full ENS for the load.
        op_cost_M = 0.0
        ens_M = (total_load_mw * _C_ENS_USD_PER_MWH * _HORIZON_HOURS
                  / 1e6)

    payoff = -(capex_M + op_cost_M + ens_M)
    if cfg.normalize and theta.number_of_nodes() > 0:
        payoff /= theta.number_of_nodes()

    if return_meta:
        return payoff, {
            "converged": converged,
            "overloaded": overloaded,
            "max_loading_pct": max_loading,
            "total_load_mw": total_load_mw,
            "capex_M": capex_M,
            "op_cost_M": op_cost_M,
            "ens_M": ens_M,
        }
    return payoff

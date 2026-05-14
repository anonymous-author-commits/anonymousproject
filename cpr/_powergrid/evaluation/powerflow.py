"""DC power flow analysis for generated power grids.

Implements linearized DC power flow (P = B * θ) to validate that
generated grids can sustain realistic load scenarios without violating
voltage angle or line loading limits.

DC power flow assumptions:
  - Lossless lines (R << X)
  - Flat voltage profile (|V| ≈ 1.0 pu)
  - Small angle differences (sin(θ) ≈ θ)

This is appropriate for operations planning validation — we don't need
full AC power flow accuracy, but we need feasibility checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from ..approaches.common import NodeType, PowerGraph, VoltageClass
from ..demand.profiles import LoadProfile

logger = logging.getLogger(__name__)

# Per-unit base (100 MVA, nominal voltage per class)
_SBASE_MVA = 100.0
_VBASE_KV = {VoltageClass.HV: 220.0, VoltageClass.MV: 20.0, VoltageClass.LV: 0.4}

# Thermal limits by voltage class (MVA)
_LINE_CAPACITY_MVA = {
    VoltageClass.HV: 200.0,
    VoltageClass.MV: 30.0,
    VoltageClass.LV: 0.5,
    VoltageClass.UNKNOWN: 10.0,
}


@dataclass
class DCPowerFlowResult:
    """Results from a DC power flow solve."""

    feasible: bool
    converged: bool
    max_angle_deg: float = 0.0
    max_line_loading_pct: float = 0.0
    total_generation_mw: float = 0.0
    total_load_mw: float = 0.0
    n_overloaded_lines: int = 0
    n_angle_violations: int = 0
    line_loadings: dict[str, float] = field(default_factory=dict)
    node_angles_deg: dict[str, float] = field(default_factory=dict)
    loss_estimate_mw: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "converged": self.converged,
            "max_angle_deg": round(self.max_angle_deg, 3),
            "max_line_loading_pct": round(self.max_line_loading_pct, 1),
            "total_generation_mw": round(self.total_generation_mw, 4),
            "total_load_mw": round(self.total_load_mw, 4),
            "n_overloaded_lines": self.n_overloaded_lines,
            "n_angle_violations": self.n_angle_violations,
            "loss_estimate_mw": round(self.loss_estimate_mw, 4),
            "summary": self.summary,
        }


class DCPowerFlow:
    """DC power flow solver for a PowerGraph.

    Builds the B-matrix (susceptance matrix) from edge impedances,
    assigns generation and load injections, and solves P = B * θ.
    """

    def __init__(
        self,
        graph: PowerGraph,
        profiles: list[LoadProfile] | None = None,
        hour: int = 19,  # default to evening peak
        angle_limit_deg: float = 30.0,
        loading_limit_pct: float = 100.0,
    ):
        self.graph = graph
        self.profiles = profiles or []
        self.hour = hour
        self.angle_limit_deg = angle_limit_deg
        self.loading_limit_pct = loading_limit_pct

        # Work with largest connected component only
        G = graph.G
        if G.number_of_nodes() == 0:
            self._nodes = []
            self._node_idx = {}
            return

        largest_cc = max(nx.connected_components(G), key=len)
        self._nodes = sorted(largest_cc)
        self._node_idx = {nid: i for i, nid in enumerate(self._nodes)}

    def solve(self) -> DCPowerFlowResult:
        """Run DC power flow and return results."""
        n = len(self._nodes)
        if n < 2:
            return DCPowerFlowResult(
                feasible=False, converged=False,
                summary="Graph too small for power flow (< 2 nodes in largest component)",
            )

        # Build susceptance matrix B
        B = lil_matrix((n, n))
        edge_data = []  # (u_idx, v_idx, b_pu, capacity_mva, edge_key)

        for edge in self.graph._edges:
            if edge.u not in self._node_idx or edge.v not in self._node_idx:
                continue

            i = self._node_idx[edge.u]
            j = self._node_idx[edge.v]

            # Compute susceptance from impedance
            x_ohm = edge.attrs.get("x_ohm", 0.0)
            r_ohm = edge.attrs.get("r_ohm", 0.0)

            if x_ohm <= 0:
                # Estimate from length and voltage class
                x_per_km = {VoltageClass.HV: 0.3, VoltageClass.MV: 0.4, VoltageClass.LV: 0.1}
                x_ohm = x_per_km.get(edge.voltage_class, 0.3) * edge.length_m / 1000.0

            if x_ohm <= 1e-6:
                x_ohm = 0.01  # minimum reactance to avoid singularity

            # Convert to per-unit
            vbase = _VBASE_KV.get(edge.voltage_class, 20.0)
            zbase = vbase ** 2 / _SBASE_MVA
            x_pu = x_ohm / zbase
            b_pu = 1.0 / x_pu  # susceptance

            B[i, i] += b_pu
            B[j, j] += b_pu
            B[i, j] -= b_pu
            B[j, i] -= b_pu

            capacity = _LINE_CAPACITY_MVA.get(
                edge.voltage_class,
                edge.attrs.get("capacity_mva", 10.0),
            )
            edge_data.append((i, j, b_pu, capacity, f"{edge.u}-{edge.v}"))

        # Build injection vector P (generation - load)
        P = np.zeros(n)
        load_map = {p.node_id: p for p in self.profiles}

        total_load = 0.0
        for nid in self._nodes:
            idx = self._node_idx[nid]
            if nid in load_map:
                load_mw = load_map[nid].demand_at_hour(self.hour)
                P[idx] -= load_mw / _SBASE_MVA  # load is negative injection
                total_load += load_mw

        # If no profiles, assign synthetic loads to demand nodes
        if not self.profiles:
            for nid in self._nodes:
                idx = self._node_idx[nid]
                node = self.graph.get_node(nid)
                if node and node.node_type in (NodeType.LOAD_PROXY, NodeType.MV_SUBSTATION):
                    load_mw = 0.005  # 5 kW default
                    P[idx] -= load_mw / _SBASE_MVA
                    total_load += load_mw

        # Assign generation to plants (distribute total load evenly)
        plant_nodes = [nid for nid in self._nodes
                       if self.graph.get_node(nid) and
                       self.graph.get_node(nid).node_type == NodeType.PLANT]

        if not plant_nodes:
            # Use HV substations as generation proxies
            plant_nodes = [nid for nid in self._nodes
                           if self.graph.get_node(nid) and
                           self.graph.get_node(nid).node_type == NodeType.HV_SUBSTATION]

        total_gen = 0.0
        if plant_nodes:
            gen_per_plant = total_load / len(plant_nodes) if total_load > 0 else 0.01
            for nid in plant_nodes:
                idx = self._node_idx[nid]
                P[idx] += gen_per_plant / _SBASE_MVA
                total_gen += gen_per_plant

        # Select slack bus (first generator/plant)
        slack = self._node_idx.get(plant_nodes[0] if plant_nodes else self._nodes[0], 0)

        # Remove slack bus row/column
        mask = np.ones(n, dtype=bool)
        mask[slack] = False
        B_red = B.tocsc()[mask][:, mask]
        P_red = P[mask]

        # Solve θ = B^-1 * P
        try:
            theta_red = spsolve(B_red, P_red)
            if np.any(np.isnan(theta_red)) or np.any(np.isinf(theta_red)):
                return DCPowerFlowResult(
                    feasible=False, converged=False,
                    summary="DC power flow diverged (NaN/Inf in solution)",
                )
        except Exception as exc:
            return DCPowerFlowResult(
                feasible=False, converged=False,
                summary=f"DC power flow solve failed: {exc}",
            )

        # Reconstruct full theta vector
        theta = np.zeros(n)
        theta[mask] = theta_red
        theta_deg = np.degrees(theta)

        # Compute line flows and loadings
        line_loadings = {}
        n_overloaded = 0
        n_angle_violations = 0
        max_loading = 0.0
        max_angle = 0.0
        total_loss = 0.0

        for (i, j, b_pu, capacity, key) in edge_data:
            flow_pu = b_pu * (theta[i] - theta[j])
            flow_mva = abs(flow_pu) * _SBASE_MVA
            loading_pct = (flow_mva / max(capacity, 0.01)) * 100.0

            line_loadings[key] = loading_pct
            max_loading = max(max_loading, loading_pct)

            if loading_pct > self.loading_limit_pct:
                n_overloaded += 1

            angle_diff = abs(theta_deg[i] - theta_deg[j])
            if angle_diff > self.angle_limit_deg:
                n_angle_violations += 1
            max_angle = max(max_angle, angle_diff)

            # Rough loss estimate: P_loss ≈ R * I² ≈ R/X * P_flow² / V²
            # For DC approximation, losses ∝ flow² * r/x
            total_loss += flow_mva ** 2 * 0.05 / max(capacity, 0.01)

        # Node angles
        node_angles = {self._nodes[i]: float(theta_deg[i]) for i in range(n)}

        feasible = n_overloaded == 0 and n_angle_violations == 0

        summary_parts = [
            f"DC-PF {'FEASIBLE' if feasible else 'INFEASIBLE'}",
            f"gen={total_gen:.3f}MW load={total_load:.3f}MW",
            f"max_loading={max_loading:.1f}%",
            f"max_angle={max_angle:.1f}°",
        ]
        if n_overloaded > 0:
            summary_parts.append(f"{n_overloaded} overloaded lines")
        if n_angle_violations > 0:
            summary_parts.append(f"{n_angle_violations} angle violations")

        result = DCPowerFlowResult(
            feasible=feasible,
            converged=True,
            max_angle_deg=float(max_angle),
            max_line_loading_pct=float(max_loading),
            total_generation_mw=total_gen,
            total_load_mw=total_load,
            n_overloaded_lines=n_overloaded,
            n_angle_violations=n_angle_violations,
            line_loadings=line_loadings,
            node_angles_deg=node_angles,
            loss_estimate_mw=total_loss,
            summary=", ".join(summary_parts),
        )

        logger.info("DC-PF: %s", result.summary)
        return result


def run_dc_powerflow(
    graph: PowerGraph,
    profiles: list[LoadProfile] | None = None,
    hours: list[int] | None = None,
) -> dict[str, Any]:
    """Run DC power flow for multiple hours and aggregate results.

    Parameters
    ----------
    graph : PowerGraph
        Power grid graph.
    profiles : list[LoadProfile], optional
        Load profiles. If None, uses synthetic minimum loads.
    hours : list[int], optional
        Hours to analyze. Default: [4, 10, 14, 19] (off-peak, morning, afternoon, evening).

    Returns
    -------
    dict
        Aggregated power flow metrics with "powerflow_" prefix.
    """
    if hours is None:
        hours = [4, 10, 14, 19]

    metrics: dict[str, Any] = {}
    results: list[DCPowerFlowResult] = []

    for hour in hours:
        pf = DCPowerFlow(graph, profiles, hour=hour)
        result = pf.solve()
        results.append(result)

        metrics[f"powerflow_h{hour:02d}_feasible"] = result.feasible
        metrics[f"powerflow_h{hour:02d}_max_loading_pct"] = result.max_line_loading_pct
        metrics[f"powerflow_h{hour:02d}_max_angle_deg"] = result.max_angle_deg
        metrics[f"powerflow_h{hour:02d}_overloaded"] = result.n_overloaded_lines
        metrics[f"powerflow_h{hour:02d}_loss_mw"] = result.loss_estimate_mw

    # Aggregate
    metrics["powerflow_all_hours_feasible"] = all(r.feasible for r in results)
    metrics["powerflow_any_hour_feasible"] = any(r.feasible or r.converged for r in results)
    metrics["powerflow_worst_loading_pct"] = max(
        (r.max_line_loading_pct for r in results), default=0.0
    )
    metrics["powerflow_worst_angle_deg"] = max(
        (r.max_angle_deg for r in results), default=0.0
    )
    metrics["powerflow_total_overloaded"] = sum(r.n_overloaded_lines for r in results)
    metrics["powerflow_avg_loss_mw"] = float(np.mean([r.loss_estimate_mw for r in results]))

    return metrics

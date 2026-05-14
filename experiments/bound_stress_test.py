"""Bound stress test (empirical + synthetic).

Empirical: scans the existing 1600-pair × 5-payoff Lipschitz pairs
to find where the bound CPR <= 2 * L * W1 becomes near-tight or
formally breaks at L_p95 (5%-tail).

Synthetic: constructs a controlled 6-node graph perturbation
trajectory under coordinate shifts Δ ∈ {1, 5, 25, 100} px and
measures (W1, |Δπ|) per payoff. Identifies the connectivity-flip
threshold where small W1 produces large CPR.

Outputs:
    results/bound_stress_test.csv
    results/bound_stress_synthetic.csv
    figures/bound_stress.pdf (5 panels)
    figures/bound_stress_synthetic.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr.payoff import PayoffConfig, allocation_payoff, grid_action_set
from cpr.payoff_routing import routing_mst_payoff, precompute_components
from cpr.payoff_coverage import coverage_payoff_interactive
from cpr.payoff_powerflow import dc_flow_payoff, payoff_n_minus_1
from cpr.transport import node_w1_distance

RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
PRETTY = {"wire": "Wire", "routing": "Routing", "coverage": "Coverage",
          "dc_flow": "DC-flow", "n_minus_1": r"$N{-}1$"}


# ---------------------------------------------------------------------
# Empirical scan
# ---------------------------------------------------------------------

def empirical_scan() -> pd.DataFrame:
    pairs = pd.read_csv(RESULTS / "lipschitz_pairs.csv")
    with (RESULTS / "lipschitz_estimates.json").open() as f:
        L_data = json.load(f)
    L_p95 = {p: float(L_data[p]["p95"]) for p in PAYOFFS}
    L_max = {p: float(L_data[p]["max"]) for p in PAYOFFS}
    rows = []
    for p in PAYOFFS:
        sub = pairs[pairs["payoff"] == p].copy()
        sub["L_p95"] = L_p95[p]
        sub["L_max"] = L_max[p]
        sub["near_tight"] = sub["ratio"] >= 0.95 * L_p95[p]
        sub["broke_p95"] = sub["ratio"] > L_p95[p]
        sub["broke_max"] = sub["ratio"] > L_max[p]
        rows.append(sub)
    out = pd.concat(rows, ignore_index=True)
    return out


def _plot_empirical(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(PAYOFFS),
                              figsize=(4.0 * len(PAYOFFS), 4.0),
                              constrained_layout=True)
    for ax, p in zip(axes, PAYOFFS):
        sub = df[df["payoff"] == p]
        ax.scatter(sub["w1"], np.abs(sub["p1"] - sub["p2"]),
                   s=8, alpha=0.5, color="#14375E", label="pairs")
        # The L_p95 line.
        L = float(sub["L_p95"].iloc[0])
        L_max = float(sub["L_max"].iloc[0])
        x_max = float(sub["w1"].max()) * 1.05
        x = np.linspace(0, x_max, 100)
        ax.plot(x, L * x, "--", color="#d62728", lw=1.0,
                label=fr"$L_{{p95}}={L:.3f}$")
        ax.plot(x, L_max * x, ":", color="#888", lw=1.0,
                label=fr"$L_{{\max}}={L_max:.3f}$")
        ax.set_xlabel(r"$W_1$ (node-Wasserstein)")
        ax.set_ylabel(r"$|\Delta\pi|$")
        ax.set_title(PRETTY[p])
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Bound stress test: empirical |Δπ| vs W₁ "
                  "across 200 panel-pair × 8 actions; "
                  "dashed line = $L_{p95}$, dotted = $L_{\\max}$",
                  fontsize=10)
    out = FIGS / "bound_stress.pdf"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------
# Synthetic worst case: 6-node graph + coordinate perturbation
# ---------------------------------------------------------------------

def _make_test_graph() -> nx.MultiGraph:
    """Two clusters of 3 nodes each, joined by a single bridge edge.

    The bridge node sits at (768, 768). Perturbing one cluster's
    centroid coordinate is what causes the connectivity flip.
    """
    g = nx.MultiGraph()
    nodes = {
        "A1": (300, 300), "A2": (300, 500), "A3": (400, 400),
        "B1": (1100, 1000), "B2": (1100, 1200), "B3": (1200, 1100),
        "BR": (768, 768),
    }
    for name, (r, c) in nodes.items():
        g.add_node((r, c), px=(r, c))
    # Cluster A internal MST.
    for u, v in [("A1", "A2"), ("A2", "A3"), ("A3", "BR")]:
        a, b = nodes[u], nodes[v]
        g.add_edge(a, b,
                   length_px=float(np.hypot(a[0]-b[0], a[1]-b[1])))
    for u, v in [("B1", "B2"), ("B2", "B3"), ("B3", "BR")]:
        a, b = nodes[u], nodes[v]
        g.add_edge(a, b,
                   length_px=float(np.hypot(a[0]-b[0], a[1]-b[1])))
    return g


def _perturb_node(g: nx.MultiGraph, target_key,
                    delta_px: tuple[float, float]) -> nx.MultiGraph:
    """Return a copy of g with target_key node moved by delta_px."""
    new_g = nx.MultiGraph()
    new_key = (target_key[0] + int(delta_px[0]),
                target_key[1] + int(delta_px[1]))
    key_map = {n: (new_key if n == target_key else n) for n in g.nodes}
    for n in g.nodes:
        new_n = key_map[n]
        new_g.add_node(new_n, px=new_n)
    for u, v, d in g.edges(data=True):
        nu, nv = key_map[u], key_map[v]
        # Recompute length so the perturbation propagates.
        length = float(np.hypot(nu[0] - nv[0], nu[1] - nv[1]))
        new_g.add_edge(nu, nv, length_px=length)
    return new_g


def synthetic_trajectory(deltas_px: list[float]) -> pd.DataFrame:
    g0 = _make_test_graph()
    cfg = PayoffConfig(K=3, beta=100.0, alpha_node=0.0, normalize=True)
    actions = grid_action_set(K=3, grid_size=8, tile_size=1536,
                                n_actions=8, seed=0)

    # Per-payoff baseline payoffs on g0.
    truth_w = np.array([allocation_payoff(a, g0, cfg=cfg) for a in actions])
    truth_r = np.array([routing_mst_payoff(a, g0, cfg=cfg) for a in actions])
    truth_c = np.array([coverage_payoff_interactive(a, g0, cfg=cfg)
                          for a in actions])
    truth_dc = np.array([dc_flow_payoff(a, g0, cfg=cfg) for a in actions])
    truth_n1 = np.array([payoff_n_minus_1(a, g0, cfg=cfg, n_outages=4)
                            for a in actions])

    # Target: bridge node BR. Push it laterally toward cluster A.
    target = (768, 768)
    rows = []
    for d in deltas_px:
        gp = _perturb_node(g0, target, (-d, 0))
        # W1 to original.
        w1 = node_w1_distance(g0, gp, max_pts=200, seed=0)
        for ai, a in enumerate(actions):
            for pname, truth_v, payoff_fn in [
                ("wire", truth_w[ai], lambda a, t: allocation_payoff(a, t, cfg=cfg)),
                ("routing", truth_r[ai], lambda a, t: routing_mst_payoff(a, t, cfg=cfg)),
                ("coverage", truth_c[ai], lambda a, t: coverage_payoff_interactive(a, t, cfg=cfg)),
                ("dc_flow", truth_dc[ai], lambda a, t: dc_flow_payoff(a, t, cfg=cfg)),
                ("n_minus_1", truth_n1[ai], lambda a, t: payoff_n_minus_1(
                    a, t, cfg=cfg, n_outages=4)),
            ]:
                v_p = float(payoff_fn(a, gp))
                rows.append({
                    "delta_px": d, "payoff": pname,
                    "action_idx": ai,
                    "w1": float(w1),
                    "delta_pi": abs(v_p - float(truth_v)),
                    "ratio": (abs(v_p - float(truth_v)) / max(w1, 1e-9)),
                })
    return pd.DataFrame(rows)


def _plot_synthetic(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, len(PAYOFFS),
                              figsize=(4.0 * len(PAYOFFS), 4.0),
                              constrained_layout=True)
    for ax, p in zip(axes, PAYOFFS):
        sub = df[df["payoff"] == p]
        # For each delta, the median |Δπ| / W1.
        med = sub.groupby("delta_px")["ratio"].agg(
            ["median", "min", "max"]).reset_index()
        ax.plot(med["delta_px"], med["median"], "o-", color="#d62728")
        ax.fill_between(med["delta_px"], med["min"], med["max"],
                          alpha=0.2, color="#d62728")
        ax.set_xlabel(r"perturbation $\Delta$ (px)")
        ax.set_ylabel(r"$|\Delta\pi| / W_1$")
        ax.set_title(PRETTY[p])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle("Synthetic stress: |Δπ| / W₁ vs node-coordinate "
                  "perturbation Δ across 4 magnitudes",
                  fontsize=10)
    out = FIGS / "bound_stress_synthetic.pdf"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


def main() -> int:
    print("=== Empirical scan ===")
    emp = empirical_scan()
    out = RESULTS / "bound_stress_test.csv"
    emp.to_csv(out, index=False)
    print(f"Wrote {out} ({len(emp)} rows)")
    print()
    for p in PAYOFFS:
        sub = emp[emp["payoff"] == p]
        n_total = len(sub)
        n_near_tight = int(sub["near_tight"].sum())
        n_broke_p95 = int(sub["broke_p95"].sum())
        n_broke_max = int(sub["broke_max"].sum())
        print(f"  {p:>10s}: n={n_total}  near-tight (>=0.95*L_p95): "
              f"{n_near_tight} ({100*n_near_tight/n_total:.1f}%)  "
              f"break L_p95: {n_broke_p95} "
              f"({100*n_broke_p95/n_total:.1f}%)  "
              f"break L_max: {n_broke_max} ({100*n_broke_max/n_total:.1f}%)")
    fig_path = _plot_empirical(emp)
    print(f"Wrote {fig_path}")

    print()
    print("=== Synthetic stress ===")
    deltas = [1.0, 5.0, 25.0, 100.0, 250.0, 500.0]
    syn = synthetic_trajectory(deltas)
    out2 = RESULTS / "bound_stress_synthetic.csv"
    syn.to_csv(out2, index=False)
    print(f"Wrote {out2} ({len(syn)} rows)")
    print()
    for p in PAYOFFS:
        sub = syn[syn["payoff"] == p]
        for d in sorted(syn["delta_px"].unique()):
            cell = sub[sub["delta_px"] == d]
            if cell.empty:
                continue
            med = float(cell["ratio"].median())
            mx = float(cell["ratio"].max())
            print(f"  {p:>10s}  delta={d:>6.1f} px  "
                  f"median ratio={med:.3f}  max ratio={mx:.3f}")
    fig_path = _plot_synthetic(syn)
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Empirical Lipschitz constants for all five canonical payoffs.

Replaces the fabricated $L_{DC,p95} = 1.2 \times 10^4$ that previously
appeared in the paper. For each payoff
$\pi(a, \theta)$ we sample $B$ pairs of held-out graphs and
$A$ random actions per pair, compute the empirical Lipschitz ratio

    |\pi(a, \theta) - \pi(a, \theta')| / d(\theta, \theta')

under the **node-W1** metric, and report the 95th-percentile value
as the working $\hat L_\mathrm{p95}$. The 95th percentile is more
robust than the maximum to ill-conditioned pairs of nearly-identical
topologies (where W1 -> 0).

Outputs
-------
    results/lipschitz_estimates.json -- per-payoff (mean, p95, max)
                                        in (payoff_units / pixel)
    results/lipschitz_pairs.csv      -- raw pair-level ratios for diagnostics
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr.graphs import graph_signature  # noqa: E402
from cpr.payoff import (  # noqa: E402
    PayoffConfig, allocation_payoff, grid_action_set,
)
from cpr.payoff_routing import (  # noqa: E402
    routing_mst_payoff, precompute_components,
)
from cpr.payoff_coverage import coverage_payoff_interactive  # noqa: E402
from cpr.payoff_powerflow import dc_flow_payoff, payoff_n_minus_1  # noqa: E402
from cpr.transport import node_w1_distance  # noqa: E402

RESULTS = ROOT / "results"
GRAPHS = RESULTS / "graphs"
GRAPHS_MC = RESULTS / "graphs_mc"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]
RUNS = [
    "multi_city_hv_v2", "multi_city_hv_v3",
    "multi_city_hv_cgan_v1", "multi_city_hv_cgan_v2",
    "multi_city_hv_v2_6ch", "multi_city_hv_cgan_v3",
    "baseline_random", "baseline_perturbed",
    "voronoi_density", "mst_substations",
    "digress_v1",
]

CFG = PayoffConfig(K=4, beta=100.0, alpha_node=0.0, normalize=True)
B_PAIRS = 200       # number of (theta, theta') pairs sampled
A_ACTIONS = 8       # actions per pair
N_OUTAGES_FOR_L = 4  # smaller than panel's 16 to keep this driver tractable


def _load_graphml(path: Path) -> nx.MultiGraph:
    g = nx.read_graphml(str(path))
    h = nx.MultiGraph()
    id_map = {}
    for n, d in g.nodes(data=True):
        key = (int(float(d.get("px_r", 0))), int(float(d.get("px_c", 0))))
        id_map[n] = key
        h.add_node(key, px=key)
    for u, v, d in g.edges(data=True):
        h.add_edge(id_map[u], id_map[v],
                   length_px=float(d.get("length_px", 0.0)))
    return h


def _gather_graphs() -> list[nx.MultiGraph]:
    """All available (run, city) graphs from the panel."""
    out = []
    for run in RUNS:
        for city in CITIES:
            p = GRAPHS_MC / run / f"{city}_00.graphml"
            if p.exists():
                try:
                    g = _load_graphml(p)
                    if g.number_of_nodes() > 0:
                        out.append(g)
                except Exception:
                    pass
    return out


def _payoff_dispatch(name: str):
    """Return a payoff function compatible with (a, theta) -> float."""
    if name == "wire":
        return lambda a, t: allocation_payoff(a, t, cfg=CFG)
    if name == "routing":
        # Cache the precompute_components per call (graph changes pair-by-pair).
        return lambda a, t: routing_mst_payoff(
            a, t, cfg=CFG, cache=precompute_components(t),
        )
    if name == "coverage":
        return lambda a, t: coverage_payoff_interactive(a, t, cfg=CFG)
    if name == "dc_flow":
        return lambda a, t: dc_flow_payoff(a, t, cfg=CFG)
    if name == "n_minus_1":
        return lambda a, t: payoff_n_minus_1(
            a, t, cfg=CFG, n_outages=N_OUTAGES_FOR_L,
        )
    raise ValueError(name)


def main() -> int:
    graphs = _gather_graphs()
    if len(graphs) < 2:
        print(f"[error] Not enough graphs to sample pairs: {len(graphs)} loaded.")
        return 2
    print(f"Loaded {len(graphs)} non-empty graphs across the panel.")

    rng = np.random.default_rng(0)
    actions = grid_action_set(K=CFG.K, grid_size=12, tile_size=1536,
                              n_actions=A_ACTIONS, seed=0)

    # Pre-compute graph signatures (used for the W1 denominator).
    pairs = []
    for _ in range(B_PAIRS):
        i, j = rng.choice(len(graphs), size=2, replace=False)
        if i == j:
            continue
        pairs.append((graphs[i], graphs[j]))

    payoff_names = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
    results: dict[str, list[float]] = {p: [] for p in payoff_names}
    pair_rows = []

    print(f"\nComputing per-pair node-W1 (this is the slow part)...")
    pair_w1 = []
    for k, (g1, g2) in enumerate(pairs):
        w = node_w1_distance(g1, g2, max_pts=500, seed=k)
        if not np.isfinite(w) or w < 1e-6:
            pair_w1.append(None)
            continue
        pair_w1.append(float(w))

    valid_pairs = [(g1, g2, w) for (g1, g2), w in zip(pairs, pair_w1)
                   if w is not None]
    print(f"Valid pairs (W1 > 0): {len(valid_pairs)}/{len(pairs)}")

    print(f"\nFor each payoff, sampling {A_ACTIONS} actions per pair "
          f"and computing |\\Delta\\pi| / W1...")
    for name in payoff_names:
        pay_fn = _payoff_dispatch(name)
        ratios = []
        for k, (g1, g2, w1) in enumerate(valid_pairs):
            for a in actions:
                try:
                    p1 = pay_fn(a, g1)
                    p2 = pay_fn(a, g2)
                    if not (np.isfinite(p1) and np.isfinite(p2)):
                        continue
                    r = abs(p1 - p2) / w1
                    ratios.append(r)
                    pair_rows.append({
                        "payoff": name, "pair_idx": k,
                        "p1": p1, "p2": p2, "w1": w1, "ratio": r,
                    })
                except Exception:
                    continue
        arr = np.asarray(ratios) if ratios else np.array([np.nan])
        results[name] = arr.tolist()
        print(f"  [{name:>10s}]  n_ratios={len(ratios)}  "
              f"mean={float(np.mean(arr)):8.4f}  "
              f"median={float(np.median(arr)):8.4f}  "
              f"p95={float(np.percentile(arr, 95)):8.4f}  "
              f"max={float(np.max(arr)):8.4f}")

    summary = {}
    for name in payoff_names:
        arr = np.asarray(results[name])
        summary[name] = {
            "n_ratios": int(len(arr)),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "B_pairs_used": int(len(valid_pairs)),
            "A_actions_per_pair": A_ACTIONS,
        }
    out = RESULTS / "lipschitz_estimates.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")

    pairs_csv = RESULTS / "lipschitz_pairs.csv"
    with pairs_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["payoff", "pair_idx",
                                            "p1", "p2", "w1", "ratio"])
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)
    print(f"Wrote {pairs_csv} ({len(pair_rows)} rows)")

    print(f"\n=== Summary table for the paper ===")
    print(f"{'Payoff':>12s}  {'L_p95':>10s}  {'L_mean':>10s}  {'L_max':>10s}")
    for name in payoff_names:
        s = summary[name]
        print(f"{name:>12s}  {s['p95']:10.4f}  {s['mean']:10.4f}  {s['max']:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

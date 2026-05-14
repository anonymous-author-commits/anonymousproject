"""Run CPR with the AC-OPF payoff over the full panel.

Compares the AC-flow CPR ranking to the DC-flow CPR ranking
(already in cpr_panel.csv) to test whether the bound's argument
generalises to nonlinear AC operation.

Output:
    results/ac_flow_cpr.csv -- per (run, city) AC-flow CPR + meta
    results/dc_vs_ac_spearman.json -- ranking concordance
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr.payoff import PayoffConfig, grid_action_set
from cpr.payoff_powerflow_ac import ac_flow_payoff
from cpr.payoff_routing import precompute_components

GRAPHS = ROOT / "results" / "graphs"
GRAPHS_MC = ROOT / "results" / "graphs_mc"
RESULTS = ROOT / "results"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]
CFG = PayoffConfig(K=4, beta=100.0, alpha_node=0.0, normalize=True)

RUN_FOLDER = {
    "v2": "multi_city_hv_v2", "v3": "multi_city_hv_v3",
    "v2_6ch": "multi_city_hv_v2_6ch",
    "cgan_v1": "multi_city_hv_cgan_v1",
    "cgan_v2": "multi_city_hv_cgan_v2",
    "cgan_v3": "multi_city_hv_cgan_v3",
    "digress_v1": "digress_v1",
    "voronoi_density": "voronoi_density",
    "mst_substations": "mst_substations",
    "baseline_random": "baseline_random",
    "baseline_perturbed": "baseline_perturbed",
}


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


def _load_samples(run: str, city: str, n_max: int = 4):
    folder = RUN_FOLDER.get(run, run)
    rd = GRAPHS_MC / folder
    if not rd.exists():
        return []
    out = []
    for p in sorted(rd.glob(f"{city}_*.graphml"))[:n_max]:
        try:
            g = _load_graphml(p)
            if g.number_of_nodes() > 0:
                out.append(g)
        except Exception:
            pass
    return out


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]; b = b[finite]
    if a.size < 2:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    den = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / den) if den > 0 else float("nan")


def main() -> int:
    # Tractable budget: 5 actions, 2 samples per cell.
    # Total = 11 gens * 5 cities * 5 actions * (1 truth + 2 samples) = 825
    # AC-flow calls. At ~5 s each conservatively that's ~70 min; with
    # the per-call timeout (next paragraph), the worst case is bounded.
    actions = grid_action_set(K=CFG.K, grid_size=12, tile_size=1536,
                              n_actions=5, seed=0)
    rows = []
    n_total = 0
    n_converged = 0
    n_skipped = 0
    import time as _time
    PER_CELL_BUDGET_S = 60.0  # don't spend more than this on any one
    # (run, city) cell; mark remaining calls as nan if exceeded.
    for run in RUN_FOLDER:
        for city in CITIES:
            truth_path = GRAPHS / "_truth" / f"{city}.graphml"
            if not truth_path.exists():
                continue
            truth = _load_graphml(truth_path)
            samples = _load_samples(run, city, n_max=2)
            if not samples:
                rows.append({"run": run, "city": city,
                              "cpr_ac": float("nan"),
                              "convergence_rate": 0.0})
                continue

            t0 = _time.time()
            truth_pay = []
            truth_conv = []
            for a in actions:
                if _time.time() - t0 > PER_CELL_BUDGET_S:
                    truth_pay.append(float("nan")); truth_conv.append(False)
                    n_skipped += 1
                    continue
                v, meta = ac_flow_payoff(a, truth, cfg=CFG,
                                            return_meta=True)
                truth_pay.append(v)
                truth_conv.append(meta["converged"])
                n_total += 1
                n_converged += int(meta["converged"])
            truth_pay = np.array(truth_pay)

            model_pay = np.zeros(len(actions))
            model_conv = []
            for ai, a in enumerate(actions):
                if _time.time() - t0 > PER_CELL_BUDGET_S:
                    model_pay[ai] = float("nan")
                    model_conv.append(0.0)
                    n_skipped += len(samples)
                    continue
                ps = []; convs = []
                for s in samples:
                    if _time.time() - t0 > PER_CELL_BUDGET_S:
                        ps.append(float("nan")); convs.append(False)
                        n_skipped += 1
                        continue
                    v, meta = ac_flow_payoff(a, s, cfg=CFG,
                                                return_meta=True)
                    ps.append(v)
                    convs.append(meta["converged"])
                    n_total += 1
                    n_converged += int(meta["converged"])
                model_pay[ai] = float(np.nanmean(ps)) if any(
                    np.isfinite(p) for p in ps) else float("nan")
                model_conv.append(float(np.mean(convs)))

            if not np.any(np.isfinite(truth_pay)):
                cpr = float("nan")
            else:
                chosen = int(np.nanargmax(model_pay))
                oracle = int(np.nanargmax(truth_pay))
                cpr = float(truth_pay[oracle] - truth_pay[chosen])
            conv_rate = float(np.mean(truth_conv + model_conv))
            rows.append({"run": run, "city": city,
                          "cpr_ac": cpr,
                          "convergence_rate": conv_rate})
            print(f"  [{run:<22s}/{city:<10s}]  AC CPR = {cpr:.3f}  "
                  f"conv = {100*conv_rate:.0f}%", flush=True)

    out = RESULTS / "ac_flow_cpr.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out}")
    print(f"Total convergence rate: {100*n_converged/max(n_total,1):.1f}% "
          f"({n_converged}/{n_total})")

    # Spearman vs DC CPR ranking.
    panel = pd.read_csv(RESULTS / "cpr_panel.csv")
    dc = (panel.assign(dc=panel[["cpr_dc_flow_indep",
                                     "cpr_dc_flow_corr"]].mean(axis=1))
                 [["run", "city", "dc"]])
    ac = pd.DataFrame(rows)
    merged = dc.merge(ac, on=["run", "city"], how="inner")

    rho_per_city = {}
    for city in CITIES:
        sub = merged[merged["city"] == city]
        rho_per_city[city] = _spearman(sub["dc"].values,
                                          sub["cpr_ac"].values)
    rho_aggregate = _spearman(merged["dc"].values, merged["cpr_ac"].values)

    summary = {
        "rho_per_city": rho_per_city,
        "rho_aggregate": float(rho_aggregate),
        "convergence_rate": float(n_converged / max(n_total, 1)),
        "n_total_calls": int(n_total),
        "n_converged_calls": int(n_converged),
    }
    out_json = RESULTS / "dc_vs_ac_spearman.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_json}")
    print()
    for city, rho in rho_per_city.items():
        print(f"  rho(DC, AC | {city:>10s}) = {rho:+.3f}")
    print(f"  rho(DC, AC | aggregate)        = {rho_aggregate:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

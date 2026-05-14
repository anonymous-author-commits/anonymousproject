"""W6 -- Run the road-corridor-aware routing payoff over the panel.

Computes road-corridor CPR for every (run, city) cell, then prints
the ranking and Spearman correlation against the existing
routing-MST CPR. If the road payoff produces a different ranking,
that's evidence the realism upgrade matters.

Output: results/road_corridor_cpr.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cpr.payoff import PayoffConfig, grid_action_set
from cpr.payoff_road_corridor import road_corridor_payoff

GRAPHS = ROOT / "results" / "graphs"
GRAPHS_MC = ROOT / "results" / "graphs_mc"
RESULTS = ROOT / "results"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]

# Use the same panel + folder mapping as run_cpr_panel.py.
RUN_FOLDER = {
    "v2": "multi_city_hv_v2",
    "v3": "multi_city_hv_v3",
    "cgan_v1": "multi_city_hv_cgan_v1",
    "cgan_v2": "multi_city_hv_cgan_v2",
    "cgan_v3": "multi_city_hv_cgan_v3",
    "v2_6ch": "multi_city_hv_v2_6ch",
    "digress_v1": "digress_v1",
    "voronoi_density": "voronoi_density",
    "mst_substations": "mst_substations",
    "baseline_random": "baseline_random",
    "baseline_perturbed": "baseline_perturbed",
}

CFG = PayoffConfig(K=4, beta=100.0, alpha_node=0.0, normalize=True)


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


def _load_samples(run: str, city: str, n_max: int = 4) -> list[nx.MultiGraph]:
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


def main() -> int:
    actions = grid_action_set(K=CFG.K, grid_size=12, tile_size=1536,
                              n_actions=10, seed=0)
    rows = []
    for run in RUN_FOLDER:
        for city in CITIES:
            truth_path = GRAPHS / "_truth" / f"{city}.graphml"
            if not truth_path.exists():
                continue
            truth = _load_graphml(truth_path)
            samples = _load_samples(run, city, n_max=4)
            if not samples:
                rows.append({"run": run, "city": city,
                              "cpr_road_corridor": float("nan")})
                continue

            truth_pay = np.array([road_corridor_payoff(a, truth, city=city,
                                                          cfg=CFG)
                                   for a in actions])
            model_pay = np.zeros(len(actions))
            for ai, a in enumerate(actions):
                ps = [road_corridor_payoff(a, s, city=city, cfg=CFG)
                      for s in samples]
                model_pay[ai] = float(np.mean(ps))
            chosen = int(np.argmax(model_pay))
            oracle = int(np.argmax(truth_pay))
            cpr = float(truth_pay[oracle] - truth_pay[chosen])
            rows.append({"run": run, "city": city,
                          "cpr_road_corridor": cpr})
            print(f"  [{run:<22s}/{city:<10s}]  road CPR = {cpr:.2f}")

    out = RESULTS / "road_corridor_cpr.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out}")

    # Compare to existing routing-CPR ranking via Spearman.
    import pandas as pd
    panel = pd.read_csv(RESULTS / "cpr_panel.csv")
    panel["routing_cpr"] = panel[["cpr_routing_indep",
                                    "cpr_routing_corr"]].mean(axis=1)
    road_df = pd.DataFrame(rows)
    merged = panel[["run", "city", "routing_cpr"]].merge(
        road_df, on=["run", "city"], how="inner"
    )

    def _spearman(a, b):
        a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
        if a.size < 2:
            return float("nan")
        ar = np.argsort(np.argsort(a)).astype(float)
        br = np.argsort(np.argsort(b)).astype(float)
        ar -= ar.mean(); br -= br.mean()
        den = np.sqrt((ar * ar).sum() * (br * br).sum())
        return float((ar * br).sum() / den) if den > 0 else float("nan")

    finite = merged.dropna(subset=["routing_cpr", "cpr_road_corridor"])
    rho = _spearman(finite["routing_cpr"], finite["cpr_road_corridor"])
    print(f"\nSpearman rho(routing-MST, road-corridor) = {rho:+.3f} "
          f"(n={len(finite)} cells)")

    by_run = (finite.groupby("run")[["routing_cpr", "cpr_road_corridor"]]
               .mean().sort_values("cpr_road_corridor"))
    print("\nPer-run mean CPR:")
    print(by_run.to_string(float_format=lambda x: f"{x:7.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

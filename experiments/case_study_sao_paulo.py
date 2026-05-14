"""W9 -- Miniature planning case study on São Paulo.

Picks one city (São Paulo, the sparsest and most uncertain in our
panel). Defines a planning task with explicit demand nodes and
candidate substations from the truth graph. Compares the K=4
substation-siting decision under each generator's belief plus
three baseline selectors (Dice-best, CPR-best, random) against the
oracle decision computed from the truth graph.

Reports: kilometres of excess routing, dollars of excess capex.

Output: results/case_study_sao_paulo.json +.csv
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

from cpr.payoff import PayoffConfig, grid_action_set
from cpr.payoff_routing import precompute_components, routing_mst_payoff

GRAPHS = ROOT / "results" / "graphs"
GRAPHS_MC = ROOT / "results" / "graphs_mc"
RESULTS = ROOT / "results"

CITY = "sao_paulo"
RUN_FOLDER = {
    "v2": "multi_city_hv_v2", "v3": "multi_city_hv_v3",
    "cgan_v1": "multi_city_hv_cgan_v1", "cgan_v2": "multi_city_hv_cgan_v2",
    "cgan_v3": "multi_city_hv_cgan_v3", "v2_6ch": "multi_city_hv_v2_6ch",
    "digress_v1": "digress_v1",
    "voronoi_density": "voronoi_density",
    "mst_substations": "mst_substations",
}
DICE_BY_RUN = {"v2": 0.078, "v3": 0.077, "cgan_v1": 0.080,
                "cgan_v2": 0.043, "v2_6ch": 0.085, "cgan_v3": 0.091}

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


def _load_samples(run: str, n_max: int = 6) -> list[nx.MultiGraph]:
    folder = RUN_FOLDER.get(run, run)
    rd = GRAPHS_MC / folder
    if not rd.exists():
        return []
    out = []
    for p in sorted(rd.glob(f"{CITY}_*.graphml"))[:n_max]:
        try:
            g = _load_graphml(p)
            if g.number_of_nodes() > 0:
                out.append(g)
        except Exception:
            pass
    return out


def _action_set(seed: int = 0, n: int = 32):
    return grid_action_set(K=CFG.K, grid_size=12, tile_size=1536,
                            n_actions=n, seed=seed)


def main() -> int:
    truth_path = GRAPHS / "_truth" / f"{CITY}.graphml"
    truth = _load_graphml(truth_path)
    truth_cache = precompute_components(truth)
    actions = _action_set()

    truth_pay = np.array([routing_mst_payoff(a, truth, cfg=CFG,
                                                cache=truth_cache)
                           for a in actions])
    oracle_idx = int(np.argmax(truth_pay))
    oracle_cost = -float(truth_pay[oracle_idx])
    print(f"[{CITY}] truth-graph |V|={truth.number_of_nodes()}, "
          f"|E|={truth.number_of_edges()}")
    print(f"  Oracle action #{oracle_idx}; "
          f"oracle truth-graph routing cost = {oracle_cost:.3f} "
          f"(per-node-mean px)")

    rows = []
    rng = np.random.default_rng(0)
    # Random selector: pick a random action.
    random_idx = int(rng.integers(0, len(actions)))
    random_chosen_cost = -float(truth_pay[random_idx])
    rows.append({
        "selector": "Random action (no model)",
        "selected_run": "n/a", "chosen_action_idx": random_idx,
        "deployed_truth_cost": random_chosen_cost,
        "regret_vs_oracle_px_per_node": random_chosen_cost - oracle_cost,
    })

    for run, folder in RUN_FOLDER.items():
        samples = _load_samples(run, n_max=6)
        if not samples:
            continue
        sample_caches = [precompute_components(s) for s in samples]
        model_pay = np.zeros(len(actions))
        for ai, a in enumerate(actions):
            ps = [routing_mst_payoff(a, samples[i], cfg=CFG,
                                       cache=sample_caches[i])
                   for i in range(len(samples))]
            model_pay[ai] = float(np.mean(ps))
        chosen_idx = int(np.argmax(model_pay))
        chosen_cost = -float(truth_pay[chosen_idx])
        rows.append({
            "selector": f"Generator: {run}",
            "selected_run": run, "chosen_action_idx": chosen_idx,
            "deployed_truth_cost": chosen_cost,
            "regret_vs_oracle_px_per_node": chosen_cost - oracle_cost,
        })

    # Convert regret to capex band (HV mid).
    PIXEL_M = 10.0
    HV_MID = 2.5  # USD M / km
    K = 4
    for r in rows:
        excess_km_per_round = (r["regret_vs_oracle_px_per_node"]
                                 * PIXEL_M * K / 1000.0)
        r["excess_km_per_round"] = excess_km_per_round
        r["excess_capex_HV_mid_M"] = excess_km_per_round * HV_MID

    rows.sort(key=lambda r: r["regret_vs_oracle_px_per_node"])
    print()
    print(f"{'Selector':<32s} {'regret(px/node)':>15s} "
          f"{'km/round':>10s} {'$M/round':>10s}")
    for r in rows:
        print(f"{r['selector']:<32s} "
              f"{r['regret_vs_oracle_px_per_node']:15.3f} "
              f"{r['excess_km_per_round']:10.3f} "
              f"{r['excess_capex_HV_mid_M']:10.3f}")

    csv_path = RESULTS / f"case_study_{CITY}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {csv_path}")

    summary = {
        "city": CITY,
        "truth_n_nodes": truth.number_of_nodes(),
        "truth_n_edges": truth.number_of_edges(),
        "oracle_routing_cost_px_per_node": oracle_cost,
        "K": K,
        "PIXEL_M": PIXEL_M,
        "HV_mid_M_per_km": HV_MID,
        "selectors": rows,
    }
    json_path = RESULTS / f"case_study_{CITY}.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

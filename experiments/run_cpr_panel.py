"""T1.2b — Run CPR under all three canonical payoffs on both
sampling regimes (independent + correlated) and produce the
3-payoff cross-Spearman matrix.

Outputs:
  results/cpr_panel.csv         — per-cell CPR for all 3 payoffs x 2 samplers
  results/cpr_panel_summary.csv — per-run mean for all 6 (payoff, sampler) cols
  results/cross_payoff_rho.json — 3x3 Spearman matrix (averaged over samplers)

This driver intentionally re-uses the same evaluation harness for
each payoff so the comparison is apples-to-apples.
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

from cpr.payoff import PayoffConfig, allocation_payoff, grid_action_set  # noqa: E402
from cpr.payoff_routing import routing_mst_payoff, precompute_components  # noqa: E402
from cpr.payoff_coverage import coverage_payoff_interactive  # noqa: E402
from cpr.payoff_powerflow import dc_flow_payoff, payoff_n_minus_1  # noqa: E402

GRAPHS = ROOT / "results" / "graphs"
RESULTS = ROOT / "results"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]
RUNS = [
    "multi_city_hv_v2",
    "multi_city_hv_v3",
    "multi_city_hv_cgan_v1",
    "multi_city_hv_cgan_v2",
    "multi_city_hv_v2_6ch",
    "multi_city_hv_cgan_v3",
    "baseline_random",
    "baseline_perturbed",
    "voronoi_density",
    "mst_substations",
    "digress_v1",   # F3 graph-native learned generator
]

# Independent and correlated graph stores.
SAMPLE_DIRS = {
    "indep": ROOT / "results" / "graphs_mc",
    "corr":  ROOT / "results" / "graphs_mc_corr",
}

# Three canonical payoffs.
CFG = PayoffConfig(K=4, beta=100.0, alpha_node=0.0, normalize=True)
PAYOFFS = {
    "wire":     lambda a, t: allocation_payoff(a, t, cfg=CFG),
    "routing":  lambda a, t: routing_mst_payoff(a, t, cfg=CFG),
    "coverage": lambda a, t: coverage_payoff_interactive(a, t, cfg=CFG),
    "dc_flow":  lambda a, t: dc_flow_payoff(a, t, cfg=CFG),
    # N-1 is more expensive (multiplies DC-flow cost by n_outages).
    # We run it with n_outages=16 in the main panel; a separate
    # supplementary experiment uses n_outages=8 on a subset of cells
    # to verify ranking stability.
    "n_minus_1": lambda a, t: payoff_n_minus_1(a, t, cfg=CFG, n_outages=16),
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
        h.add_edge(id_map[u], id_map[v], length_px=float(d.get("length_px", 0.0)))
    return h


def _load_samples(sample_dir: Path, run: str, city: str) -> list[nx.MultiGraph]:
    rd = sample_dir / run
    if not rd.exists():
        return []
    out = []
    for p in sorted(rd.glob(f"{city}_*.graphml")):
        try:
            out.append(_load_graphml(p))
        except Exception:
            pass
    return [g for g in out if g.number_of_nodes() > 0]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.size != b.size:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / denom) if denom > 0 else float("nan")


def cpr_three_payoffs(
    samples: list[nx.MultiGraph],
    truth: nx.MultiGraph,
    actions: list,
) -> dict:
    """Compute CPR under all five payoffs in one pass, caching MSTs.

    The function name is preserved for backwards compatibility but
    now evaluates wire / routing / coverage / dc_flow / n_minus_1.
    """
    PAYOFF_KEYS = ("wire", "routing", "coverage", "dc_flow", "n_minus_1")
    if not samples:
        return {p: float("nan") for p in PAYOFF_KEYS}

    # Pre-cache MSTs (action-independent) for routing.
    truth_cache = precompute_components(truth)
    sample_caches = [precompute_components(s) for s in samples]

    truth_w = np.array([allocation_payoff(a, truth, cfg=CFG) for a in actions])
    truth_r = np.array([
        routing_mst_payoff(a, truth, cfg=CFG, cache=truth_cache)
        for a in actions
    ])
    truth_c = np.array([
        coverage_payoff_interactive(a, truth, cfg=CFG)
        for a in actions
    ])
    truth_dc = np.array([
        dc_flow_payoff(a, truth, cfg=CFG) for a in actions
    ])
    truth_n1 = np.array([
        payoff_n_minus_1(a, truth, cfg=CFG, n_outages=16) for a in actions
    ])

    model_w = np.zeros(len(actions))
    model_r = np.zeros(len(actions))
    model_c = np.zeros(len(actions))
    model_dc = np.zeros(len(actions))
    model_n1 = np.zeros(len(actions))
    for ai, a in enumerate(actions):
        ws = [allocation_payoff(a, s, cfg=CFG) for s in samples]
        rs = [routing_mst_payoff(a, samples[i], cfg=CFG, cache=sample_caches[i])
              for i in range(len(samples))]
        cs = [coverage_payoff_interactive(a, s, cfg=CFG) for s in samples]
        ds = [dc_flow_payoff(a, s, cfg=CFG) for s in samples]
        n1 = [payoff_n_minus_1(a, s, cfg=CFG, n_outages=16) for s in samples]
        model_w[ai] = float(np.mean(ws))
        model_r[ai] = float(np.mean(rs))
        model_c[ai] = float(np.mean(cs))
        model_dc[ai] = float(np.mean(ds))
        model_n1[ai] = float(np.mean(n1))

    return {
        "wire":      float(truth_w.max()  - truth_w[int(np.argmax(model_w))]),
        "routing":   float(truth_r.max()  - truth_r[int(np.argmax(model_r))]),
        "coverage":  float(truth_c.max()  - truth_c[int(np.argmax(model_c))]),
        "dc_flow":   float(truth_dc.max() - truth_dc[int(np.argmax(model_dc))]),
        "n_minus_1": float(truth_n1.max() - truth_n1[int(np.argmax(model_n1))]),
    }


def main() -> int:
    actions = grid_action_set(K=CFG.K, grid_size=12, tile_size=1536,
                              n_actions=20, seed=0)
    rows: list[dict] = []
    for run in RUNS:
        for city in CITIES:
            truth_path = GRAPHS / "_truth" / f"{city}.graphml"
            if not truth_path.exists():
                continue
            truth = _load_graphml(truth_path)
            row = {"run": run.replace("multi_city_hv_", ""), "city": city}
            for sampler_name, sd in SAMPLE_DIRS.items():
                samples = _load_samples(sd, run, city)
                if not samples:
                    for pname in PAYOFFS:
                        row[f"cpr_{pname}_{sampler_name}"] = float("nan")
                    continue
                cpr_vals = cpr_three_payoffs(samples, truth, actions)
                for pname, v in cpr_vals.items():
                    row[f"cpr_{pname}_{sampler_name}"] = v
            rows.append(row)
            print(f"  [{row['run']}/{row['city']}]  "
                  + "  ".join(f"{k}={v:.2f}" for k, v in row.items()
                              if k.startswith("cpr_"))[:200])

    fields = ["run", "city"] + [f"cpr_{p}_{s}"
                                 for p in PAYOFFS for s in SAMPLE_DIRS]
    out = RESULTS / "cpr_panel.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, float("nan")) for k in fields})
    print(f"\nWrote {out} ({len(rows)} cells)")

    # Cross-payoff Spearman matrix (averaged over samplers).
    print("\nCross-payoff Spearman matrix (averaged over both samplers):")
    rho_matrix: dict[tuple[str, str], float] = {}
    pnames = list(PAYOFFS.keys())
    for p1 in pnames:
        for p2 in pnames:
            rhos = []
            for s in SAMPLE_DIRS:
                vals1 = np.array([r[f"cpr_{p1}_{s}"] for r in rows])
                vals2 = np.array([r[f"cpr_{p2}_{s}"] for r in rows])
                mask = ~(np.isnan(vals1) | np.isnan(vals2))
                if mask.sum() >= 2:
                    rho = _spearman(vals1[mask], vals2[mask])
                    if not np.isnan(rho):
                        rhos.append(rho)
            rho_matrix[(p1, p2)] = float(np.mean(rhos)) if rhos else float("nan")
        print("  " + p1 + "  " + "  ".join(
            f"{p2}={rho_matrix[(p1, p2)]:+.3f}" for p2 in pnames))

    with (RESULTS / "cross_payoff_rho.json").open("w", encoding="utf-8") as f:
        json.dump(
            {f"{a}__{b}": v for (a, b), v in rho_matrix.items()},
            f, indent=2,
        )

    # Per-run summary across samplers/payoffs.
    summary: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        srow = summary.setdefault(r["run"], {f"cpr_{p}_{s}": []
                                             for p in PAYOFFS for s in SAMPLE_DIRS})
        for k, v in r.items():
            if k.startswith("cpr_") and not np.isnan(v):
                srow[k].append(v)
    sumrows = []
    for run, d in summary.items():
        sr = {"run": run}
        for k, vals in d.items():
            sr[f"mean_{k}"] = float(np.mean(vals)) if vals else float("nan")
        sumrows.append(sr)
    sum_path = RESULTS / "cpr_panel_summary.csv"
    fields = ["run"] + [f"mean_cpr_{p}_{s}" for p in PAYOFFS for s in SAMPLE_DIRS]
    sumrows.sort(key=lambda r: r["mean_cpr_routing_corr"]
                  if not np.isnan(r["mean_cpr_routing_corr"])
                  else r["mean_cpr_routing_indep"])
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sumrows:
            w.writerow({k: r.get(k, float("nan")) for k in fields})
    print(f"Wrote {sum_path}")

    print("\nPer-run mean CPR (routing payoff, both samplers):")
    for r in sumrows:
        print(f"  {r['run']:25s}  indep={r['mean_cpr_routing_indep']:7.2f}  "
              f"corr={r['mean_cpr_routing_corr']:7.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

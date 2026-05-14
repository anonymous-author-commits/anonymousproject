"""evaluate the Birchfield 2017 baseline through the CPR panel
evaluator and append its rows to results/cpr_panel.csv (the
authoritative panel).

This is the bridge between the Birchfield reproduction
(``external_baselines/birchfield_2017``) and the panel CSV that every
downstream analysis (cross-payoff matrix, capex translation,
per-city breakouts, etc.) reads. The script mirrors the protocol from
``experiments/run_cpr_panel.py`` exactly so the new rows are
numerically commensurate with the existing eleven generators.

Outputs
-------
    results/cpr_panel_birchfield.csv         (new generator only)
    results/cpr_panel.csv                    (appended; backup preserved)
"""
from __future__ import annotations

import csv
import shutil
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpr.payoff import (  # noqa: E402
    PayoffConfig, allocation_payoff, grid_action_set,
)
from cpr.payoff_routing import precompute_components, routing_mst_payoff  # noqa: E402
from cpr.payoff_coverage import coverage_payoff_interactive  # noqa: E402
from cpr.payoff_powerflow import dc_flow_payoff, payoff_n_minus_1  # noqa: E402

GRAPHS_TRUTH = ROOT / "results" / "graphs" / "_truth"
GRAPHS_MC = ROOT / "results" / "graphs_mc" / "birchfield_2017"
PANEL_CSV = ROOT / "results" / "cpr_panel.csv"
BACKUP_CSV = ROOT / "results" / "cpr_panel__pre_birchfield_backup.csv"
OUT_CSV = ROOT / "results" / "cpr_panel_birchfield.csv"

CITIES = [
    "zurich", "berlin", "chicago", "bangkok", "sao_paulo",
    "cairo", "madrid", "warsaw", "toronto", "amsterdam",
    "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco",
]
PAYOFF_KEYS = ("wire", "routing", "coverage", "dc_flow", "n_minus_1")
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
        h.add_edge(
            id_map[u], id_map[v],
            length_px=float(d.get("length_px", 0.0)),
        )
    return h


def _cpr_all_payoffs(
    samples: list[nx.MultiGraph],
    truth: nx.MultiGraph,
    actions: list,
) -> dict:
    """Five-payoff CPR for a single (run, city) cell."""
    if not samples:
        return {p: float("nan") for p in PAYOFF_KEYS}

    truth_cache = precompute_components(truth)
    sample_caches = [precompute_components(s) for s in samples]

    truth_w = np.array([allocation_payoff(a, truth, cfg=CFG) for a in actions])
    truth_r = np.array([
        routing_mst_payoff(a, truth, cfg=CFG, cache=truth_cache)
        for a in actions
    ])
    truth_c = np.array([
        coverage_payoff_interactive(a, truth, cfg=CFG) for a in actions
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
        rs = [
            routing_mst_payoff(a, samples[i], cfg=CFG, cache=sample_caches[i])
            for i in range(len(samples))
        ]
        cs = [coverage_payoff_interactive(a, s, cfg=CFG) for s in samples]
        ds = [dc_flow_payoff(a, s, cfg=CFG) for s in samples]
        n1 = [
            payoff_n_minus_1(a, s, cfg=CFG, n_outages=16) for s in samples
        ]
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
    actions = grid_action_set(
        K=CFG.K, grid_size=12, tile_size=1536, n_actions=20, seed=0,
    )
    rows: list[dict] = []
    for city in CITIES:
        t0 = time.perf_counter()
        truth_path = GRAPHS_TRUTH / f"{city}.graphml"
        sample_path = GRAPHS_MC / f"{city}_00.graphml"
        if not truth_path.exists() or not sample_path.exists():
            print(f"  [skip] {city}: missing graph(s)")
            continue
        truth = _load_graphml(truth_path)
        sample = _load_graphml(sample_path)
        results = _cpr_all_payoffs([sample], truth, actions)
        row = {"run": "birchfield_2017", "city": city}
        # Birchfield is deterministic: indep and corr columns are equal.
        for p in PAYOFF_KEYS:
            row[f"cpr_{p}_indep"] = results[p]
            row[f"cpr_{p}_corr"] = results[p]
        rows.append(row)
        dt = time.perf_counter() - t0
        print(
            f"  [{city:>15s}]  wire={results['wire']:6.1f}  "
            f"routing={results['routing']:6.1f}  "
            f"coverage={results['coverage']:6.3f}  "
            f"dc_flow={results['dc_flow']:7.3f}  "
            f"n_minus_1={results['n_minus_1']:7.3f}   ({dt:.1f}s)"
        )

    if not rows:
        print("no rows to write")
        return 1

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run", "city"] + [
        f"cpr_{p}_{s}" for p in PAYOFF_KEYS for s in ("indep", "corr")
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")

    if PANEL_CSV.exists() and not BACKUP_CSV.exists():
        shutil.copy(PANEL_CSV, BACKUP_CSV)
        print(f"Backup of cpr_panel.csv -> {BACKUP_CSV.name}")
    # Append (or merge if rows already exist).
    existing = []
    if PANEL_CSV.exists():
        with PANEL_CSV.open(newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        existing = [r for r in existing if r.get("run") != "birchfield_2017"]
    merged = existing + [
        {k: (str(v) if v is not None else "") for k, v in r.items()}
        for r in rows
    ]
    # Use header from existing (preserves column order).
    if existing:
        header = list(existing[0].keys())
    else:
        header = fieldnames
    with PANEL_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(merged)
    print(f"Updated {PANEL_CSV} (now {len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

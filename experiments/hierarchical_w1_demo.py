"""1 -- Hierarchical W1 decomposition demo.

For a panel of (truth, generator) graph pairs, compute the global
node-W1 distance and the hierarchical node-W1 at several K values;
report approximation error and wall-clock speedup as a function of
K and |V|.

The substantive question: can a planner running CPR on a continental-
scale network (10^5+ nodes) avoid the O(N^3) global OT computation
by partitioning the network into K spatial regions, computing W1
independently on each, and aggregating? At what K does the
approximation become unusable, and how much speedup is realised
before that?

Outputs
-------
    results/hierarchical_w1.csv          per (city, generator, K) row
    figures/S_hierarchical_w1.pdf        2-panel summary
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import networkx as nx
import numpy as np

from cpr.transport import (
    hierarchical_node_w1_distance,
    node_w1_distance,
)

ROOT = Path(__file__).resolve().parents[1]
TRUTH_DIR = ROOT / "results" / "graphs" / "_truth"
GEN_DIR = ROOT / "results" / "graphs_mc"
OUT_CSV = ROOT / "results" / "hierarchical_w1.csv"
OUT_FIG = ROOT / "figures" / "S_hierarchical_w1.pdf"

# A diverse cross-section of (city, generator) pairs, covering small
# and large truth graphs and both pixel-native and graph-native
# generators. We use sample index 0 for determinism.
PANEL = [
    # (city, generator_dir, sample_idx)
    ("zurich",      "digress_v1",       0),
    ("zurich",      "voronoi_density",  0),
    ("berlin",      "digress_v1",       0),
    ("berlin",      "voronoi_density",  0),
    ("chicago",     "digress_v1",       0),
    ("chicago",     "voronoi_density",  0),
    ("bangkok",     "digress_v1",       0),
    ("sao_paulo",   "digress_v1",       0),
    ("sao_paulo",   "voronoi_density",  0),
    ("madrid",      "digress_v1",       0),
    ("warsaw",      "digress_v1",       0),
    ("toronto",     "digress_v1",       0),
    ("amsterdam",   "digress_v1",       0),
    ("beijing",     "digress_v1",       0),
    ("melbourne",   "digress_v1",       0),
    ("rome",        "digress_v1",       0),
    ("san_francisco", "digress_v1",     0),
]

K_VALUES = [2, 4, 8, 16]


def _load_truth(city: str) -> nx.Graph | None:
    p = TRUTH_DIR / f"{city}.graphml"
    if not p.exists():
        return None
    return nx.read_graphml(p)


def _load_sample(generator: str, city: str, idx: int) -> nx.Graph | None:
    cands = [
        GEN_DIR / generator / f"{city}_{idx:02d}.graphml",
        GEN_DIR / generator / city / f"sample_{idx:03d}.graphml",
    ]
    for p in cands:
        if p.exists():
            return nx.read_graphml(p)
    return None


def _ensure_px(g: nx.Graph) -> nx.Graph:
    """Ensure each node has a 'px' attribute as a list of (row, col).

    GraphML readers may store px as a string; normalize to list[float].
    """
    for n in g.nodes():
        v = g.nodes[n].get("px")
        if v is None:
            v = g.nodes[n].get("px_r"), g.nodes[n].get("px_c")
        if isinstance(v, str):
            v = [float(x) for x in v.strip("[](){} ").split(",")]
        elif isinstance(v, (tuple, list)):
            v = [float(v[0]), float(v[1])]
        else:
            v = [float(n), 0.0]
        g.nodes[n]["px"] = v
    return g


def main() -> int:
    rows: list[dict] = []
    for city, gen, idx in PANEL:
        t = _load_truth(city)
        s = _load_sample(gen, city, idx)
        if t is None or s is None:
            print(f"  [skip] missing graphs for ({city}, {gen}, {idx})")
            continue
        t, s = _ensure_px(t), _ensure_px(s)
        n_truth, n_gen = t.number_of_nodes(), s.number_of_nodes()
        if n_truth < 10 or n_gen < 10:
            print(f"  [skip] too small ({city}, {gen}): |V|_t={n_truth}, |V|_g={n_gen}")
            continue

        # Global W1.
        t0 = time.perf_counter()
        w1_global = node_w1_distance(t, s, max_pts=500, seed=0)
        t_global = time.perf_counter() - t0

        # Hierarchical W1 at several K.
        for K in K_VALUES:
            t0 = time.perf_counter()
            w1_h = hierarchical_node_w1_distance(t, s, k_regions=K, seed=0)
            t_h = time.perf_counter() - t0
            err = (w1_h - w1_global) / max(abs(w1_global), 1e-9)
            row = {
                "city": city,
                "generator": gen,
                "sample_idx": idx,
                "n_truth": n_truth,
                "n_gen": n_gen,
                "K": K,
                "w1_global": w1_global,
                "w1_hier": w1_h,
                "rel_err": err,
                "t_global_s": t_global,
                "t_hier_s": t_h,
                "speedup": (t_global / t_h) if t_h > 0 else float("nan"),
            }
            rows.append(row)
            print(f"  [{city:>15s} {gen:>22s}]  K={K:2d}  "
                  f"|V|=({n_truth:4d},{n_gen:4d})  "
                  f"W1_global={w1_global:7.2f}  W1_hier={w1_h:7.2f}  "
                  f"err={err:+.3f}  speedup={row['speedup']:.2f}x")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {OUT_CSV} ({len(rows)} rows)")

    # Render the 2-panel summary figure.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return 0

    if not rows:
        print("no data; skipping figure")
        return 1

    import numpy as np
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                     constrained_layout=True)

    for K in K_VALUES:
        sub = [r for r in rows if r["K"] == K]
        if not sub:
            continue
        errs = np.array([abs(r["rel_err"]) * 100 for r in sub])
        sizes = np.array([r["n_truth"] for r in sub])
        ax1.scatter(sizes, errs, label=f"K={K}", alpha=0.7, s=40)
    ax1.set_xlabel("Truth graph |V|")
    ax1.set_ylabel("|W$_1^{\\mathrm{hier}}$ - W$_1^{\\mathrm{global}}$| / W$_1^{\\mathrm{global}}$  [%]")
    ax1.set_title("Approximation error vs panel cell size")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    for K in K_VALUES:
        sub = [r for r in rows if r["K"] == K]
        if not sub:
            continue
        speeds = np.array([r["speedup"] for r in sub])
        sizes = np.array([r["n_truth"] for r in sub])
        ax2.scatter(sizes, speeds, label=f"K={K}", alpha=0.7, s=40)
    ax2.axhline(1.0, color="black", lw=0.5, ls="--")
    ax2.set_xlabel("Truth graph |V|")
    ax2.set_ylabel("Wall-clock speedup over global W$_1$")
    ax2.set_title("Speedup vs panel cell size")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")

    print("\nPer-K summary:")
    for K in K_VALUES:
        sub = [r for r in rows if r["K"] == K]
        if not sub:
            continue
        errs = np.array([abs(r["rel_err"]) * 100 for r in sub])
        speeds = np.array([r["speedup"] for r in sub])
        print(f"  K={K:2d}  n={len(sub)}  "
              f"mean_err={errs.mean():5.1f}%  max_err={errs.max():5.1f}%  "
              f"mean_speedup={speeds.mean():4.2f}x  max_speedup={speeds.max():4.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

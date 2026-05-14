"""Visual galleries: 5-panel grid per city showing the
truth + four (CPR, Dice) quadrant exemplars.

Reproducible quadrant assignment (no manual cherry-picking):
  1. For each of the 6 DUPT generators (the only ones with Dice
     scores), look up its (Dice, mean-routing-CPR_per_city).
  2. Define the four quadrants by the median split of Dice and CPR.
  3. For each quadrant, pick the generator whose (Dice, CPR) is
     closest to the quadrant's centroid in z-score space (Manhattan).
  4. Render five panels (truth + 4 quadrant exemplars) using
     networkx + matplotlib.

Quadrant labels (per the paper's framing):
  - High-CPR, Low-Dice  : "bad on both" (worst)
  - Low-CPR, High-Dice  : "good on both" (best)
  - Low-CPR, Low-Dice   : "useful but ugly" (the headline finding)
  - High-CPR, High-Dice : "pretty but useless" (the headline finding)

Outputs:
    figures/gallery_<city>.pdf for each --cities argument.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GRAPHS_TRUTH = RESULTS / "graphs" / "_truth"
GRAPHS_MC = RESULTS / "graphs_mc"
FIGS = ROOT / "figures"

DICE_BY_RUN = {
    "v2": 0.078, "v3": 0.077, "cgan_v1": 0.080,
    "cgan_v2": 0.043, "v2_6ch": 0.085, "cgan_v3": 0.091,
}
RUN_FOLDER = {
    "v2": "multi_city_hv_v2", "v3": "multi_city_hv_v3",
    "v2_6ch": "multi_city_hv_v2_6ch",
    "cgan_v1": "multi_city_hv_cgan_v1",
    "cgan_v2": "multi_city_hv_cgan_v2",
    "cgan_v3": "multi_city_hv_cgan_v3",
}


def _load_graphml(path: Path):
    if not path.exists():
        return [], []
    g = nx.read_graphml(str(path))
    nodes = []
    coord_map: dict[str, tuple[int, int]] = {}
    for n, d in g.nodes(data=True):
        try:
            r = int(float(d.get("px_r", 0)))
            c = int(float(d.get("px_c", 0)))
        except (TypeError, ValueError):
            continue
        coord_map[str(n)] = (r, c)
        nodes.append((r, c))
    edges = []
    for u, v in g.edges():
        if str(u) in coord_map and str(v) in coord_map:
            edges.append((coord_map[str(u)], coord_map[str(v)]))
    return nodes, edges


def _draw(ax, nodes, edges, title, color="#14375E"):
    if edges:
        for (r1, c1), (r2, c2) in edges:
            ax.plot([c1, c2], [r1, r2], color=color, lw=0.9, alpha=0.8)
    if nodes:
        ns = np.array(nodes)
        ax.scatter(ns[:, 1], ns[:, 0], s=8, color=color,
                   edgecolors="black", lw=0.3, zorder=5)
    ax.set_xlim(0, 1536); ax.set_ylim(1536, 0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def _quadrant_picks(panel: pd.DataFrame, city: str
                       ) -> dict[str, tuple[str, float, float]]:
    """For city, return {quadrant_label: (run, dice, cpr)}."""
    row_dict: dict[str, tuple[float, float]] = {}
    sub = panel[panel["city"] == city]
    for run in DICE_BY_RUN:
        cell = sub[sub["run"] == run]
        if cell.empty:
            continue
        cpr_i = float(cell["cpr_routing_indep"].iloc[0])
        cpr_c = float(cell["cpr_routing_corr"].iloc[0]) if "cpr_routing_corr" in cell else float("nan")
        cpr = cpr_i if np.isnan(cpr_c) else (cpr_i + cpr_c) / 2.0
        row_dict[run] = (DICE_BY_RUN[run], cpr)
    if len(row_dict) < 4:
        return {}
    runs = list(row_dict.keys())
    dices = np.array([row_dict[r][0] for r in runs])
    cprs = np.array([row_dict[r][1] for r in runs])
    med_d = float(np.median(dices))
    med_c = float(np.median(cprs))
    # z-score for centroid distance.
    sd_d = max(float(dices.std()), 1e-9)
    sd_c = max(float(cprs.std()), 1e-9)

    quadrants = {
        "high_cpr_low_dice": (med_d - sd_d, med_c + sd_c),
        "low_cpr_high_dice": (med_d + sd_d, med_c - sd_c),
        "low_cpr_low_dice":  (med_d - sd_d, med_c - sd_c),
        "high_cpr_high_dice": (med_d + sd_d, med_c + sd_c),
    }
    out: dict[str, tuple[str, float, float]] = {}
    used: set[str] = set()
    # Process quadrants in priority order: paper's two headline-finding
    # cells first (so we pick the most-representative for them, leaving
    # the leftovers for the less-narrative-critical quadrants).
    priority = ["high_cpr_high_dice", "low_cpr_low_dice",
                "low_cpr_high_dice", "high_cpr_low_dice"]
    for label in priority:
        target_d, target_c = quadrants[label]
        # Strict quadrant filter.
        if "high_cpr" in label:
            cand = [r for r in runs if row_dict[r][1] >= med_c]
        else:
            cand = [r for r in runs if row_dict[r][1] < med_c]
        if "low_dice" in label:
            cand = [r for r in cand if row_dict[r][0] < med_d]
        else:
            cand = [r for r in cand if row_dict[r][0] >= med_d]
        # Exclude already-used runs.
        cand = [r for r in cand if r not in used]
        if not cand:
            # Fallback: pick from any unused run, closest to target.
            cand = [r for r in runs if r not in used]
        if not cand:
            # Genuinely no unused runs left; allow re-use as a last resort.
            cand = runs

        def _dist(r):
            d, c = row_dict[r]
            return abs(d - target_d) / sd_d + abs(c - target_c) / sd_c
        best = min(cand, key=_dist)
        out[label] = (best, row_dict[best][0], row_dict[best][1])
        used.add(best)
    return out


def render_city_gallery(panel: pd.DataFrame, city: str) -> Path | None:
    picks = _quadrant_picks(panel, city)
    if not picks:
        print(f"  [{city}] insufficient data for quadrants.")
        return None

    # Load truth + 4 picks.
    truth_n, truth_e = _load_graphml(GRAPHS_TRUTH / f"{city}.graphml")
    panels = [("Ground truth (OSM)", truth_n, truth_e, "#14375E", None, None)]
    label_titles = {
        "high_cpr_high_dice": "Pretty but useless\n(High CPR, High Dice)",
        "low_cpr_low_dice":   "Useful but ugly\n(Low CPR, Low Dice)",
        "low_cpr_high_dice":  "Good on both\n(Low CPR, High Dice)",
        "high_cpr_low_dice":  "Bad on both\n(High CPR, Low Dice)",
    }
    # Order: truth, pretty-but-useless, useful-but-ugly, good-both, bad-both.
    layout = ["high_cpr_high_dice", "low_cpr_low_dice",
              "low_cpr_high_dice", "high_cpr_low_dice"]
    for label in layout:
        run, dice, cpr = picks[label]
        folder = RUN_FOLDER.get(run, run)
        path = GRAPHS_MC / folder / f"{city}_00.graphml"
        nodes, edges = _load_graphml(path)
        title = (f"{label_titles[label]}\n"
                  f"{run}: Dice={dice:.3f}, CPR={cpr:.1f}")
        color = ("#d62728" if "high_cpr" in label and "high_dice" in label
                 else "#2ca02c" if "low_cpr" in label and "high_dice" in label
                 else "#ff7f0e")
        panels.append((title, nodes, edges, color, run, label))

    fig, axes = plt.subplots(1, 5, figsize=(18, 4),
                              constrained_layout=True)
    for ax, p in zip(axes, panels):
        _draw(ax, p[1], p[2], p[0], color=p[3])

    fig.suptitle(f"{city.replace('_', ' ').title()}: gallery of "
                  f"truth + 4 (CPR, Dice) quadrants", fontsize=12)
    out = FIGS / f"gallery_{city}.pdf"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{city}] picks:")
    for label, (run, d, c) in picks.items():
        print(f"    {label:>22s}: {run:>10s} (Dice={d:.3f}, CPR={c:.1f})")
    print(f"  [{city}] wrote {out}")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="*",
                        default=["sao_paulo", "zurich"])
    args = parser.parse_args(argv)

    panel = pd.read_csv(RESULTS / "cpr_panel.csv")
    for city in args.cities:
        render_city_gallery(panel, city)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

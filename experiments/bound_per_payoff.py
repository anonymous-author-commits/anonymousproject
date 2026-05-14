"""5-panel scatter of CPR vs L*W1 per payoff (bound informativity).

For each canonical payoff p in {wire, routing, coverage, dc_flow,
n_minus_1}, we plot per-cell (L_p * W1, CPR_p) and overlay the
y = x diagonal. The bound informativity ratio CPR / (L*W1) is the
slope of the data cloud below the diagonal: closer to the diagonal
means a tighter bound.

Inputs
------
    results/lipschitz_estimates.json   (per-payoff L_p95)
    results/cpr_panel.csv              (per-cell CPR per payoff)
    results/bound_table_node_w1.csv    (per-cell node-W1; routing-base)

Output
------
    figures/bound_per_payoff.pdf
    results/bound_per_payoff_summary.csv  (per-payoff ratio mean, max, p95)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
PRETTY = {"wire": "Wire", "routing": "Routing", "coverage": "Coverage",
          "dc_flow": "DC-flow", "n_minus_1": r"$N{-}1$"}


def _load_lipschitz() -> dict[str, float]:
    p = RESULTS / "lipschitz_estimates.json"
    if not p.exists():
        print(f"[error] {p} not found. Run experiments.lipschitz_estimate first.")
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v["p95"]) for k, v in data.items()}


def _load_cpr_panel() -> list[dict]:
    p = RESULTS / "cpr_panel.csv"
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_node_w1() -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    p = RESULTS / "bound_table_node_w1.csv"
    with p.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                w = float(r["w1_node"])
            except (KeyError, ValueError):
                continue
            out[(r["run"], r["city"])] = w
    return out


def main() -> int:
    L = _load_lipschitz()
    if not L:
        return 1
    panel = _load_cpr_panel()
    w1 = _load_node_w1()

    # For each payoff and each cell, average over indep + corr samplers.
    rows = []
    for r in panel:
        run, city = r["run"], r["city"]
        if (run, city) not in w1:
            continue
        w = w1[(run, city)]
        for p in PAYOFFS:
            try:
                cpr_i = float(r.get(f"cpr_{p}_indep", "nan"))
                cpr_c = float(r.get(f"cpr_{p}_corr", "nan"))
            except ValueError:
                continue
            cprs = [v for v in (cpr_i, cpr_c) if np.isfinite(v)]
            if not cprs:
                continue
            cpr = float(np.mean(cprs))
            bound = L[p] * w
            ratio = cpr / max(bound, 1e-9)
            rows.append({
                "run": run, "city": city, "payoff": p,
                "cpr": cpr, "w1": w, "L": L[p],
                "bound": bound, "ratio": ratio,
            })

    by_payoff: dict[str, list[dict]] = {p: [] for p in PAYOFFS}
    for r in rows:
        by_payoff[r["payoff"]].append(r)

    # Summary CSV.
    summary_rows = []
    for p in PAYOFFS:
        ratios = np.array([r["ratio"] for r in by_payoff[p]])
        if ratios.size == 0:
            continue
        summary_rows.append({
            "payoff": p,
            "n_cells": int(ratios.size),
            "L_p95": L[p],
            "ratio_mean": float(np.mean(ratios)),
            "ratio_median": float(np.median(ratios)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "ratio_max": float(np.max(ratios)),
            "frac_holding": float(np.mean(ratios <= 1.0)),
        })
    out_csv = RESULTS / "bound_per_payoff_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w_csv = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w_csv.writeheader()
        for r in summary_rows:
            w_csv.writerow(r)
    print(f"Wrote {out_csv}")
    print()
    for r in summary_rows:
        print(f"  [{r['payoff']:>10s}] L_p95={r['L_p95']:.3e}  "
              f"ratio mean={r['ratio_mean']:.3f}  p95={r['ratio_p95']:.3f}  "
              f"max={r['ratio_max']:.3f}  holds={100*r['frac_holding']:.1f}%")

    # 5-panel figure.
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(PAYOFFS), figsize=(15, 3.6),
                              constrained_layout=True)
    for ax, p in zip(axes, PAYOFFS):
        d = by_payoff[p]
        if not d:
            ax.set_title(f"{PRETTY[p]} (no data)")
            continue
        bx = np.array([r["bound"] for r in d])
        cy = np.array([r["cpr"] for r in d])
        ax.scatter(bx, cy, s=14, alpha=0.7, color="#14375E")
        lo = float(min(bx.min(), cy.min())) * 0.5
        hi = float(max(bx.max(), cy.max())) * 1.2
        if lo <= 0:
            lo = 1e-6
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6, label="$y = x$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\hat L_{p95} \cdot W_1$ (bound)")
        ax.set_ylabel("CPR")
        s = next(s for s in summary_rows if s["payoff"] == p)
        ax.set_title(f"{PRETTY[p]}\nratio mean={s['ratio_mean']:.2f}, "
                     f"holds={100*s['frac_holding']:.0f}%")
        ax.grid(True, alpha=0.3, which="both")
    out = FIGS / "bound_per_payoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

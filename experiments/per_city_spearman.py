"""Per-city Spearman decomposition of the 5x5 cross-payoff matrix.

Computes one cross-payoff matrix per held-out city (5 separate
matrices) to test whether the matrix changes with morphology.

Hypothesis: dense regular cities (Berlin, Chicago) have higher
cross-payoff correlations; sparse / informal cities (Bangkok,
Sao Paulo) have lower correlations because each payoff stresses
different aspects of sparsity.

Outputs
-------
    results/cross_payoff_per_city.json -- {city: {p1__p2: rho}}
    figures/S_per_city_xpayoff.pdf     -- 5-panel matrix
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
SAMPLERS = ["indep", "corr"]
CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or a.size != b.size:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean()
    br -= br.mean()
    denom = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / denom) if denom > 0 else float("nan")


def _city_matrix(rows: list[dict], city: str) -> dict[tuple[str, str], float]:
    sub = [r for r in rows if r["city"] == city]
    out: dict[tuple[str, str], float] = {}
    for p1 in PAYOFFS:
        for p2 in PAYOFFS:
            rhos = []
            for s in SAMPLERS:
                _F = lambda x: (float(x) if x != "" else float("nan")) if not isinstance(x, float) else x
                v1 = np.array([_F(r[f"cpr_{p1}_{s}"]) for r in sub])
                v2 = np.array([_F(r[f"cpr_{p2}_{s}"]) for r in sub])
                m = ~(np.isnan(v1) | np.isnan(v2))
                if m.sum() >= 2:
                    rho = _spearman(v1[m], v2[m])
                    if not np.isnan(rho):
                        rhos.append(rho)
            out[(p1, p2)] = float(np.mean(rhos)) if rhos else float("nan")
    return out


def main() -> int:
    csv_path = RESULTS / "cpr_panel.csv"
    if not csv_path.exists():
        print(f"[error] {csv_path} not found.")
        return 1
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} cells.")

    out_data: dict[str, dict[str, float]] = {}
    mean_off_diag: dict[str, float] = {}
    for city in CITIES:
        m = _city_matrix(rows, city)
        out_data[city] = {f"{p1}__{p2}": float(v) for (p1, p2), v in m.items()}
        offs = [v for (p1, p2), v in m.items()
                if p1 != p2 and not np.isnan(v)]
        mean_off_diag[city] = float(np.mean(offs)) if offs else float("nan")
        print(f"  [{city:>10s}]  mean off-diagonal rho = "
              f"{mean_off_diag[city]:+.3f}  (n_cells={len(offs)})")

    out = RESULTS / "cross_payoff_per_city.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({"matrices": out_data,
                   "mean_off_diag": mean_off_diag}, f, indent=2)
    print(f"\nWrote {out}")

    # Try to render the 5-panel figure if matplotlib is available.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping figure")
        return 0

    pretty = {"wire": "Wire", "routing": "Rout.", "coverage": "Cov.",
              "dc_flow": "DC", "n_minus_1": r"$N{-}1$"}
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CITIES), figsize=(15, 3.6),
                              constrained_layout=True)
    for ax, city in zip(axes, CITIES):
        m = _city_matrix(rows, city)
        M = np.array([[m[(p1, p2)] for p2 in PAYOFFS] for p1 in PAYOFFS])
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
        ax.set_xticks(range(len(PAYOFFS)))
        ax.set_yticks(range(len(PAYOFFS)))
        ax.set_xticklabels([pretty[p] for p in PAYOFFS], rotation=45, fontsize=8)
        ax.set_yticklabels([pretty[p] for p in PAYOFFS], fontsize=8)
        title = city.replace("_", " ").title()
        ax.set_title(f"{title}\n$\\bar\\rho$={mean_off_diag[city]:+.2f}",
                     fontsize=10)
        for i in range(len(PAYOFFS)):
            for j in range(len(PAYOFFS)):
                v = M[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=axes, shrink=0.7, label=r"Spearman $\rho$")
    out_fig = FIGS / "S_per_city_xpayoff.pdf"
    fig.savefig(out_fig, bbox_inches="tight")
    print(f"Wrote {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Per-city breakouts.

Emits the full (city, run, payoff) panel as a single tidy CSV plus a
per-payoff heatmap figure. Lets a reviewer verify that aggregate
results are not driven by a single outlier city.

Outputs:
    results/per_city_breakout.csv -- tidy per (city, run, payoff) CPR
    results/per_city_top_bottom.csv -- per-city top-3 / bottom-3 generators
    figures/per_city_heatmap.pdf -- 5-payoff heatmap (run x city)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]
PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
PRETTY = {"wire": "Wire", "routing": "Routing", "coverage": "Coverage",
          "dc_flow": "DC-flow", "n_minus_1": r"$N{-}1$"}


def main() -> int:
    panel = pd.read_csv(RESULTS / "cpr_panel.csv")

    rows = []
    for _, r in panel.iterrows():
        for p in PAYOFFS:
            ci = float(r.get(f"cpr_{p}_indep", float("nan")))
            cc = float(r.get(f"cpr_{p}_corr", float("nan")))
            cpr = ci if np.isnan(cc) else (ci + cc) / 2.0
            rows.append({
                "city": r["city"], "run": r["run"], "payoff": p,
                "cpr_indep": ci, "cpr_corr": cc, "cpr_avg": cpr,
            })
    tidy = pd.DataFrame(rows)
    out = RESULTS / "per_city_breakout.csv"
    tidy.to_csv(out, index=False)
    print(f"Wrote {out} ({len(tidy)} rows)")

    # Per-city top-3 / bottom-3 by routing-CPR.
    summary = []
    for city in CITIES:
        for payoff in PAYOFFS:
            sub = tidy[(tidy["city"] == city)
                         & (tidy["payoff"] == payoff)
                         & tidy["cpr_avg"].notna()]
            if sub.empty:
                continue
            sub = sub.sort_values("cpr_avg")
            for rank, (_, r) in enumerate(sub.head(3).iterrows(), start=1):
                summary.append({"city": city, "payoff": payoff,
                                  "rank": rank, "kind": "best",
                                  "run": r["run"], "cpr": r["cpr_avg"]})
            for rank, (_, r) in enumerate(sub.tail(3).iloc[::-1].iterrows(),
                                              start=1):
                summary.append({"city": city, "payoff": payoff,
                                  "rank": rank, "kind": "worst",
                                  "run": r["run"], "cpr": r["cpr_avg"]})
    s_df = pd.DataFrame(summary)
    s_path = RESULTS / "per_city_top_bottom.csv"
    s_df.to_csv(s_path, index=False)
    print(f"Wrote {s_path}")

    # Per-payoff heatmap (run x city).
    runs = sorted(tidy["run"].unique())
    fig, axes = plt.subplots(1, len(PAYOFFS),
                              figsize=(4.0 * len(PAYOFFS), 4.5),
                              constrained_layout=True)
    for ax, payoff in zip(axes, PAYOFFS):
        mat = np.full((len(runs), len(CITIES)), np.nan)
        for i, run in enumerate(runs):
            for j, city in enumerate(CITIES):
                cell = tidy[(tidy["run"] == run)
                              & (tidy["city"] == city)
                              & (tidy["payoff"] == payoff)]
                if not cell.empty:
                    mat[i, j] = float(cell["cpr_avg"].iloc[0])
        # Normalize per-payoff so the colour scale is comparable.
        finite = mat[np.isfinite(mat)]
        if finite.size > 0:
            vmin, vmax = float(finite.min()), float(finite.max())
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r",
                        vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(CITIES)))
        ax.set_xticklabels([c[:6] for c in CITIES],
                            rotation=45, fontsize=8)
        ax.set_yticks(range(len(runs)))
        ax.set_yticklabels(runs, fontsize=7)
        ax.set_title(PRETTY[payoff], fontsize=10)
        for i in range(len(runs)):
            for j in range(len(CITIES)):
                v = mat[i, j]
                if np.isfinite(v):
                    color = "white" if abs(v - vmin) > (vmax - vmin) * 0.5 else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=6, color=color)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Per-city CPR breakout: every (run, city) cell, "
                  "per payoff", fontsize=11)
    out_pdf = FIGS / "per_city_heatmap.pdf"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_pdf}")

    # Per-city aggregate: which generator wins on average for each city?
    print()
    print("Per-city best (lowest mean CPR across all 5 payoffs):")
    for city in CITIES:
        sub = tidy[tidy["city"] == city]
        # Mean CPR per run across payoffs (z-score normalised within payoff).
        sub_z = sub.copy()
        for p in PAYOFFS:
            mask = sub_z["payoff"] == p
            vals = sub_z.loc[mask, "cpr_avg"]
            if vals.notna().sum() > 0:
                m = vals.mean()
                s = vals.std() if vals.std() > 0 else 1.0
                sub_z.loc[mask, "z"] = (vals - m) / s
        agg = sub_z.groupby("run")["z"].mean().sort_values()
        print(f"  {city:>10s}: best = {agg.index[0]:>20s} "
              f"(z = {agg.iloc[0]:+.2f})  "
              f"worst = {agg.index[-1]:>20s} (z = {agg.iloc[-1]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

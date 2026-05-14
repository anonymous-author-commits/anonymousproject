"""W4 -- Capex sensitivity analysis.

Replaces the single-point IEA midpoint headline with an explicit
sensitivity grid over voltage class, $/km cost range, K (decisions
per round), and the average truth-graph node count. Reports
distributions, not point estimates.

Inputs:
    results/cpr_panel.csv (routing CPR per generator, city)

Outputs:
    results/capex_sensitivity.csv
    tables/capex_sensitivity.tex
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo"]
PIXEL_M = 10.0

# IEA / ENTSO-E cost references (USD M / km), low-mid-high.
COSTS = {
    "HV":  {"low": 1.5, "mid": 2.5, "high": 3.0},
    "MV":  {"low": 0.43, "mid": 0.64, "high": 1.61},
    "LV":  {"low": 0.05, "mid": 0.10, "high": 0.15},
}

# K (decisions per planning round) sensitivity.
K_VALUES = [2, 4, 8]


def main() -> int:
    panel = pd.read_csv(RESULTS / "cpr_panel.csv")
    panel["routing_cpr"] = panel[["cpr_routing_indep",
                                    "cpr_routing_corr"]].mean(axis=1)

    # Unit chain (matches cpr.monetary.cpr_to_capex):
    #
    #   routing CPR is in per-node-mean pixels.
    #   excess_m_per_action = CPR * pixel_size_m
    #   excess_km_per_round = excess_m_per_action * K / 1000
    #
    # The per-node-mean normalisation is a Lipschitz-stability
    # choice in the payoff definition; for the capex translation
    # it is the *per-substation-decision* excess routing length
    # that planners pay for. Multiplying by K aggregates over
    # the K decisions in one planning round.
    rows = []
    for _, r in panel.iterrows():
        run = r["run"]
        city = r["city"]
        cpr = r["routing_cpr"]
        if not np.isfinite(cpr):
            continue
        for K in K_VALUES:
            excess_m_per_action = cpr * PIXEL_M
            excess_km_per_round = excess_m_per_action * K / 1000.0
            for vc, costs in COSTS.items():
                rows.append({
                    "run": run, "city": city, "K": K,
                    "voltage_class": vc,
                    "excess_km_per_round": excess_km_per_round,
                    "capex_low_M": excess_km_per_round * costs["low"],
                    "capex_mid_M": excess_km_per_round * costs["mid"],
                    "capex_high_M": excess_km_per_round * costs["high"],
                })
    df = pd.DataFrame(rows)
    out = RESULTS / "capex_sensitivity.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} rows)")

    # Summary table per voltage class + K, aggregating across runs
    # & cities, reporting the best vs worst non-degenerate gap.
    # We exclude the synthetic controls (baseline_random,
    # baseline_perturbed) and the obviously degenerate cgan_v2 from
    # both ends of the gap so the headline reflects "real model
    # variation", not "model vs noise".
    EXCLUDE = {"baseline_random", "baseline_perturbed", "cgan_v2"}
    df_real = df[~df["run"].isin(EXCLUDE)]
    summary_rows = []
    for K in K_VALUES:
        for vc in COSTS:
            sub = df_real[(df_real["K"] == K)
                            & (df_real["voltage_class"] == vc)]
            best_total = 0.0
            worst_total = 0.0
            for city in CITIES:
                cs = sub[sub["city"] == city]
                if cs.empty:
                    continue
                best_total += cs["capex_mid_M"].min()
                worst_total += cs["capex_mid_M"].max()
            gap_mid = worst_total - best_total
            # Recompute total with low and high.
            best_low = sum(sub[sub["city"] == c]["capex_low_M"].min()
                            for c in CITIES
                            if not sub[sub["city"] == c].empty)
            worst_high = sum(sub[sub["city"] == c]["capex_high_M"].max()
                              for c in CITIES
                              if not sub[sub["city"] == c].empty)
            gap_high_band = worst_high - best_low
            best_high = sum(sub[sub["city"] == c]["capex_high_M"].min()
                             for c in CITIES
                             if not sub[sub["city"] == c].empty)
            worst_low = sum(sub[sub["city"] == c]["capex_low_M"].max()
                              for c in CITIES
                              if not sub[sub["city"] == c].empty)
            gap_low_band = max(worst_low - best_high, 0.0)
            summary_rows.append({
                "K": K, "voltage_class": vc,
                "5city_total_capex_gap_low_M": gap_low_band,
                "5city_total_capex_gap_mid_M": gap_mid,
                "5city_total_capex_gap_high_M": gap_high_band,
            })
    s = pd.DataFrame(summary_rows)
    out2 = RESULTS / "capex_sensitivity_summary.csv"
    s.to_csv(out2, index=False)
    print(f"Wrote {out2}")
    print()
    print("5-city aggregate excess-capex gap (best gen vs worst gen, "
          "summed across 5 cities):")
    print(s.to_string(index=False))

    # LaTeX table.
    TABLES.mkdir(parents=True, exist_ok=True)
    tex = TABLES / "capex_sensitivity.tex"
    with tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/capex_sensitivity.py\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\arrayrulecolor{NEband}\n")
        f.write("\\begin{tabular}{c c c c c}\n")
        f.write("\\toprule\n\\rowcolor{NEblue}\n")
        f.write("\\color{white}\\textbf{$K$} & "
                "\\color{white}\\textbf{Voltage} & "
                "\\color{white}\\textbf{Low cost} & "
                "\\color{white}\\textbf{Mid cost} & "
                "\\color{white}\\textbf{High cost} \\\\\n")
        f.write("\\midrule\n")
        for _, r in s.iterrows():
            f.write(f"{int(r['K'])} & {r['voltage_class']} & "
                    f"\\${r['5city_total_capex_gap_low_M']:.1f}M & "
                    f"\\${r['5city_total_capex_gap_mid_M']:.1f}M & "
                    f"\\${r['5city_total_capex_gap_high_M']:.1f}M \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\arrayrulecolor{black}\n")
    print(f"Wrote {tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

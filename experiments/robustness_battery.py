"""W1, W3, W7, W10 robustness battery.

Reads cpr_panel.csv and cpr_multiseed.csv. Computes:

  W1 (taxonomy): emits a model_taxonomy.tex /.csv classifying each
       run as published-vs-internal-variant, architecture, loss,
       conditioning, and degeneracy status.

  W3 (non-degenerate Spearman): repeats the headline Dice <-> CPR
       Spearman with degenerate generators removed. Degeneracy is
       defined quantitatively: predicted-foreground fraction outside
       [0.005, 0.20], or per-city node count more than 5x the truth
       median, or zero edges in any held-out city.

  W7 (model-selection experiment): for each metric in {Dice, W1,
       routing-CPR, wire-CPR}, pick the top-1 generator on a leave-
       one-city-out training fold, evaluate its routing-CPR on the
       held-out city, report mean deployed-CPR per selector. The
       selector with lowest deployed-CPR is the most useful for
       downstream planning.

  W10 (sensitivity): leave-one-city-out Dice<->CPR Spearman; K
       sensitivity (re-uses panel where possible); Bernoulli vs
       Gaussian-copula sampler comparison; threshold-sensitivity
       table for digress_v1 (re-uses the panel rows with the
       calibrated threshold and prints the truth-edge-count
       it was matched against).

Outputs (under results/):
  model_taxonomy.csv              W1
  cpr_dice_nondegenerate.json     W3
  selector_experiment.csv         W7
  loco_dice_cpr_rho.csv           W10
  sampler_compare.csv             W10
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
GRAPHS_MC = RESULTS / "graphs_mc"
GRAPHS_TRUTH = RESULTS / "graphs" / "_truth"

CITIES = ["zurich", "berlin", "chicago", "bangkok", "sao_paulo", "cairo", "madrid", "warsaw", "toronto", "amsterdam", "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco"]
PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]


# ---------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------

def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or a.size != b.size:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    ar -= ar.mean(); br -= br.mean()
    den = np.sqrt((ar * ar).sum() * (br * br).sum())
    return float((ar * br).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------
# Truth edge counts (for degeneracy + threshold context)
# ---------------------------------------------------------------------

def _truth_edge_count(city: str) -> int:
    p = GRAPHS_TRUTH / f"{city}.graphml"
    if not p.exists():
        return 0
    g = nx.read_graphml(str(p))
    return g.number_of_edges()


def _gen_edge_count(run: str, city: str, sample_idx: int = 0) -> int:
    """Resolve the panel run name to the graphs_mc folder name.

    DUPT runs are stored under multi_city_hv_<short>; non-DUPT runs
    are stored under their short name directly.
    """
    candidates = [
        GRAPHS_MC / run / f"{city}_{sample_idx:02d}.graphml",
        GRAPHS_MC / f"multi_city_hv_{run}" / f"{city}_{sample_idx:02d}.graphml",
    ]
    for p in candidates:
        if p.exists():
            return nx.read_graphml(str(p)).number_of_edges()
    return -1


# ---------------------------------------------------------------------
# W1: Model taxonomy
# ---------------------------------------------------------------------

# Architecture / loss / conditioning / source per run.
# Distinguishes "internal variant" (DUPT family, this project) from
# "external published" (none, currently) from "deterministic baseline"
# from "synthetic control".
TAXONOMY = {
    "v2": {
        "name": "DUPT-v2 (mask-cls)",
        "architecture": "UNet (mask classification)",
        "loss": "weighted-BCE",
        "conditioning": "6-channel raster (built-up, water, roads, "
                         "rails, elevation, OSM substations)",
        "source": "internal variant (this project, DUPT/training/)",
        "published": "no",
    },
    "v3": {
        "name": "DUPT-v3 (mask-cls + sched. BCE)",
        "architecture": "UNet (mask classification)",
        "loss": "scheduled-weight BCE",
        "conditioning": "6-channel raster",
        "source": "internal variant (this project, DUPT/training/)",
        "published": "no",
    },
    "v2_6ch": {
        "name": "DUPT-v2-6ch (mask-cls, dropout)",
        "architecture": "UNet w/ dropout",
        "loss": "weighted-BCE",
        "conditioning": "6-channel raster",
        "source": "internal variant",
        "published": "no",
    },
    "cgan_v1": {
        "name": "DUPT-cgan-v1 (PatchGAN, 70px RF)",
        "architecture": "U-Net G + PatchGAN D (70-px receptive field)",
        "loss": "adversarial + L1 reconstruction",
        "conditioning": "6-channel raster",
        "source": "internal variant",
        "published": "no",
    },
    "cgan_v2": {
        "name": "DUPT-cgan-v2 (global-head)",
        "architecture": "U-Net G + global-head D",
        "loss": "adversarial + L1",
        "conditioning": "6-channel raster",
        "source": "internal variant",
        "published": "no",
    },
    "cgan_v3": {
        "name": "DUPT-cgan-v3 (PatchGAN, richer cond.)",
        "architecture": "U-Net G + PatchGAN D",
        "loss": "adversarial + L1 + perceptual",
        "conditioning": "6-channel raster",
        "source": "internal variant",
        "published": "no",
    },
    "digress_v1": {
        "name": "DiGress (graph diffusion)",
        "architecture": "Transformer-on-graph denoiser, 4.92M params",
        "loss": "BCE on edges + L2 on coords; raster cross-attention",
        "conditioning": "6-channel raster (cross-attention)",
        "source": "this project, adapted from Vignac et al. 2023",
        "published": "no (adapted from published architecture)",
    },
    "voronoi_density": {
        "name": "Voronoi (density partition)",
        "architecture": "deterministic; Voronoi over built-up density",
        "loss": "n/a (no training)",
        "conditioning": "built-up raster",
        "source": "deterministic baseline (this project)",
        "published": "no",
    },
    "mst_substations": {
        "name": "MST (Euclidean over substations)",
        "architecture": "deterministic; Euclidean MST over OSM subs",
        "loss": "n/a",
        "conditioning": "OSM substation locations only",
        "source": "deterministic baseline (this project)",
        "published": "no",
    },
    "baseline_random": {
        "name": "Random control",
        "architecture": "uniform-random graph at truth-edge density",
        "loss": "n/a",
        "conditioning": "none",
        "source": "synthetic control",
        "published": "no",
    },
    "baseline_perturbed": {
        "name": "Perturbed-truth control",
        "architecture": "truth graph + Gaussian coordinate noise",
        "loss": "n/a",
        "conditioning": "truth graph",
        "source": "synthetic control",
        "published": "no",
    },
    "birchfield_2017": {
        "name": "Birchfield et al.\\ 2017 (synthetic-network)",
        "architecture": "weighted k-means siting + k=3 NN edges",
        "loss": "n/a (procedural, no training)",
        "conditioning": "built-up density (ESA WorldCover)",
        "source": "external reproduction (this work) of Birchfield"
                   " et al.\\ 2017 IEEE TPS",
        "published": "yes (algorithm; reproduction by this study)",
    },
}


def write_taxonomy() -> Path:
    rows = []
    for run, t in TAXONOMY.items():
        rows.append({"run": run, **t})
    df = pd.DataFrame(rows)
    out = RESULTS / "model_taxonomy.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    # LaTeX rendering.
    tex = TABLES / "model_taxonomy.tex"
    TABLES.mkdir(parents=True, exist_ok=True)
    with tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/robustness_battery.py\n")
        f.write("\\renewcommand{\\arraystretch}{1.18}\n")
        f.write("\\arrayrulecolor{NEband}\n")
        f.write("\\begin{tabular}{l p{3.0cm} p{3.5cm} p{1.4cm}}\n")
        f.write("\\toprule\n\\rowcolor{NEblue}\n")
        f.write("\\color{white}\\textbf{Run} & "
                "\\color{white}\\textbf{Architecture} & "
                "\\color{white}\\textbf{Loss / training} & "
                "\\color{white}\\textbf{Source} \\\\\n")
        f.write("\\midrule\n")
        for run, t in TAXONOMY.items():
            run_safe = run.replace("_", "\\_")
            arch = t["architecture"].replace("_", "\\_")
            loss = t["loss"].replace("_", "\\_")
            src = (
                "internal" if t["source"].startswith("internal")
                else ("baseline" if "baseline" in t["source"] else "control")
                if "synthetic" in t["source"] else "internal"
            )
            f.write(f"{run_safe} & {arch} & {loss} & {src} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\arrayrulecolor{black}\n")
    print(f"  Wrote {tex}")
    return out


# ---------------------------------------------------------------------
# Degeneracy classifier
# ---------------------------------------------------------------------

DICE_BY_RUN = {
    "v2": 0.078, "v3": 0.077, "cgan_v1": 0.080,
    "cgan_v2": 0.043, "v2_6ch": 0.085, "cgan_v3": 0.091,
}


def classify_degeneracy() -> dict[str, dict]:
    """Multi-criterion degeneracy at the *generator* level.

    A run is degenerate if any of the following hold:
      (a) zero or one extracted edges on any city.
      (b) cross-city scale-invariant signature variance below 5% of
          truth's (catches density / mode collapse like cgan_v2).
      (c) Dice score below 0.05 (pixel-level foreground collapse;
          captures pre-Track-B v1).
    """
    mem_path = RESULTS / "memorisation_table_scale_invariant.csv"
    mem_var: dict[str, float] = {}
    truth_var = 0.0043
    if mem_path.exists():
        mem_df = pd.read_csv(mem_path)
        for _, r in mem_df.iterrows():
            mem_var[r["run"]] = float(r["cross_city_var"])

    out: dict[str, dict] = {}
    for run in TAXONOMY:
        info = {"degenerate_cells": [], "edge_counts": {},
                 "rel_to_truth": {}, "criteria_failed": []}
        any_low_edge = False
        for city in CITIES:
            ge = _gen_edge_count(run, city)
            te = _truth_edge_count(city)
            info["edge_counts"][city] = ge
            if ge < 0:
                info["rel_to_truth"][city] = float("nan")
                continue
            ratio = ge / max(te, 1)
            info["rel_to_truth"][city] = ratio
            if 0 <= ge <= 1:
                any_low_edge = True
                info["degenerate_cells"].append(city)
        if any_low_edge:
            info["criteria_failed"].append("(a) extraction failure / collapse")
        v = mem_var.get(run)
        if v is not None and v < 0.05 * truth_var:
            info["criteria_failed"].append("(b) signature mode-collapse")
        info["scale_inv_var"] = v
        # Criterion (c) only applies to runs that have a Dice score.
        d = DICE_BY_RUN.get(run)
        if d is not None and d < 0.05:
            info["criteria_failed"].append("(c) Dice<0.05 foreground collapse")
        info["any_degenerate"] = len(info["criteria_failed"]) > 0
        info["mostly_degenerate"] = (
            len(info["degenerate_cells"]) >= 3
            or "(b) signature mode-collapse" in info["criteria_failed"]
            or "(c) Dice<0.05 foreground collapse" in info["criteria_failed"]
        )
        info["all_degenerate"] = (len(info["degenerate_cells"])
                                    == len(CITIES))
        out[run] = info
    return out


# ---------------------------------------------------------------------
# W3: Non-degenerate Dice <-> CPR Spearman
# ---------------------------------------------------------------------


def w3_nondegenerate_spearman(panel: pd.DataFrame,
                                 degen: dict) -> dict:
    """Compute Dice<->CPR Spearman with and without degenerate
    generators across all 5 cities (panel mean) and per-payoff.
    Only DUPT generators have a Dice score; non-DUPT runs are
    excluded from the Spearman regardless.
    """
    out: dict[str, dict] = {}
    for payoff in PAYOFFS:
        col_i = f"cpr_{payoff}_indep"
        col_c = f"cpr_{payoff}_corr"
        agg = (panel.assign(cpr=panel[[col_i, col_c]].mean(axis=1))
                     .groupby("run", as_index=False)["cpr"].mean())
        agg["dice"] = agg["run"].map(DICE_BY_RUN)
        full = agg.dropna(subset=["dice"])
        # Non-degenerate: drop runs that are degenerate on >=3/5 cities.
        nondegen_runs = [r for r in full["run"]
                          if not degen.get(r, {}).get(
                              "mostly_degenerate", False)]
        nd = full[full["run"].isin(nondegen_runs)]
        out[payoff] = {
            "all_dupt": {
                "n": int(len(full)),
                "rho": _spearman(full["dice"], full["cpr"]),
                "runs": full["run"].tolist(),
            },
            "non_degenerate_only": {
                "n": int(len(nd)),
                "rho": _spearman(nd["dice"], nd["cpr"]),
                "runs": nd["run"].tolist(),
            },
        }
    return out


# ---------------------------------------------------------------------
# W7: Model-selection experiment (Dice vs W1 vs CPR)
# ---------------------------------------------------------------------

def w7_selector_experiment(panel: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-city-out: for each held-out city, pick the
    'best' generator according to each selector metric, then
    measure routing-CPR on the held-out city.

    Selectors:
      Dice           : argmax per-run Dice (constant across cities)
      Routing-CPR    : argmin mean routing-CPR on the 4 training cities
      Wire-CPR       : argmin mean wire-CPR on the 4 training cities
      Coverage-CPR   : argmin mean coverage-CPR on the 4 training cities
    """
    rows = []
    runs_dupt = list(DICE_BY_RUN.keys())
    routing_avg = (panel
        .assign(r_avg=panel[["cpr_routing_indep",
                                "cpr_routing_corr"]].mean(axis=1))
        .groupby(["run", "city"])["r_avg"].mean()
        .unstack("city"))
    wire_avg = (panel
        .assign(w_avg=panel[["cpr_wire_indep",
                                "cpr_wire_corr"]].mean(axis=1))
        .groupby(["run", "city"])["w_avg"].mean()
        .unstack("city"))
    coverage_avg = (panel
        .assign(c_avg=panel[["cpr_coverage_indep",
                                "cpr_coverage_corr"]].mean(axis=1))
        .groupby(["run", "city"])["c_avg"].mean()
        .unstack("city"))

    # Use only DUPT runs (Dice exists) so all selectors are
    # comparable on the same candidate set.
    routing_avg = routing_avg.loc[runs_dupt]
    wire_avg = wire_avg.loc[runs_dupt]
    coverage_avg = coverage_avg.loc[runs_dupt]

    for held_out in CITIES:
        train = [c for c in CITIES if c != held_out]
        # Selectors: pick run minimizing each criterion on train cities.
        sel_dice = max(runs_dupt, key=lambda r: DICE_BY_RUN[r])
        sel_route = routing_avg[train].mean(axis=1).idxmin()
        sel_wire = wire_avg[train].mean(axis=1).idxmin()
        sel_cov = coverage_avg[train].mean(axis=1).idxmin()
        # Deployed-CPR (routing) on held-out city for each selector.
        for sel_name, sel_run in [
            ("Dice", sel_dice),
            ("Routing-CPR (argmin train)", sel_route),
            ("Wire-CPR (argmin train)", sel_wire),
            ("Coverage-CPR (argmin train)", sel_cov),
        ]:
            deployed = float(routing_avg.loc[sel_run, held_out])
            rows.append({
                "held_out_city": held_out,
                "selector": sel_name,
                "selected_run": sel_run,
                "deployed_routing_cpr": deployed,
            })
    df = pd.DataFrame(rows)
    out = RESULTS / "selector_experiment.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    summary = (df.groupby("selector")["deployed_routing_cpr"]
                  .agg(["mean", "std", "min", "max"])
                  .reset_index())
    print()
    print("  Selector | mean deployed routing CPR | std | min | max")
    for _, r in summary.iterrows():
        print(f"  {r['selector']:>30s}  "
              f"{r['mean']:7.2f}  ±{r['std']:6.2f}  "
              f"[{r['min']:.2f}, {r['max']:.2f}]")
    return df


# ---------------------------------------------------------------------
# W10: Leave-one-city-out Spearman
# ---------------------------------------------------------------------

def w10_loco_spearman(panel: pd.DataFrame, payoff: str = "routing"
                        ) -> pd.DataFrame:
    rows = []
    col_i = f"cpr_{payoff}_indep"
    col_c = f"cpr_{payoff}_corr"
    for held_out in CITIES + ["(all 5)"]:
        if held_out == "(all 5)":
            sub = panel
        else:
            sub = panel[panel["city"] != held_out]
        agg = (sub.assign(cpr=sub[[col_i, col_c]].mean(axis=1))
                    .groupby("run", as_index=False)["cpr"].mean())
        agg["dice"] = agg["run"].map(DICE_BY_RUN)
        full = agg.dropna(subset=["dice"])
        rho = _spearman(full["dice"], full["cpr"])
        rows.append({
            "held_out_city": held_out,
            "n_dupt_runs": int(len(full)),
            "spearman_rho_dice_cpr": rho,
        })
    df = pd.DataFrame(rows)
    out = RESULTS / "loco_dice_cpr_rho.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    print()
    for _, r in df.iterrows():
        print(f"  held out = {r['held_out_city']:>10s}  "
              f"n={r['n_dupt_runs']}  "
              f"rho(Dice,{payoff} CPR) = "
              f"{r['spearman_rho_dice_cpr']:+.3f}")
    return df


# ---------------------------------------------------------------------
# W10: indep vs corr sampler comparison
# ---------------------------------------------------------------------

def w10_sampler_compare(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for payoff in PAYOFFS:
        col_i = f"cpr_{payoff}_indep"
        col_c = f"cpr_{payoff}_corr"
        sub = panel[[col_i, col_c]].dropna()
        rho = _spearman(sub[col_i], sub[col_c])
        diff = (sub[col_i] - sub[col_c]).mean()
        rows.append({
            "payoff": payoff,
            "n_cells": int(len(sub)),
            "spearman_indep_corr": rho,
            "mean_diff_indep_minus_corr": float(diff),
        })
    df = pd.DataFrame(rows)
    out = RESULTS / "sampler_compare.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return df


def main() -> int:
    panel = pd.read_csv(RESULTS / "cpr_panel.csv")
    print("=" * 60)
    print("W1: Model taxonomy")
    print("=" * 60)
    write_taxonomy()

    print("\n" + "=" * 60)
    print("Degeneracy classifier")
    print("=" * 60)
    degen = classify_degeneracy()
    for run, info in degen.items():
        if info["any_degenerate"]:
            crit = " + ".join(info["criteria_failed"])
            print(f"  {run:>20s}  DEGENERATE: {crit}")

    print("\n" + "=" * 60)
    print("W3: Non-degenerate Dice <-> CPR Spearman")
    print("=" * 60)
    w3_out = w3_nondegenerate_spearman(panel, degen)
    out_path = RESULTS / "cpr_dice_nondegenerate.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(w3_out, f, indent=2)
    print(f"  Wrote {out_path}")
    print()
    for payoff, d in w3_out.items():
        a, n = d["all_dupt"], d["non_degenerate_only"]
        print(f"  [{payoff:>10s}]  all DUPT (n={a['n']}): rho={a['rho']:+.3f}"
              f" | non-degenerate (n={n['n']}): rho={n['rho']:+.3f}")

    print("\n" + "=" * 60)
    print("W7: Model-selection experiment (Dice vs CPR-on-train)")
    print("=" * 60)
    w7_selector_experiment(panel)

    print("\n" + "=" * 60)
    print("W10: Leave-one-city-out Dice <-> routing-CPR Spearman")
    print("=" * 60)
    w10_loco_spearman(panel)

    print("\n" + "=" * 60)
    print("W10: indep vs corr sampler comparison")
    print("=" * 60)
    w10_sampler_compare(panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

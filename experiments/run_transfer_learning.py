"""Transfer learning across grid morphology classes.

Trains DiGress on three different
training corpora and evaluates each on the standard 5-city held-out
panel:

  T-Zurich    : Zurich only (single-city control)
  T-Planned   : 'planned' subset of the 21-city corpus
  T-Organic   : 'organic' subset of the 21-city corpus

The hand-curated planned/organic split :

  planned (regular grid):
      buenos_aires, los_angeles, paris, new_york, riyadh
  organic (informal sprawl):
      lagos, mumbai, jakarta, ho_chi_minh, accra

(Other training cities are 'mixed' and skipped from this experiment.)

For each variant, we monkey-patch HELDOUT in train_v2 to exclude
non-target training cities and run training. Sampling + panel
patching uses the existing infer.py + rerun_panel_digress.py.

Outputs:
    generators/graph_diffusion/checkpoints/final_T<variant>.pt
    results/transfer_learning_panel.csv
    results/transfer_learning_matrix.json
    figures/transfer_matrix.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DUPT_CACHE = REPO_ROOT / "DUPT" / "data" / "cache_hv_v2"
CHECKPOINT_DIR = ROOT / "generators" / "graph_diffusion" / "checkpoints"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"
GRAPHS_MC = RESULTS / "graphs_mc"

# 5-city held-out panel (always excluded from training).
HELDOUT_CITIES = {"zurich", "berlin", "chicago", "bangkok", "sao_paulo"}

# Hand-curated training-class split (subset of the 21-city corpus).
PLANNED = ["buenos_aires", "los_angeles", "paris", "new_york", "riyadh"]
ORGANIC = ["lagos", "mumbai", "jakarta", "ho_chi_minh", "accra"]
ZURICH_ONLY = ["zurich"]  # uses zurich for training only; eval on the
                          # other 4 held-out cities.

VARIANTS = {
    "T_Zurich": ZURICH_ONLY,
    "T_Planned": PLANNED,
    "T_Organic": ORGANIC,
}


def _train(variant: str, training_cities: list[str]) -> Path:
    """Train DiGress on a specific training corpus.

        The standard train_v2 loops over all.npz files NOT in HELDOUT.
        We use --cities to override that and pass the exact training set.
        Note: this means train_v2 must support --cities (which it doesn't
        yet) -- so we pass the cities through an environment variable
        that the trainer reads.

    """
    suffix = variant
    ckpt = CHECKPOINT_DIR / f"final_{suffix}.pt"
    if ckpt.exists():
        print(f"  [{variant}] checkpoint exists, skip train.", flush=True)
        return ckpt

    # Pass the training-city set via env variable. The trainer
    # reads CPR_TRAIN_CITIES and overrides its city list.
    env = os.environ.copy()
    env["CPR_TRAIN_CITIES"] = ",".join(training_cities)
    cmd = [
        sys.executable, "-u", "-X", "utf8",
        "-m", "generators.graph_diffusion.train_v2",
        "--epochs", "8000",
        "--edge-head", "bilinear",
        "--mst-loss-weight", "5.0",
        "--w1-coord-weight", "1.0",
        "--deg-weight", "0.5",
        "--ckpt-suffix", suffix,
    ]
    print(f"  [{variant}] training on {len(training_cities)} cities: "
          f"{training_cities}", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    if rc != 0:
        raise RuntimeError(f"train_v2 failed for {variant}: rc={rc}")
    return ckpt


def _sample(variant: str, ckpt: Path) -> Path:
    folder = f"_T_{variant.replace('T_', '')}"
    out = GRAPHS_MC / f"digress_v1{folder}"
    if out.exists() and any(out.glob("*.graphml")):
        print(f"  [{variant}] samples exist, skip infer.", flush=True)
        return out
    cmd = [
        sys.executable, "-u", "-X", "utf8",
        "-m", "generators.graph_diffusion.infer",
        "--ckpt", str(ckpt), "--calibrate-to-truth",
        "--out-suffix", folder, "--n-samples", "8",
    ]
    print(f"  [{variant}] sampling...", flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    if rc != 0:
        raise RuntimeError(f"infer failed for {variant}: rc={rc}")
    return out


def _evaluate(variant: str, samples_folder: Path) -> dict[str, dict[str, float]]:
    """Patch the panel digress_v1 row using this variant's samples,
    read back the per-(city, payoff) CPR.
    """
    sweep_panel = RESULTS / f"cpr_panel_transfer_{variant}.csv"
    panel_csv = RESULTS / "cpr_panel.csv"
    backup = RESULTS / "cpr_panel__pre_transfer_backup.csv"
    if not backup.exists():
        shutil.copy(panel_csv, backup)
    shutil.copy(panel_csv, sweep_panel)

    target = GRAPHS_MC / "digress_v1"
    target_backup = GRAPHS_MC / "digress_v1__pre_transfer_backup"
    moved = False
    if target.exists() and not target_backup.exists():
        target.rename(target_backup)
        moved = True
    samples_folder.rename(target)
    try:
        cmd = [sys.executable, "-X", "utf8", "-m",
               "experiments.rerun_panel_digress"]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            raise RuntimeError(f"rerun_panel failed for {variant}: rc={rc}")
        patched = pd.read_csv(panel_csv)
        # Save the variant-specific panel for the matrix build.
        patched.to_csv(sweep_panel, index=False)
        # Restore baseline.
        shutil.copy(backup, panel_csv)
    finally:
        target.rename(samples_folder)
        if moved and target_backup.exists():
            target_backup.rename(target)

    drow = patched[patched["run"] == "digress_v1"]
    PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
    out: dict[str, dict[str, float]] = {p: {} for p in PAYOFFS}
    for _, r in drow.iterrows():
        city = r["city"]
        for p in PAYOFFS:
            ci = float(r.get(f"cpr_{p}_indep", float("nan")))
            cc = float(r.get(f"cpr_{p}_corr", float("nan")))
            v = ci if np.isnan(cc) else (ci + cc) / 2.0
            out[p][city] = v
    return out


def _city_class(city: str) -> str:
    """Classify a held-out city as planned/organic/mixed."""
    if city in {"chicago"}:
        return "planned"
    if city in {"bangkok", "sao_paulo"}:
        return "organic"
    return "mixed"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="*",
                   default=list(VARIANTS.keys()))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-sample", action="store_true")
    args = p.parse_args(argv)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for v in args.variants:
        cities = VARIANTS[v]
        try:
            if not args.skip_train:
                ckpt = _train(v, cities)
            else:
                ckpt = CHECKPOINT_DIR / f"final_{v}.pt"
            if not args.skip_sample:
                folder = _sample(v, ckpt)
            else:
                folder = GRAPHS_MC / f"digress_v1_T_{v.replace('T_', '')}"
            results[v] = _evaluate(v, folder)
        except Exception as exc:
            print(f"  [{v}] FAILED: {exc}", flush=True)
            continue

    # Per-(variant, city, payoff) tidy CSV.
    PAYOFFS = ["wire", "routing", "coverage", "dc_flow", "n_minus_1"]
    rows = []
    for v, data in results.items():
        for payoff, by_city in data.items():
            for city, cpr in by_city.items():
                rows.append({
                    "variant": v, "city": city,
                    "city_class": _city_class(city),
                    "payoff": payoff, "cpr": cpr,
                })
    df = pd.DataFrame(rows)
    out = RESULTS / "transfer_learning_panel.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # train-class × test-class CPR matrix (averaged over cities + payoffs).
    matrix = {}
    for v in results:
        train_class = ("planned" if v == "T_Planned"
                       else "organic" if v == "T_Organic"
                       else "single-city Zurich")
        for city in df[df["variant"] == v]["city"].unique():
            tc = _city_class(city)
            sub = df[(df["variant"] == v) & (df["city"] == city)
                       & df["cpr"].notna()]
            if sub.empty:
                continue
            mean_cpr = float(sub["cpr"].mean())
            matrix.setdefault(train_class, {}).setdefault(tc, []).append(
                mean_cpr)

    summary = {tr: {tc: float(np.mean(vals)) for tc, vals in by_test.items()}
                for tr, by_test in matrix.items()}
    out_json = RESULTS / "transfer_learning_matrix.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    print("\nTransfer matrix (train-class -> mean CPR by test-class):")
    classes = ["planned", "mixed", "organic"]
    for tr in summary:
        for tc in classes:
            if tc in summary[tr]:
                print(f"  {tr:>22s} -> {tc:>10s}: "
                      f"mean CPR = {summary[tr][tc]:7.2f}")

    # Heatmap figure.
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    train_classes = list(summary.keys())
    test_classes = classes
    mat = np.full((len(train_classes), len(test_classes)), np.nan)
    for i, tr in enumerate(train_classes):
        for j, tc in enumerate(test_classes):
            mat[i, j] = summary[tr].get(tc, np.nan)
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(test_classes)))
    ax.set_xticklabels(test_classes)
    ax.set_yticks(range(len(train_classes)))
    ax.set_yticklabels(train_classes)
    ax.set_xlabel("Test city class")
    ax.set_ylabel("Training corpus class")
    for i in range(len(train_classes)):
        for j in range(len(test_classes)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if abs(v - np.nanmin(mat))
                            > (np.nanmax(mat) - np.nanmin(mat)) * 0.5
                        else "black", fontsize=10)
    ax.set_title("Transfer-learning CPR matrix (mean across all 5 payoffs)")
    plt.colorbar(im, ax=ax, label="Mean CPR")
    out_pdf = FIGS / "transfer_matrix.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

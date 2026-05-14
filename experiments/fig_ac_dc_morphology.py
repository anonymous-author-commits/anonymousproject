"""Render per-city Spearman ρ(DC-CPR, AC-CPR) stratified by
morphology class. Surfaces the morphology-dependent
DC↔AC concordance finding that's invisible at n=5.

Output: figures/ac_dc_morphology.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

# Hand-curated morphology classes (documented in Methods).
PLANNED = {"berlin", "chicago", "madrid", "warsaw", "toronto",
            "amsterdam", "melbourne", "san_francisco"}
MIXED = {"zurich", "beijing", "rome"}
INFORMAL = {"bangkok", "sao_paulo", "cairo", "rio_de_janeiro"}

CLASS_COLOR = {
    "planned": "#1f77b4",
    "mixed":   "#7f7f7f",
    "informal": "#d62728",
}


def _classify(c: str) -> str:
    if c in PLANNED: return "planned"
    if c in MIXED: return "mixed"
    if c in INFORMAL: return "informal"
    return "unknown"


def main() -> int:
    with (RESULTS / "dc_vs_ac_spearman.json").open() as f:
        d = json.load(f)
    rho = d["rho_per_city"]
    rho_agg = float(d["rho_aggregate"])

    # Sort cities within each class by ρ (ascending).
    rows = sorted(
        [(c, float(v), _classify(c)) for c, v in rho.items()],
        key=lambda r: (
            {"planned": 0, "mixed": 1, "informal": 2}.get(r[2], 3),
            r[1],
        ),
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    xs = np.arange(len(rows))
    colors = [CLASS_COLOR[r[2]] for r in rows]
    bars = ax.bar(xs, [r[1] for r in rows], color=colors,
                   edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(rho_agg, color="#222", lw=1.0, ls="--",
               label=fr"aggregate $\rho={rho_agg:+.2f}$")
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [r[0].replace("_", " ") for r in rows],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel(r"per-city Spearman $\rho$(DC-CPR, AC-CPR)")
    ax.set_ylim(-0.7, 0.7)
    ax.grid(axis="y", alpha=0.3)

    # Legend.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=CLASS_COLOR["planned"], edgecolor="black",
              label="Planned grid"),
        Patch(facecolor=CLASS_COLOR["mixed"], edgecolor="black",
              label="Mixed"),
        Patch(facecolor=CLASS_COLOR["informal"], edgecolor="black",
              label="Informal sprawl"),
    ]
    handles.append(plt.Line2D([0], [0], color="#222", lw=1.0,
                                ls="--",
                                label=f"aggregate $\\rho={rho_agg:+.2f}$"))
    ax.legend(handles=handles, loc="upper left", fontsize=9)

    # Class-mean annotations.
    for cls in ["planned", "mixed", "informal"]:
        vals = [r[1] for r in rows if r[2] == cls]
        if vals:
            mean = float(np.mean(vals))
            xs_cls = [i for i, r in enumerate(rows) if r[2] == cls]
            x_mid = float(np.mean(xs_cls))
            ax.annotate(
                f"{cls}: mean ρ = {mean:+.2f}  (n={len(vals)})",
                xy=(x_mid, 0.55), ha="center",
                color=CLASS_COLOR[cls], fontsize=8, fontweight="bold",
            )
    ax.set_title(
        "DC$\\leftrightarrow$AC ranking concordance is "
        "morphology-dependent",
        fontsize=11,
    )

    out = FIGS / "ac_dc_morphology.pdf"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Also print the class means for the manuscript text.
    print()
    print("Per-class mean rho(DC, AC):")
    for cls in ["planned", "mixed", "informal"]:
        vals = [r[1] for r in rows if r[2] == cls]
        if vals:
            print(f"  {cls:>10s}  mean = {np.mean(vals):+.3f}  "
                  f"min = {min(vals):+.3f}  max = {max(vals):+.3f}  "
                  f"n = {len(vals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""A driver: produce Birchfield-2017 synthetic-grid samples for each
of the 15 held-out cities and emit a summary table for the paper.

Outputs
-------
    results/graphs_mc/birchfield_2017/<city>_NN.graphml   (per sample)
    results/birchfield_summary.csv                          (one row per city)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import networkx as nx
import numpy as np

from external_baselines.birchfield_2017 import write_panel_graphs

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT.parent / "DUPT" / "data" / "cache_hv_v2"
TRUTH_DIR = ROOT / "results" / "graphs" / "_truth"
OUT_DIR = ROOT / "results" / "graphs_mc" / "birchfield_2017"
SUMMARY_CSV = ROOT / "results" / "birchfield_summary.csv"
SUMMARY_JSON = ROOT / "results" / "birchfield_summary.json"

CITIES = [
    "zurich", "berlin", "chicago", "bangkok", "sao_paulo",
    "cairo", "madrid", "warsaw", "toronto", "amsterdam",
    "beijing", "melbourne", "rio_de_janeiro", "rome", "san_francisco",
]


def main() -> int:
    print(f"Running Birchfield 2017 on {len(CITIES)} cities ...")
    diagnostics = write_panel_graphs(
        CITIES,
        cache_dir=CACHE_DIR,
        truth_dir=TRUTH_DIR,
        out_dir=OUT_DIR,
        seed=0,
        samples_per_city=1,
    )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "city", "n_truth", "n_birchfield", "e_birchfield",
            "components", "sample_idx",
        ])
        for city, d in diagnostics.items():
            for s in d["samples"]:
                writer.writerow([
                    city, d["n_truth"], s["nodes"], s["edges"],
                    s["components"], s["sample_idx"],
                ])
    SUMMARY_JSON.write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {SUMMARY_CSV}  ({len(diagnostics)} cities)")
    print(f"Wrote {SUMMARY_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Climate Planning Regret (CPR)

Decision-theoretic evaluation framework for generative power-grid models.
Companion code for the manuscript *Decision-theoretic evaluation reveals
planning blind spots in generative power-grid models* (under review).

## What this repo contains

```
cpr/                        Core package: payoff functions, W1 estimator,
                             Lipschitz bound, monetary translation, regret loop.
  _powergrid/               Vendored subset of the DC power-flow solver used
                             by the DC-flow / N-1 payoffs (self-contained).
external_baselines/
  birchfield_2017/          Reproduction of Birchfield et al. (2017, IEEE TPS).
generators/
  graph_diffusion/          Adapted DiGress checkpoint scaffold + training/
                             inference scripts. Checkpoints are not bundled.
experiments/                Driver scripts that reproduce every numerical
                             claim in the manuscript.
paper/                      LaTeX sources for the manuscript.
data/                       Placeholder; see data/README.md to fetch the
                             raster cache.
results/, figures/          Gitignored output dirs populated by the drivers.
```

## Installation

Requires Python 3.10+. Tested on Linux, macOS, and Windows 11.

```bash
pip install -e .
# Optional extras:
pip install -e ".[ac-opf]"      # pandapower for the AC-OPF cross-validation
pip install -e ".[diffusion]"    # PyTorch for DiGress training/inference
pip install -e ".[inference]"   # OSMnx + rasterio for zero-shot inference
```

## Reproducing the manuscript figures and tables

Each manuscript claim has a corresponding driver script in `experiments/`.

### Core panel results

```bash
python -m experiments.run_cpr_panel              # main 12-generator x 15-city CPR sweep
python -m experiments.bootstrap_cross_payoff     # Table 1 + 95% CIs
python -m experiments.lipschitz_estimate         # Lipschitz constants per payoff
python -m experiments.bound_per_payoff           # Fig. 4 bound informativity
python -m experiments.monetary_translation       # capex translation tables
python -m experiments.capex_sensitivity          # Table 3 capex sensitivity grid
```

### Sub-experiments

```bash
python -m experiments.run_cpr_ac                 # AC-OPF cross-validation (Fig. 8)
python -m experiments.fig_ac_dc_morphology       # Fig. 8 morphology stratification
python -m experiments.bound_stress_test          # Fig. 7 synthetic bound stress
python -m experiments.case_study_sao_paulo       # São Paulo planning case study
python -m experiments.road_corridor_cpr          # road-corridor payoff
python -m experiments.run_transfer_learning      # Fig. 9 transfer-learning matrix
python -m experiments.hierarchical_w1_demo       # Supplementary Fig. S4
python -m experiments.visual_galleries           # Fig. 5 & 6 visual galleries
python -m experiments.per_city_breakout          # per-city decomposition table
python -m experiments.per_city_spearman          # per-city Spearman matrix
python -m experiments.robustness_battery         # selector experiment + degeneracy
```

### External baseline

```bash
python -m experiments.run_birchfield             # generate Birchfield 2017 graphs
python -m experiments.run_birchfield_cpr         # evaluate Birchfield through CPR panel
```

## Data prerequisites

The driver scripts expect:

- `results/graphs/_truth/<city>.graphml` for each of the 15 panel cities,
  built from OpenStreetMap high-voltage relations.
- `data/cache_hv_v2/<city>.npz` containing the six-channel conditioning
  cube (built-up, water, OSM substations, elevation, roads, rails) at
  10 m/pixel, 1536x1536.
- For the DiGress generator: a trained checkpoint at
  `generators/graph_diffusion/checkpoints/final_v2.pt`.

See `data/README.md` for the raster source URLs and the extraction
recipe.

## Layout notes

- `cpr/_powergrid/` is a vendored snapshot of the upstream DC power-flow
  solver. It is here so the panel-CPR loop is self-contained; the
  upstream package can be re-pulled later if extensions are needed
  (contingency analysis, reliability metrics, cross-city tooling).
- `generators/graph_diffusion/checkpoints/` is empty in this release;
  trained checkpoints are large (~250 MB) and are distributed separately
  (see `paper/main.tex` §"Code and data availability" for the Zenodo
  reference).
- `_workspace/` (local-only, gitignored) holds the assembly script that
  built this release from the working copy. Not required to use the
  package.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{cpr2026,
  title = {Decision-theoretic evaluation reveals planning blind spots
           in generative power-grid models},
  author = {[Author names withheld for review]},
  year = {2026},
  note = {Under review.}
}
```

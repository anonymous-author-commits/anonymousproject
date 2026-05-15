# Data

The driver scripts expect three datasets that are not bundled in this repo
because of licensing and size:

## 1. OpenStreetMap high-voltage relations (truth graphs)

For each of the 15 panel cities, a `_truth/<city>.graphml` file is built
from the OSM `power=line`/`power=substation` relations within the city
bounding box. Reproduce with:

```bash
python -m experiments.build_truth_graphs --cities zurich berlin chicago ...
```

The script uses `osmnx>=2.0` (install via `pip install -e ".[inference]"`)
and writes to `results/graphs/_truth/`.

## 2. Conditioning cubes

The 6-channel conditioning rasters at 10 m/pixel, 1536x1536, contain:

  channel 0  ESA WorldCover 2021 built-up classification, [-1, +1]
  channel 1  ESA WorldCover surface-water mask
  channel 2  OSM substation locations rasterised at 10 m
  channel 3  Copernicus DEM 30 m elevation, normalised
  channel 4  OSM road-network mask
  channel 5  OSM rail-network mask

Public source URLs:

  - ESA WorldCover 2021:  https://esa-worldcover.org/en/data-access
  - Copernicus DEM (GLO-30): https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model
  - OpenStreetMap (Overpass): https://overpass-api.de/

The cubes are written to `data/cache_hv_v2/<city>.npz` with two arrays:
`cond` (shape `(6, 1536, 1536)`, float32) and `hv` (shape `(1536, 1536)`,
the OSM high-voltage rasterisation used as the supervision target).

## 3. Trained generator checkpoints

Trained DiGress and DUPT-family checkpoints are not bundled in this repo
(`~250 MB` per file). They will be released via a separate, citable
archive at de-anonymisation; reviewers can request a private link via
the editor. The expected layout once installed:

```
generators/graph_diffusion/checkpoints/
  final_v2.pt                 # primary DiGress checkpoint (~250 MB)
  final_v2_noraster.pt        # no-raster ablation
  final_T_Zurich.pt           # T-Zurich transfer-learning checkpoint
  final_T_Planned.pt          # T-Planned transfer-learning checkpoint
  final_T_Organic.pt          # T-Organic transfer-learning checkpoint
  final_sweep_A1..D2.pt       # hyperparameter sweep variants
```

DUPT generator outputs (the six pixel-native variants) are distributed
as pre-computed multigraph samples under
`results/graphs_mc/multi_city_hv_*`; re-running the DUPT inference loop
requires the original DUPT training code, which is not bundled here.

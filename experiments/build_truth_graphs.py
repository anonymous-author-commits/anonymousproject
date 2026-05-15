"""Build the per-city OSM high-voltage truth graphs used by the CPR panel.

For each city in --cities, queries OpenStreetMap (via osmnx 2.x) for power
infrastructure within the city bounding box, extracts the HV line segments
and substation nodes, and writes a GraphML file with px_r / px_c integer
pixel coordinates in the city's 1536x1536 conditioning grid.

The resulting files are the truth side of every CPR-evaluator call:

    results/graphs/_truth/<city>.graphml

City bounding boxes are loaded from city_bboxes.csv (lat/lon corners +
UTM EPSG); the same boxes drive the conditioning-cube extraction. If the
csv is missing the script falls back to the small built-in dictionary
below covering the 15 panel cities.

Usage
-----
    python -m experiments.build_truth_graphs --cities zurich berlin chicago

Requires the inference extras: ``pip install -e ".[inference]"``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np

try:
    import osmnx as ox
    _HAS_OSMNX = True
except ImportError:
    _HAS_OSMNX = False

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "graphs" / "_truth"
TILE_PX = 1536
PIXEL_M = 10.0

# Fallback bounding boxes (south, west, north, east) in degrees plus
# the UTM EPSG used to project metres to pixels. Centred so the city's
# dense core sits near the centre of the 1536x1536 tile.
CITY_BBOXES: dict[str, tuple[float, float, float, float, int]] = {
    "zurich":          (47.331,  8.470, 47.469,  8.620, 32632),
    "berlin":          (52.450, 13.300, 52.580, 13.490, 32633),
    "chicago":         (41.832, -87.745, 41.952, -87.580, 32616),
    "bangkok":         (13.694, 100.490, 13.825, 100.620, 32647),
    "sao_paulo":       (-23.620, -46.690, -23.510, -46.560, 32723),
    "cairo":           (30.020, 31.190, 30.120, 31.310, 32636),
    "madrid":          (40.380, -3.770, 40.490, -3.620, 32630),
    "warsaw":          (52.180, 20.940, 52.290, 21.090, 32634),
    "toronto":         (43.630, -79.470, 43.730, -79.330, 32617),
    "amsterdam":       (52.330,  4.840, 52.420,  4.970, 32631),
    "beijing":         (39.860, 116.300, 39.980, 116.490, 32650),
    "melbourne":       (-37.860, 144.880, -37.760, 145.040, 32755),
    "rio_de_janeiro":  (-22.960, -43.260, -22.860, -43.130, 32723),
    "rome":            (41.860, 12.420, 41.960, 12.560, 32633),
    "san_francisco":   (37.730, -122.510, 37.820, -122.370, 32610),
}


def _project_to_pixels(
    lat: float, lon: float, bbox: tuple[float, float, float, float, int],
) -> tuple[int, int]:
    """Convert (lat, lon) into (px_r, px_c) for a 1536x1536 tile."""
    s, w, n, e, _epsg = bbox
    # Linear interpolation within the bbox; row 0 is north.
    frac_lon = (lon - w) / (e - w)
    frac_lat = (n - lat) / (n - s)
    px_c = int(round(frac_lon * (TILE_PX - 1)))
    px_r = int(round(frac_lat * (TILE_PX - 1)))
    px_c = max(0, min(TILE_PX - 1, px_c))
    px_r = max(0, min(TILE_PX - 1, px_r))
    return px_r, px_c


def build_truth_graph(
    city: str, bbox: tuple[float, float, float, float, int],
) -> nx.MultiGraph:
    """Fetch HV power infrastructure for city and return a pixel multigraph."""
    if not _HAS_OSMNX:
        raise RuntimeError(
            "osmnx not available; install with `pip install -e '.[inference]'`"
        )
    s, w, n, e, _ = bbox
    tags = {"power": ["line", "minor_line", "substation"]}
    gdf = ox.features_from_bbox(bbox=(w, s, e, n), tags=tags)
    g = nx.MultiGraph()
    node_idx = 0

    def _add_node(lat: float, lon: float) -> int:
        nonlocal node_idx
        pr, pc = _project_to_pixels(lat, lon, bbox)
        g.add_node(node_idx, px_r=pr, px_c=pc)
        node_idx += 1
        return node_idx - 1

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        gtype = geom.geom_type
        if gtype == "Point":
            _add_node(geom.y, geom.x)
        elif gtype in ("LineString", "MultiLineString"):
            segments = (geom.geoms if gtype == "MultiLineString" else [geom])
            for seg in segments:
                coords = list(seg.coords)
                if len(coords) < 2:
                    continue
                prev = _add_node(coords[0][1], coords[0][0])
                for lon, lat in coords[1:]:
                    cur = _add_node(lat, lon)
                    dpx = (
                        (g.nodes[prev]["px_r"] - g.nodes[cur]["px_r"]) ** 2
                        + (g.nodes[prev]["px_c"] - g.nodes[cur]["px_c"]) ** 2
                    ) ** 0.5
                    g.add_edge(prev, cur, length_px=float(dpx))
                    prev = cur
        elif gtype == "Polygon":
            cent = geom.centroid
            _add_node(cent.y, cent.x)
    return g


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build OSM HV truth graphs for the CPR panel."
    )
    parser.add_argument(
        "--cities", nargs="*", default=list(CITY_BBOXES.keys()),
        help="Cities to fetch (default: all 15 panel cities).",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR,
        help=f"Output directory (default: {OUT_DIR.relative_to(ROOT)}).",
    )
    args = parser.parse_args()

    if not _HAS_OSMNX:
        print("ERROR: osmnx is required. Install with:")
        print("       pip install -e '.[inference]'")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for city in args.cities:
        if city not in CITY_BBOXES:
            print(f"  [skip] {city}: no bbox configured")
            continue
        print(f"  fetching {city} ...", flush=True)
        g = build_truth_graph(city, CITY_BBOXES[city])
        out_path = args.out / f"{city}.graphml"
        nx.write_graphml(g, out_path)
        print(
            f"    wrote {out_path.relative_to(ROOT)}  "
            f"|V|={g.number_of_nodes()}  |E|={g.number_of_edges()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

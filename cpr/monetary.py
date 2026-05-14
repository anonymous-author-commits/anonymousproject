"""Monetary translation of CPR routing distances into excess capex.

Calibration sources
-------------------
* IEA, *Electricity Grids and Secure Energy Transitions* (2023).
  Table B.2 -- "Indicative grid investment cost ranges" --
  reports HV transmission at $1.5--3.0M/km globally with a midpoint
  of $2.5M/km used for OECD planning studies. Underground HV is at
  the upper end of this range; greenfield overhead at the lower
  end.
* ENTSO-E TYNDP 2022, *Cost Reference Tables for Transmission
  Investment*. MV (33--110 kV) overhead lines at EUR 0.4--0.8M/km;
  underground MV cables at EUR 0.8--1.5M/km. Converted to USD at
  EUR 1 = USD 1.07 (long-run average 2022--2024).
* ENTSO-E TYNDP 2022 + Eurelectric 2021 distribution-cost report
  for LV trenched cables at EUR 0.05--0.15M/km, urban-greenfield.

The midpoints we adopt (overhead, greenfield, mid-density urban):
HV $2.5M/km, MV $0.6M/km, LV $0.10M/km. Per-city deltas relative
to these midpoints are deferred to supplementary if reviewers ask;
we report uncertainty bands rather than per-city point estimates.

Pixel-to-metres calibration: the CPR-routing payoff is the
normalised (per-node-mean) wire cost in pixels. A pixel is 10 m
on the underlying conditioning grid. A routing-CPR value of x
pixels-per-node thus represents an excess of (x * 0.01) km of
infrastructure per substation siting decision, when the planner
acted on the model's belief rather than the truth.
"""
from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------
#  Calibrated $/km tables (USD millions per kilometre, 2024 dollars)
# ---------------------------------------------------------------------

# Voltage-class cost ranges. Each tuple is (low, midpoint, high) in
# millions of USD per kilometre of new construction. Sources cited
# in the module docstring.
COST_PER_KM_USD_M: dict[str, tuple[float, float, float]] = {
    "HV": (1.5, 2.5, 3.0),     # IEA 2023 Table B.2
    "MV": (0.43, 0.64, 1.61),  # ENTSO-E TYNDP 2022 (EUR converted)
    "LV": (0.05, 0.10, 0.15),  # Eurelectric 2021
}

# Pixel size on the DUPT conditioning grid, in metres.
PIXEL_SIZE_M: float = 10.0

# Number of substation-siting decisions per city per planning round.
# Default K = 4 mirrors the panel default in run_cpr_panel.py.
DEFAULT_K: int = 4


# ---------------------------------------------------------------------
#  Terrain cost multipliers (CIGRE TB 723)
# ---------------------------------------------------------------------
#
# CIGRE TB 723 (Engineering Guidelines on EHV/HV Transmission Line
# Economics, 2018) reports terrain-difficulty multipliers on the base
# $/km cost depending on the slope, soil bearing, and access of the
# right-of-way. The widely-used utility convention is:
#
#   Flat        (mean slope <  5%):  1.0
#   Rolling     (mean slope  5-15%): 1.2
#   Hilly       (mean slope 15-25%): 1.5
#   Mountainous (mean slope > 25%):  2.0 (up to 2.5 with rock blasting)
#
# We assign each of the 15 panel cities to a terrain class based on
# publicly documented mean urban-footprint slope (Copernicus DEM /
# SRTM, urban-area mask from ESA WorldCover). The assignments err on
# the conservative side: cities like Rio with mountainous interior but
# coastal flats are classified by the relevant transmission corridor
# (Tijuca / Serra do Mar route, hence "mountainous"). The mapping is
# documented in Methods (`Local-geography capex sensitivity`).
TERRAIN_CLASS: dict[str, str] = {
    "zurich":          "hilly",
    "berlin":          "flat",
    "chicago":         "flat",
    "bangkok":         "flat",
    "sao_paulo":       "hilly",
    "cairo":           "flat",
    "madrid":          "rolling",
    "warsaw":          "flat",
    "toronto":         "flat",
    "amsterdam":       "flat",
    "beijing":         "rolling",
    "melbourne":       "rolling",
    "rio_de_janeiro":  "mountainous",
    "rome":            "rolling",
    "san_francisco":   "hilly",
}

# CIGRE TB 723 multipliers. Each tuple is (low, midpoint, high) so the
# terrain-uncertainty range can be propagated through the dollar
# calculation alongside the per-km cost range. "Mountainous" covers
# 2.0-2.5x; "flat" is essentially fixed at 1.0 with negligible spread.
TERRAIN_MULTIPLIER: dict[str, tuple[float, float, float]] = {
    "flat":         (1.0, 1.0, 1.05),
    "rolling":      (1.10, 1.20, 1.30),
    "hilly":        (1.35, 1.50, 1.70),
    "mountainous":  (1.90, 2.20, 2.50),
}


def terrain_multiplier(
    city: str,
    band: str = "midpoint",
) -> float:
    """Return the CIGRE TB 723 terrain cost multiplier for a city.

    Parameters
    ----------
    city
        Lowercase underscore-separated city name (e.g. ``"sao_paulo"``).
        Must be in ``TERRAIN_CLASS``.
    band
        ``"low"``, ``"midpoint"`` (default), or ``"high"`` band of the
        multiplier. Useful for propagating terrain uncertainty into
        the dollar headline.
    """
    if city not in TERRAIN_CLASS:
        raise ValueError(
            f"No terrain class assigned for {city!r}; "
            f"see TERRAIN_CLASS in cpr.monetary."
        )
    cls = TERRAIN_CLASS[city]
    lo, mid, hi = TERRAIN_MULTIPLIER[cls]
    return {"low": lo, "midpoint": mid, "high": hi}[band]


@dataclass(frozen=True)
class CapexBand:
    """An excess-capex range, in millions of USD."""
    low: float
    midpoint: float
    high: float
    voltage_class: str
    n_actions: int
    excess_km_per_action: float

    def __str__(self) -> str:
        return (f"${self.low:.1f}-{self.high:.1f}M "
                f"(midpoint ${self.midpoint:.1f}M; "
                f"{self.excess_km_per_action:.2f} km/action over "
                f"{self.n_actions} actions, {self.voltage_class})")


def cpr_to_capex(
    cpr_routing_px: float,
    *,
    voltage_class: str = "HV",
    pixel_size_m: float = PIXEL_SIZE_M,
    n_actions: int = DEFAULT_K,
    city: str | None = None,
    apply_terrain: bool = False,
) -> CapexBand:
    """Convert a routing-CPR value (pixels) to an excess-capex band.

    Parameters
    ----------
    cpr_routing_px
        Per-action routing CPR in pixels. Numerically this is
        ``cpr_panel.csv:cpr_routing_*`` for a given (run, city) cell.
        Routing CPR is normalised per-node, so a value of x pixels
        means each graph node is on average x pixels further from
        the model-chosen substation site than from the oracle site.
    voltage_class
        One of ``"HV"``, ``"MV"``, ``"LV"``. Defaults to HV (the
        relevant class for the substation-siting decision).
    pixel_size_m
        Pixel size in metres on the conditioning grid (10 by default).
    n_actions
        Number of substation-siting decisions per planning round
        (default K = 4).
    city
        Optional city name (lowercase, underscore-separated) for
        terrain-multiplier lookup. Required if ``apply_terrain=True``.
    apply_terrain
        If True, multiply the $/km cost band by the CIGRE TB 723
        terrain class for the city (flat / rolling / hilly /
        mountainous). The (low, mid, high) of the cost band is
        multiplied by the (low, mid, high) of the terrain band
        pairwise so that the returned CapexBand reflects joint
        $/km and terrain uncertainty.

    Returns
    -------
    CapexBand
        Excess capex (low, midpoint, high) in millions of USD,
        plus the per-action excess kilometres.
    """
    if voltage_class not in COST_PER_KM_USD_M:
        raise ValueError(
            f"Unknown voltage class {voltage_class!r}; "
            f"expected one of {sorted(COST_PER_KM_USD_M)}."
        )
    lo, mid, hi = COST_PER_KM_USD_M[voltage_class]

    if apply_terrain:
        if city is None:
            raise ValueError("apply_terrain=True requires city argument.")
        t_lo = terrain_multiplier(city, "low")
        t_mid = terrain_multiplier(city, "midpoint")
        t_hi = terrain_multiplier(city, "high")
        lo = lo * t_lo
        mid = mid * t_mid
        hi = hi * t_hi

    excess_m_per_action = cpr_routing_px * pixel_size_m
    excess_km_per_action = excess_m_per_action / 1000.0
    total_excess_km = excess_km_per_action * n_actions

    return CapexBand(
        low=total_excess_km * lo,
        midpoint=total_excess_km * mid,
        high=total_excess_km * hi,
        voltage_class=voltage_class,
        n_actions=n_actions,
        excess_km_per_action=excess_km_per_action,
    )


def best_vs_worst_capex(
    cpr_best: float,
    cpr_worst: float,
    *,
    voltage_class: str = "HV",
    pixel_size_m: float = PIXEL_SIZE_M,
    n_actions: int = DEFAULT_K,
    city: str | None = None,
    apply_terrain: bool = False,
) -> CapexBand:
    """Excess capex of using the panel-worst non-degenerate generator
    instead of the panel-best, in dollars.

    Parameters
    ----------
    cpr_best, cpr_worst
        Routing CPR (in pixels per action) for the best and worst
        non-degenerate generators on the city in question.
    city, apply_terrain
        Forwarded to ``cpr_to_capex``; when ``apply_terrain=True``
        the returned band incorporates CIGRE TB 723 terrain
        multipliers for the city.
    """
    delta = max(cpr_worst - cpr_best, 0.0)
    return cpr_to_capex(
        delta,
        voltage_class=voltage_class,
        pixel_size_m=pixel_size_m,
        n_actions=n_actions,
        city=city,
        apply_terrain=apply_terrain,
    )


__all__ = [
    "COST_PER_KM_USD_M",
    "PIXEL_SIZE_M",
    "DEFAULT_K",
    "TERRAIN_CLASS",
    "TERRAIN_MULTIPLIER",
    "CapexBand",
    "cpr_to_capex",
    "best_vs_worst_capex",
    "terrain_multiplier",
]

"""Load profiles for power grid demand modeling.

Generates spatially and temporally resolved load profiles for each node
in the power graph, parameterized by electrification level, urban density,
and node type (residential, commercial, industrial proxy).

Key concept: demand is *endogenous* — it depends on grid capacity and
electrification planning decisions, not just population.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..approaches.common import NodeType, PowerGraph, VoltageClass

logger = logging.getLogger(__name__)


class ConsumerType(str, Enum):
    """Demand consumer archetypes."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED = "mixed"


# Hourly load shape factors (24h, normalized to peak=1.0)
# Source: typical European/North American profiles (IEA, ENTSO-E)
_LOAD_SHAPES = {
    ConsumerType.RESIDENTIAL: np.array([
        0.35, 0.30, 0.28, 0.27, 0.28, 0.35, 0.50, 0.70,
        0.75, 0.65, 0.55, 0.55, 0.60, 0.55, 0.50, 0.55,
        0.65, 0.80, 0.95, 1.00, 0.90, 0.75, 0.55, 0.40,
    ]),
    ConsumerType.COMMERCIAL: np.array([
        0.20, 0.18, 0.16, 0.15, 0.16, 0.20, 0.35, 0.60,
        0.85, 0.95, 1.00, 0.98, 0.90, 0.95, 0.98, 0.95,
        0.90, 0.80, 0.60, 0.40, 0.30, 0.25, 0.22, 0.20,
    ]),
    ConsumerType.INDUSTRIAL: np.array([
        0.70, 0.70, 0.70, 0.70, 0.72, 0.80, 0.90, 0.95,
        1.00, 1.00, 1.00, 0.98, 0.90, 0.95, 1.00, 1.00,
        0.98, 0.95, 0.85, 0.78, 0.75, 0.72, 0.70, 0.70,
    ]),
    ConsumerType.MIXED: np.array([
        0.40, 0.35, 0.32, 0.30, 0.32, 0.40, 0.55, 0.72,
        0.85, 0.88, 0.85, 0.82, 0.78, 0.80, 0.82, 0.80,
        0.82, 0.85, 0.80, 0.72, 0.60, 0.50, 0.45, 0.42,
    ]),
}

# Base annual energy consumption per consumer type (MWh/year per load node)
# These represent aggregate zones, not individual consumers
_BASE_ANNUAL_MWH = {
    ConsumerType.RESIDENTIAL: 5.0,    # ~5 MWh/yr per zone (moderate density)
    ConsumerType.COMMERCIAL: 15.0,    # higher intensity
    ConsumerType.INDUSTRIAL: 50.0,    # industrial zones
    ConsumerType.MIXED: 10.0,         # mixed-use default
}


@dataclass
class LoadProfile:
    """Time-varying load profile for a single node."""

    node_id: str
    consumer_type: ConsumerType
    peak_mw: float                      # Peak demand (MW)
    annual_mwh: float                   # Total annual energy
    hourly_shape: np.ndarray            # 24-element normalized profile
    electrification_factor: float = 1.0  # Multiplier from electrification scenario

    @property
    def load_factor(self) -> float:
        """Ratio of average to peak demand."""
        return float(np.mean(self.hourly_shape))

    @property
    def hourly_mw(self) -> np.ndarray:
        """Hourly demand in MW (24 values)."""
        return self.peak_mw * self.hourly_shape

    def demand_at_hour(self, hour: int) -> float:
        """Return demand in MW at given hour (0-23)."""
        return float(self.peak_mw * self.hourly_shape[hour % 24])

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "consumer_type": self.consumer_type.value,
            "peak_mw": round(self.peak_mw, 4),
            "annual_mwh": round(self.annual_mwh, 2),
            "load_factor": round(self.load_factor, 3),
            "electrification_factor": round(self.electrification_factor, 3),
        }


def _classify_consumer(node_type: str, attrs: dict) -> ConsumerType:
    """Classify a node into a consumer type based on attributes."""
    if node_type == NodeType.LOAD_PROXY:
        return ConsumerType.MIXED
    if node_type == NodeType.MV_SUBSTATION:
        subtype = attrs.get("substation", "")
        if subtype == "industrial":
            return ConsumerType.INDUSTRIAL
        return ConsumerType.COMMERCIAL
    return ConsumerType.MIXED


def generate_load_profiles(
    graph: PowerGraph,
    electrification_factor: float = 1.0,
    density_scaling: bool = True,
    seed: int = 42,
) -> list[LoadProfile]:
    """Generate load profiles for all demand nodes in the graph.

    Parameters
    ----------
    graph : PowerGraph
        Power grid graph.
    electrification_factor : float
        Multiplier on base demand (1.0 = baseline, 1.5 = moderate electrification,
        2.5 = aggressive). This is the key endogenous demand parameter.
    density_scaling : bool
        If True, scale demand by local node density (proxy for urban density).
    seed : int
        Random seed for stochastic variation.

    Returns
    -------
    list[LoadProfile]
        One profile per demand node (loads, MV substations, transformers).
    """
    rng = np.random.default_rng(seed)
    profiles: list[LoadProfile] = []

    # Identify demand nodes
    demand_types = {NodeType.LOAD_PROXY, NodeType.MV_SUBSTATION, NodeType.TRANSFORMER}
    demand_nodes = [n for n in graph._nodes.values() if n.node_type in demand_types]

    if not demand_nodes:
        logger.warning("No demand nodes found in graph")
        return profiles

    # Compute density scaling (nodes per unit area as proxy)
    density_factors = np.ones(len(demand_nodes))
    if density_scaling and len(demand_nodes) > 1:
        from scipy.spatial import cKDTree
        coords = np.array([(n.lon, n.lat) for n in demand_nodes])
        tree = cKDTree(coords)
        # Count neighbors within ~1km
        counts = tree.query_ball_point(coords, r=1.0 / 111.0)
        raw = np.array([len(c) for c in counts], dtype=float)
        # Normalize to [0.5, 2.0] range
        if raw.max() > raw.min():
            density_factors = 0.5 + 1.5 * (raw - raw.min()) / (raw.max() - raw.min())
        else:
            density_factors = np.ones(len(demand_nodes))

    for i, node in enumerate(demand_nodes):
        ctype = _classify_consumer(node.node_type, node.attrs)

        # Base demand
        base_annual = _BASE_ANNUAL_MWH[ctype]

        # Scale by electrification and density
        annual_mwh = base_annual * electrification_factor * density_factors[i]

        # Add stochastic variation (±20%)
        annual_mwh *= rng.uniform(0.8, 1.2)

        # Peak demand from annual energy and load shape
        shape = _LOAD_SHAPES[ctype].copy()
        peak_mw = annual_mwh / (8760.0 * float(np.mean(shape)))

        profiles.append(LoadProfile(
            node_id=node.id,
            consumer_type=ctype,
            peak_mw=float(peak_mw),
            annual_mwh=float(annual_mwh),
            hourly_shape=shape,
            electrification_factor=electrification_factor,
        ))

    total_peak = sum(p.peak_mw for p in profiles)
    total_energy = sum(p.annual_mwh for p in profiles)
    logger.info(
        "Generated %d load profiles: total_peak=%.2f MW, total_energy=%.0f MWh/yr, "
        "electrification=%.1fx",
        len(profiles), total_peak, total_energy, electrification_factor,
    )
    return profiles

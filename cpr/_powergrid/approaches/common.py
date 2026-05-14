"""Shared types and utilities for all power grid generation approaches.

Defines the canonical graph representation: PowerNode, PowerEdge, PowerGraph.
All approaches produce a PowerGraph, which can be serialized to graph.json
and converted to GeoJSON for frontend rendering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point, mapping, shape

logger = logging.getLogger(__name__)


class VoltageClass(str, Enum):
    HV = "HV"
    MV = "MV"
    LV = "LV"
    UNKNOWN = "UNKNOWN"


class NodeType(str, Enum):
    PLANT = "plant"
    HV_SUBSTATION = "hv_substation"
    MV_SUBSTATION = "mv_substation"
    TRANSFORMER = "transformer"
    LOAD_PROXY = "load_proxy"
    TOWER = "tower"
    JUNCTION = "junction"


@dataclass
class PowerNode:
    """A node in the power grid graph."""

    id: str
    node_type: str  # NodeType value
    lon: float
    lat: float
    voltage_class: str = "UNKNOWN"  # VoltageClass value
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def point(self) -> Point:
        return Point(self.lon, self.lat)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.node_type,
            "lon": self.lon,
            "lat": self.lat,
            "voltage_class": self.voltage_class,
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PowerNode:
        return cls(
            id=d["id"],
            node_type=d["type"],
            lon=d["lon"],
            lat=d["lat"],
            voltage_class=d.get("voltage_class", "UNKNOWN"),
            attrs=d.get("attrs", {}),
        )


@dataclass
class PowerEdge:
    """An edge in the power grid graph."""

    u: str
    v: str
    voltage_class: str = "UNKNOWN"
    length_m: float = 0.0
    cost: float = 0.0
    geometry: list[tuple[float, float]] | None = None  # list of (lon, lat)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def linestring(self) -> LineString | None:
        if self.geometry and len(self.geometry) >= 2:
            return LineString(self.geometry)
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "u": self.u,
            "v": self.v,
            "voltage_class": self.voltage_class,
            "length_m": round(self.length_m, 1),
            "cost": round(self.cost, 2),
            "attrs": self.attrs,
        }
        if self.geometry:
            d["geometry"] = self.geometry
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PowerEdge:
        return cls(
            u=d["u"],
            v=d["v"],
            voltage_class=d.get("voltage_class", "UNKNOWN"),
            length_m=d.get("length_m", 0.0),
            cost=d.get("cost", 0.0),
            geometry=d.get("geometry"),
            attrs=d.get("attrs", {}),
        )


class PowerGraph:
    """Wrapper around NetworkX graph with typed power grid nodes and edges.

    Provides serialization to/from graph.json and conversion to GeoJSON
    for frontend rendering.
    """

    def __init__(self, approach: str = "", city_id: str = "") -> None:
        self.G = nx.Graph()
        self.approach = approach
        self.city_id = city_id
        self._nodes: dict[str, PowerNode] = {}
        self._edges: list[PowerEdge] = []

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_node(self, node: PowerNode) -> None:
        self._nodes[node.id] = node
        self.G.add_node(
            node.id,
            node_type=node.node_type,
            lon=node.lon,
            lat=node.lat,
            voltage_class=node.voltage_class,
            **node.attrs,
        )

    def add_edge(self, edge: PowerEdge) -> None:
        self._edges.append(edge)
        self.G.add_edge(
            edge.u, edge.v,
            voltage_class=edge.voltage_class,
            length_m=edge.length_m,
            cost=edge.cost,
            geometry=edge.geometry,
            **edge.attrs,
        )

    def get_node(self, node_id: str) -> PowerNode | None:
        return self._nodes.get(node_id)

    def nodes_by_type(self, node_type: str) -> list[PowerNode]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def edges_by_voltage(self, vc: str) -> list[PowerEdge]:
        return [e for e in self._edges if e.voltage_class == vc]

    def subgraph_by_voltage(self, vc: str) -> nx.Graph:
        """Extract the subgraph containing only edges of a given voltage class."""
        edge_list = [(e.u, e.v) for e in self._edges if e.voltage_class == vc]
        return self.G.edge_subgraph(edge_list).copy()

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "city_id": self.city_id,
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PowerGraph:
        pg = cls(approach=d.get("approach", ""), city_id=d.get("city_id", ""))
        for nd in d.get("nodes", []):
            pg.add_node(PowerNode.from_dict(nd))
        for ed in d.get("edges", []):
            pg.add_edge(PowerEdge.from_dict(ed))
        return pg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Saved graph: %s (%d nodes, %d edges)", path, self.node_count, self.edge_count)

    @classmethod
    def load(cls, path: Path) -> PowerGraph:
        with open(path) as f:
            data = json.load(f)
        pg = cls.from_dict(data)
        logger.info("Loaded graph: %s (%d nodes, %d edges)", path, pg.node_count, pg.edge_count)
        return pg

    # ── GeoJSON conversion ───────────────────────────────────────────

    def nodes_to_geojson(self) -> dict[str, Any]:
        """Convert nodes to GeoJSON FeatureCollection."""
        features = []
        for node in self._nodes.values():
            features.append({
                "type": "Feature",
                "geometry": mapping(node.point),
                "properties": {
                    "id": node.id,
                    "type": node.node_type,
                    "voltage_class": node.voltage_class,
                    "degree": self.G.degree(node.id) if node.id in self.G else 0,
                    **node.attrs,
                },
            })
        return {"type": "FeatureCollection", "features": features}

    def edges_to_geojson(self) -> dict[str, Any]:
        """Convert edges to GeoJSON FeatureCollection."""
        features = []
        for edge in self._edges:
            if edge.geometry and len(edge.geometry) >= 2:
                geom = mapping(LineString(edge.geometry))
            else:
                # Fallback: straight line between nodes
                n_u = self._nodes.get(edge.u)
                n_v = self._nodes.get(edge.v)
                if n_u and n_v:
                    geom = mapping(LineString([(n_u.lon, n_u.lat), (n_v.lon, n_v.lat)]))
                else:
                    continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "u": edge.u,
                    "v": edge.v,
                    "voltage_class": edge.voltage_class,
                    "length_m": round(edge.length_m, 1),
                    "cost": round(edge.cost, 2),
                    **edge.attrs,
                },
            })
        return {"type": "FeatureCollection", "features": features}

    def to_geojson(self) -> dict[str, Any]:
        """Full GeoJSON with both nodes and edges as separate feature collections."""
        return {
            "approach": self.approach,
            "city_id": self.city_id,
            "nodes": self.nodes_to_geojson(),
            "edges": self.edges_to_geojson(),
        }


def geodesic_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate geodesic distance in metres using Haversine formula."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def parse_voltage_kv(voltage_str: str | None) -> float | None:
    """Parse OSM voltage tag (may be semicolon-separated) to kV.

    Returns the maximum voltage if multiple values present.
    OSM stores voltage in volts; we convert to kV.
    """
    if not voltage_str:
        return None
    try:
        parts = voltage_str.replace(",", ";").split(";")
        values = []
        for p in parts:
            p = p.strip()
            if p:
                values.append(float(p) / 1000.0)  # V → kV
        return max(values) if values else None
    except (ValueError, TypeError):
        return None

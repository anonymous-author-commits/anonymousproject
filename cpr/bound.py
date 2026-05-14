"""Empirical verification of Theorem 1: CPR(G) ≤ L · W₁(μ_G, μ*).

For each (model, city) pair the script:
  1. Computes CPR(G | θ_*) via cpr.regret.cpr.
  2. Computes W₁(μ_G, μ_*) on graph signatures via cpr.transport.w1_distance.
  3. Plugs in L (the per-payoff Lipschitz constant from
     cpr.payoff.lipschitz_constant).
  4. Records the ratio CPR / (L · W₁) — should be ≤ 1.

The constructive tightness case from Theorem 1 (linear payoff
π(a, θ) = ⟨a, θ⟩ on a 1-D state) is not run by this module, since
that is purely a proof check; the empirical multigraph case is the
relevant signal for the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np

from .graphs import graph_signature
from .payoff import PayoffConfig, allocation_payoff
from .regret import CPRResult, cpr
from .transport import w1_distance


@dataclass
class BoundResult:
    cpr_value: float
    w1_value: float
    L: float
    bound: float           # L · W₁
    ratio: float           # cpr / bound; should be ≤ 1
    holds: bool            # cpr ≤ bound (within tol)


def verify_bound(
    model_samples: Sequence[nx.MultiGraph],
    reference_graph: nx.MultiGraph,
    L: float,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    n_actions: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
) -> BoundResult:
    """Compute (CPR, W₁, L·W₁, ratio, holds) for one (model, city) pair."""
    cpr_res = cpr(
        model_samples, reference_graph,
        cfg=cfg, n_actions=n_actions, seed=seed,
    )
    cpr_value = float(cpr_res.cpr)

    # Build the empirical signature distributions on each side.
    sig_model = [graph_signature(g) for g in model_samples] if model_samples else []
    sig_ref = [graph_signature(reference_graph)]

    if not sig_model:
        # No model samples → W₁ undefined; record NaN and skip the
        # bound check.
        return BoundResult(
            cpr_value=cpr_value, w1_value=float("nan"),
            L=L, bound=float("nan"), ratio=float("nan"), holds=False,
        )

    w1_val = w1_distance(sig_model, sig_ref)
    bound = L * w1_val
    ratio = cpr_value / bound if bound > 0 else float("inf")
    holds = cpr_value <= bound + tol
    return BoundResult(
        cpr_value=cpr_value, w1_value=float(w1_val),
        L=float(L), bound=float(bound), ratio=float(ratio), holds=bool(holds),
    )

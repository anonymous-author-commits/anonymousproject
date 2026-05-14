"""Climate Planning Regret loop.

Given:
  - μ_G: a model belief over θ, represented as a finite sample of
         multigraphs from the model.
  - μ_*: a reference graph (single sample treated as the truth).
  - π:   the allocation payoff (cpr.payoff.allocation_payoff).

The CPR procedure:
  1. Enumerate or sample a finite action set A.
  2. For each a ∈ A, compute the model's expected payoff
     E_{θ ~ μ_G}[π(a, θ)] and the truth payoff π(a, θ_*).
  3. Pick a* = argmax_a E_{θ ~ μ_G}[π(a, θ)] (the model's chosen plan).
  4. Pick a*_oracle = argmax_a π(a, θ_*) (the oracle plan under truth).
  5. CPR(G | θ_*) = π(a*_oracle, θ_*) − π(a*, θ_*).

This implements the *deterministic-truth* version: μ_* is a point
mass on the OSM HV graph for that city, which is appropriate when the
ground-truth topology is observable (the paper's "data-sparse, not
data-denied" framing — see §1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import networkx as nx

from .payoff import PayoffConfig, allocation_payoff, grid_action_set


@dataclass
class CPRResult:
    cpr: float                  # the regret value
    pi_oracle: float            # max π under θ_*
    pi_chosen: float            # π of the model's chosen plan under θ_*
    a_chosen: list              # the model-chosen action
    a_oracle: list              # the oracle action
    n_actions_evaluated: int


def cpr(
    model_samples: Sequence[nx.MultiGraph],
    reference_graph: nx.MultiGraph,
    *,
    cfg: PayoffConfig = PayoffConfig(),
    n_actions: int = 200,
    grid_size: int = 16,
    tile_size: int = 1536,
    seed: int = 0,
) -> CPRResult:
    """Compute CPR for a single (model, city) pair.

    The action space is a fixed grid of substation-site coordinates,
    shared between the model and the truth graph. The model's chosen
    plan is the action that maximises its expected payoff over μ_G;
    regret is computed against the oracle plan under θ_*.
    """
    actions = grid_action_set(
        K=cfg.K, grid_size=grid_size, tile_size=tile_size,
        n_actions=n_actions, seed=seed,
    )
    if not actions:
        return CPRResult(
            cpr=float("nan"), pi_oracle=float("nan"), pi_chosen=float("nan"),
            a_chosen=[], a_oracle=[], n_actions_evaluated=0,
        )

    truth_payoff = np.array([
        allocation_payoff(a, reference_graph, cfg=cfg)
        for a in actions
    ])

    if not model_samples:
        chosen_idx = int(np.random.default_rng(seed).integers(len(actions)))
    else:
        model_payoff = np.zeros(len(actions))
        for a_i, a in enumerate(actions):
            scores = [
                allocation_payoff(a, theta_g, cfg=cfg)
                for theta_g in model_samples
            ]
            model_payoff[a_i] = float(np.mean(scores)) if scores else -np.inf
        chosen_idx = int(np.argmax(model_payoff))

    oracle_idx = int(np.argmax(truth_payoff))
    pi_oracle = float(truth_payoff[oracle_idx])
    pi_chosen = float(truth_payoff[chosen_idx])
    return CPRResult(
        cpr=pi_oracle - pi_chosen,
        pi_oracle=pi_oracle,
        pi_chosen=pi_chosen,
        a_chosen=actions[chosen_idx],
        a_oracle=actions[oracle_idx],
        n_actions_evaluated=len(actions),
    )

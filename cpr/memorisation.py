"""Memorisation / mode-collapse diagnostics.

Two metrics for each (generator, city) cell:

  nn_train_signature_distance(g, training_signatures):
      L2 distance from the generator's predicted graph signature
      to its closest training-corpus city's truth signature in the
      same signature space. Low values indicate the generator may
      be reproducing a memorised training-corpus pattern; high
      values indicate genuine adaptation to the held-out city's
      conditioning.

  cross_city_variance(per_city_signatures):
      Variance of a generator's signature across held-out cities.
      A generator that produces the same graph regardless of
      conditioning input will have near-zero across-city variance
      (mode collapse); a generator that adapts to per-city
      conditioning will have variance comparable to the truth
      graphs' across-city variance.

Together, these two scalars distinguish four regimes:
    NN_train low  + cross-city low   ->  memorised single pattern
    NN_train low  + cross-city high  ->  per-city memorisation
    NN_train high + cross-city low   ->  novel single pattern (mode collapse)
    NN_train high + cross-city high  ->  genuine generalisation

CPR penalises modes that look "right" globally but lack
spatial alignment to the conditioning, which corresponds to the
upper-right and lower-right quadrants of this taxonomy. Pixel
metrics (Dice) cannot distinguish these regimes from each other.
"""
from __future__ import annotations

import numpy as np
import networkx as nx


def nn_train_signature_distance(
    generator_signature: np.ndarray,
    training_signatures: list[np.ndarray],
) -> tuple[float, int]:
    """Distance from `generator_signature` to its closest training
    signature, in L2 over the (28-dim) graph signature space.

    Returns ``(distance, idx)`` where ``idx`` is the index into
    ``training_signatures`` of the nearest training graph.
    """
    if not training_signatures:
        return float("nan"), -1
    sig = np.asarray(generator_signature, dtype=np.float64)
    dists = [
        float(np.linalg.norm(sig - np.asarray(s, dtype=np.float64)))
        for s in training_signatures
    ]
    idx = int(np.argmin(dists))
    return float(dists[idx]), idx


def cross_city_variance(per_city_signatures: list[np.ndarray]) -> float:
    """Mean per-coordinate variance of signatures across held-out
    cities. A generator that emits the same graph regardless of
    input has variance near zero; a generator that adapts to the
    conditioning has variance comparable to the truth graphs'.
    """
    if len(per_city_signatures) < 2:
        return float("nan")
    arr = np.stack(
        [np.asarray(s, dtype=np.float64) for s in per_city_signatures]
    )
    return float(arr.var(axis=0, ddof=1).mean())


def memorisation_taxonomy(
    nn_train_dist: float,
    cross_city_var: float,
    *,
    nn_train_threshold: float,
    var_threshold: float,
) -> str:
    """Categorise a (generator) into one of the four diagnostic
    regimes. Thresholds are typically the median across the panel.
    """
    if nn_train_dist < nn_train_threshold:
        if cross_city_var < var_threshold:
            return "memorised-single-pattern"
        return "per-city-memorisation"
    if cross_city_var < var_threshold:
        return "mode-collapse-novel"
    return "genuine-generalisation"


__all__ = [
    "nn_train_signature_distance",
    "cross_city_variance",
    "memorisation_taxonomy",
]

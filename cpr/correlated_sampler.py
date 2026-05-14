"""Correlated Bernoulli sampling via Gaussian copula.

A reviewer of the Track A draft pointed out (correctly) that
independent per-pixel Bernoulli sampling on a generator's sigmoid
output destroys spatial correlations -- exactly the structural
priors a generative topology model is supposed to encode. The
sampled masks become fragmented dust rather than line topology.

This module implements the standard Gaussian-copula construction
that preserves the Bernoulli marginals while introducing tunable
spatial correlation:

  1. Probit-transform the per-pixel probabilities:
       q(i, j) = Phi^{-1}(p(i, j))
  2. Sample isotropic white noise z(i, j) ~ N(0, 1).
  3. Smooth z with a Gaussian kernel of bandwidth sigma_px;
     re-normalise to unit marginal variance to give g.
  4. Threshold: mask(i, j) = 1 iff g(i, j) < q(i, j).

By construction, each pixel's marginal probability is
Phi(q) = p, matching the Bernoulli. The spatial correlation of the
mask follows the correlation of g, which is determined by sigma_px.
At sigma_px = 0 (no smoothing) we recover independent Bernoulli;
larger sigma_px gives smoother, more topology-preserving samples.

We use sigma_px = 8 px = 80 m as a reasonable default for HV mask
synthesis at 10 m/px. Sensitivity to this choice is reported in
the paper.
"""
from __future__ import annotations

import numpy as np

# scipy is already a dependency of the broader pipeline (used by
# scikit-image and POT). Import locally to keep this module
# import-clean when scipy is unavailable.


def _probit(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF, applied elementwise. Clips p
    away from {0, 1} to avoid +/-inf."""
    from scipy.stats import norm

    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return norm.ppf(p_clipped)


def correlated_bernoulli_sample(
    prob: np.ndarray,
    *,
    sigma_px: float = 8.0,
    seed: int = 0,
) -> np.ndarray:
    """Sample a binary mask from a Gaussian-copula model.

    Marginal P(mask[i, j] = 1) = prob[i, j] exactly. Spatial
    correlation of mask is induced by Gaussian smoothing of an
    underlying white-noise field with bandwidth ``sigma_px``.

    Parameters
    ----------
    prob : 2-D float array in [0, 1]
        Per-pixel Bernoulli marginals.
    sigma_px : float
        Smoothing kernel standard deviation in pixels. Set to 0 for
        independent Bernoulli (recovers the baseline used in the
        Track A pipeline).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    mask : 2-D uint8 array
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    q = _probit(prob.astype(np.float64))

    z = rng.standard_normal(prob.shape)
    if sigma_px > 0:
        g = gaussian_filter(z, sigma=sigma_px, mode="reflect")
        # Renormalise: Gaussian smoothing reduces variance.
        g_std = g.std()
        if g_std > 0:
            g = g / g_std
    else:
        g = z  # already unit variance

    return (g < q).astype(np.uint8)


def empirical_correlation_length(
    prob: np.ndarray,
    *,
    sigma_px: float,
    n_samples: int = 32,
    seed: int = 0,
) -> float:
    """Diagnostic: empirical horizontal autocorrelation length of the
    sampled mask, in pixels.

    Useful to verify the sigma_px hyperparameter produces samples
    with the expected spatial-correlation scale.
    """
    rng = np.random.default_rng(seed)
    sumcorr = np.zeros(64)
    for _ in range(n_samples):
        s = rng.integers(0, 2**31 - 1)
        m = correlated_bernoulli_sample(prob, sigma_px=sigma_px, seed=int(s))
        # Compute 1-D horizontal autocorrelation along a random row.
        row_idx = rng.integers(0, m.shape[0])
        row = m[row_idx].astype(np.float64) - m[row_idx].mean()
        denom = (row * row).sum()
        if denom > 0:
            for k in range(64):
                if k == 0:
                    c = 1.0
                else:
                    c = (row[:-k] * row[k:]).sum() / denom
                sumcorr[k] += c
    sumcorr /= n_samples
    # First lag where autocorrelation drops below 1/e.
    target = 1.0 / np.e
    above = np.where(sumcorr > target)[0]
    return float(above[-1]) if len(above) else 0.0

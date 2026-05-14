"""Sample-based W₁ estimator on graph-signature embeddings.

W₁(μ_a, μ_b) is computed by:
  1. Embedding each graph sample through ``graph_signature``.
  2. Solving the discrete optimal-transport problem on the resulting
     point cloud with the L2 ground metric.

Backends:
  - POT (``ot``) ``ot.emd2`` — exact, preferred when N ≤ 1000.
  - SciPy ``linear_sum_assignment`` — exact only when point counts
    are equal; used as a fallback if POT is unavailable.

For the paper's empirical bound verification (Theorem 1), we never
need W₁ > 1000 samples per side, so the exact path is always
sufficient.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import ot  # POT
    _HAS_POT = True
except Exception:
    _HAS_POT = False

from scipy.optimize import linear_sum_assignment
import networkx as nx


def _l2_cost_matrix(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Pairwise L2 cost matrix (N_a, N_b)."""
    return np.sqrt(((xs[:, None, :] - ys[None, :, :]) ** 2).sum(axis=-1))


def w1_distance(
    samples_a: Sequence[np.ndarray],
    samples_b: Sequence[np.ndarray],
) -> float:
    """W₁ between two empirical distributions of graph signatures.

    samples_a, samples_b: sequences of fixed-length feature vectors.
    Returns the W₁ distance as a non-negative float.
    """
    xs = np.asarray(samples_a, dtype=np.float64)
    ys = np.asarray(samples_b, dtype=np.float64)
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    if xs.ndim == 1:
        xs = xs[None, :]
    if ys.ndim == 1:
        ys = ys[None, :]

    cost = _l2_cost_matrix(xs, ys)

    if _HAS_POT:
        wa = np.full(xs.shape[0], 1.0 / xs.shape[0])
        wb = np.full(ys.shape[0], 1.0 / ys.shape[0])
        return float(ot.emd2(wa, wb, cost))

    # SciPy fallback — requires equal sizes; pad with zero-cost dummies
    # if uneven (tight only when sizes match exactly).
    n_a, n_b = xs.shape[0], ys.shape[0]
    if n_a != n_b:
        # Tile the smaller side to equalize. This is exact only when
        # one side divides the other; otherwise it is a tight upper
        # bound. For the paper we use ≤ 5 samples per side and POT is
        # the primary backend, so this branch is a safety net.
        if n_a < n_b:
            reps = (n_b + n_a - 1) // n_a
            xs = np.tile(xs, (reps, 1))[:n_b]
        else:
            reps = (n_a + n_b - 1) // n_b
            ys = np.tile(ys, (reps, 1))[:n_a]
        cost = _l2_cost_matrix(xs, ys)
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def signature_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Single-pair L2 distance on graph signatures."""
    return float(np.linalg.norm(np.asarray(sig_a) - np.asarray(sig_b)))


def node_w1_distance(
    g1, g2, *, max_pts: int = 500, seed: int = 0,
) -> float:
    """W1 between two graphs' empirical node-coordinate distributions.

    Each graph contributes its set of node (row, col) coordinates as
    a uniform empirical measure. Ground metric is L2 in pixel space.
    For balanced cardinalities this reduces to the optimal-assignment
    cost; for unbalanced, POT's emd2 handles the mass imbalance via
    uniform weights.

    Bernoulli-sampled graphs can have 10k+ nodes, on which POT's
    exact ``emd2`` solves an LP with prohibitive cost. We
    deterministically subsample distributions larger than ``max_pts``
    using ``numpy.random.default_rng(seed)``. Subsampling preserves
    the expected W1 to within $O(\\log(n) / \\sqrt{\\mathrm{max\\_pts}})$
    by standard concentration, which we tolerate for the bound
    informativity claim.

    This is the **natural metric for the wire-cost payoff**: per
    Theorem~1, the per-node-mean payoff is exactly 1-Lipschitz
    in this metric.
    """
    pts1 = np.array(
        [list(g1.nodes[n].get("px", n)) for n in g1.nodes()],
        dtype=np.float64,
    )
    pts2 = np.array(
        [list(g2.nodes[n].get("px", n)) for n in g2.nodes()],
        dtype=np.float64,
    )
    if pts1.size == 0 or pts2.size == 0:
        return float("nan")
    if pts1.ndim == 1:
        pts1 = pts1.reshape(1, -1)
    if pts2.ndim == 1:
        pts2 = pts2.reshape(1, -1)

    rng = np.random.default_rng(seed)
    if pts1.shape[0] > max_pts:
        idx = rng.choice(pts1.shape[0], size=max_pts, replace=False)
        pts1 = pts1[idx]
    if pts2.shape[0] > max_pts:
        idx = rng.choice(pts2.shape[0], size=max_pts, replace=False)
        pts2 = pts2[idx]

    cost = _l2_cost_matrix(pts1, pts2)

    if _HAS_POT:
        wa = np.full(pts1.shape[0], 1.0 / pts1.shape[0])
        wb = np.full(pts2.shape[0], 1.0 / pts2.shape[0])
        return float(ot.emd2(wa, wb, cost))

    # Fallback: balanced Hungarian assignment, exact only for n_a==n_b.
    n_a, n_b = pts1.shape[0], pts2.shape[0]
    if n_a != n_b:
        if n_a < n_b:
            reps = (n_b + n_a - 1) // n_a
            pts1 = np.tile(pts1, (reps, 1))[:n_b]
        else:
            reps = (n_a + n_b - 1) // n_b
            pts2 = np.tile(pts2, (reps, 1))[:n_a]
        cost = _l2_cost_matrix(pts1, pts2)
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def stratified_node_w1_distance(
    g1, g2, *,
    max_pts: int = 500,
    weight_fn: str = "betweenness",
    seed: int = 0,
) -> float:
    """Node-W1 with sampling weighted toward high-betweenness nodes.

    The default uniform subsampler in ``node_w1_distance`` preserves
    the overall coordinate distribution by construction but does not
    preserve specific cut-vertices --- nodes that, if removed,
    disconnect the graph. Cut-set-sensitive payoffs (e.g.\\ N-1
    contingency planning) therefore want their W1 estimate biased
    toward retaining such nodes in the sample.

    This function samples each graph's nodes proportionally to their
    betweenness centrality (weighted by 1 + b(v) where b is the
    NetworkX-computed betweenness, so all nodes have non-zero
    sampling probability and high-betweenness nodes are
    over-represented). The remaining computation is identical to
    ``node_w1_distance``.

    Parameters
    ----------
    g1, g2 : networkx.Graph or networkx.MultiGraph
        Graphs with ``px`` (row, col) node attributes.
    max_pts : int
        Maximum nodes per side after subsampling.
    weight_fn : {"betweenness", "degree"}
        Centrality weighting. ``"betweenness"`` is the default;
        ``"degree"`` is a faster heuristic that approximates it.
    seed : int
        Numpy RNG seed.
    """
    rng = np.random.default_rng(seed)

    def _stratified_sample(g):
        nodes = list(g.nodes())
        if len(nodes) <= max_pts:
            pts = np.array(
                [list(g.nodes[n].get("px", n)) for n in nodes],
                dtype=np.float64,
            )
            return pts
        if weight_fn == "betweenness":
            try:
                bw = nx.betweenness_centrality(
                    g, k=min(64, len(nodes)), seed=int(seed),
                )
            except Exception:
                bw = {n: 0.0 for n in nodes}
        else:  # "degree"
            bw = dict(g.degree())
        weights = np.array([1.0 + float(bw[n]) for n in nodes])
        weights = weights / weights.sum()
        idx = rng.choice(len(nodes), size=max_pts, replace=False, p=weights)
        chosen = [nodes[i] for i in idx]
        return np.array(
            [list(g.nodes[n].get("px", n)) for n in chosen],
            dtype=np.float64,
        )

    pts1 = _stratified_sample(g1)
    pts2 = _stratified_sample(g2)
    if pts1.size == 0 or pts2.size == 0:
        return float("nan")

    cost = _l2_cost_matrix(pts1, pts2)
    if _HAS_POT:
        wa = np.full(pts1.shape[0], 1.0 / pts1.shape[0])
        wb = np.full(pts2.shape[0], 1.0 / pts2.shape[0])
        return float(ot.emd2(wa, wb, cost))

    n_a, n_b = pts1.shape[0], pts2.shape[0]
    if n_a != n_b:
        if n_a < n_b:
            reps = (n_b + n_a - 1) // n_a
            pts1 = np.tile(pts1, (reps, 1))[:n_b]
        else:
            reps = (n_a + n_b - 1) // n_b
            pts2 = np.tile(pts2, (reps, 1))[:n_a]
        cost = _l2_cost_matrix(pts1, pts2)
    row, col = linear_sum_assignment(cost)
    return float(cost[row, col].mean())


def _node_coords(g) -> np.ndarray:
    """Pixel coordinates as (N, 2) float array; empty array if no nodes."""
    if g.number_of_nodes() == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.array(
        [list(g.nodes[n].get("px", n)) for n in g.nodes()],
        dtype=np.float64,
    )
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    return pts


def _kmeans_labels(
    points: np.ndarray, k: int, *, seed: int = 0, n_iter: int = 50,
) -> np.ndarray:
    """Lloyd's k-means returning a per-point cluster label in [0, k).

    Self-contained NumPy implementation; avoids the scikit-learn
    dependency for this lightweight in-package call. Initialisation is
    k-means++ on a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    if n == 0 or k <= 0:
        return np.zeros(n, dtype=np.int64)
    k = min(k, n)

    # k-means++ initialisation.
    centres = np.empty((k, points.shape[1]), dtype=np.float64)
    centres[0] = points[rng.integers(0, n)]
    for i in range(1, k):
        d2 = ((points[:, None, :] - centres[None, :i, :]) ** 2).sum(axis=-1)
        d2 = d2.min(axis=1)
        if d2.sum() == 0:
            centres[i] = points[rng.integers(0, n)]
        else:
            probs = d2 / d2.sum()
            centres[i] = points[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(axis=-1)
        new_labels = d2.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if mask.any():
                centres[j] = points[mask].mean(axis=0)
    return labels


def hierarchical_node_w1_distance(
    g1, g2, *,
    k_regions: int = 8,
    max_pts: int = 500,
    seed: int = 0,
    return_breakdown: bool = False,
):
    """Hierarchical optimal-transport (HOT) approximation to node-W1.

    Two-tier decomposition: (i) intra-cluster W1 on each of K spatial
    regions formed by k-means on the union of node coordinates;
    (ii) inter-cluster mass-imbalance OT on the K cluster centroids
    using truth/generator mass profiles. The two terms together avoid
    the systematic under-estimation of a naive per-region sum, which
    misses transport of mass across regions when the two graphs have
    different cluster mass profiles.

    Concretely we compute
        W1_hier  =  intra  +  inter
    where ``intra`` is the mass-weighted average of per-region W1
    distances and ``inter`` is the W1 between the K-dimensional truth
    and generator cluster-mass profiles under L2 cost on cluster
    centroids. This is a standard HOT relaxation; it is a lower bound
    on global W1, but a tighter one than the per-region sum alone.

    Parameters
    ----------
    g1, g2
        NetworkX graphs with ``px`` node attributes (row, col pixel
        coordinates).
    k_regions
        Number of spatial clusters. Recommended K=4 for cells with
        |V| < 200; K=8 for |V| in [200, 2000]; K=16 for larger.
    max_pts
        Forwarded to ``node_w1_distance`` per region.
    seed
        Reproducibility seed for the k-means initialisation and the
        per-region subsample.
    return_breakdown
        If True, return ``(W1_hier, info)`` where ``info`` is a dict
        ``{intra, inter, regions: [...]}`` so callers can inspect
        the decomposition.

    Returns
    -------
    float (or (float, dict) if return_breakdown=True)
        Hierarchical W1 estimate. If either graph is empty, returns
        ``nan``.
    """
    p1 = _node_coords(g1)
    p2 = _node_coords(g2)
    if p1.size == 0 or p2.size == 0:
        nan = float("nan")
        if return_breakdown:
            return nan, {"intra": nan, "inter": nan, "regions": []}
        return nan

    # Cluster the union for a stable region partition.
    union = np.vstack([p1, p2])
    labels_union = _kmeans_labels(union, k_regions, seed=seed)
    labels_1 = labels_union[: p1.shape[0]]
    labels_2 = labels_union[p1.shape[0] :]
    K = min(k_regions, union.shape[0])

    # Intra-cluster: mass-weighted average of per-region node-W1.
    centroids = np.zeros((K, 2), dtype=np.float64)
    mass_1 = np.zeros(K, dtype=np.float64)
    mass_2 = np.zeros(K, dtype=np.float64)
    n1_total = float(p1.shape[0])
    n2_total = float(p2.shape[0])
    intra = 0.0
    regions: list[dict] = []

    for r in range(K):
        mask1 = labels_1 == r
        mask2 = labels_2 == r
        n1 = int(mask1.sum())
        n2 = int(mask2.sum())
        mass_1[r] = n1 / max(n1_total, 1.0)
        mass_2[r] = n2 / max(n2_total, 1.0)
        if (n1 + n2) > 0:
            stack = np.vstack([p1[mask1], p2[mask2]])
            centroids[r] = stack.mean(axis=0)
        if n1 == 0 or n2 == 0:
            w1_r = 0.0  # mass-imbalance is captured by the inter term
            regions.append({
                "region": r, "n1": n1, "n2": n2, "w1": w1_r,
                "mass_1": mass_1[r], "mass_2": mass_2[r],
            })
            continue
        sub1 = nx.Graph()
        sub1.add_nodes_from(
            (i, {"px": p1[mask1][i].tolist()}) for i in range(n1)
        )
        sub2 = nx.Graph()
        sub2.add_nodes_from(
            (i, {"px": p2[mask2][i].tolist()}) for i in range(n2)
        )
        w1_r = node_w1_distance(sub1, sub2, max_pts=max_pts, seed=seed)
        # Local intra weight: harmonic mean of marginal masses, which
        # equals min(mass_1[r], mass_2[r]) up to a constant -- bounds
        # the mass transported locally by the smaller cluster mass.
        intra += min(mass_1[r], mass_2[r]) * w1_r
        regions.append({
            "region": r, "n1": n1, "n2": n2, "w1": w1_r,
            "mass_1": mass_1[r], "mass_2": mass_2[r],
        })

    # Inter-cluster: W1 between the K-dim cluster-mass profiles on
    # cluster centroids. This catches mass that must move *between*
    # clusters (the term the naive per-region sum misses).
    if K > 1 and _HAS_POT:
        cost = _l2_cost_matrix(centroids, centroids)
        inter = float(ot.emd2(mass_1, mass_2, cost))
    elif K > 1:
        # SciPy fallback via balanced LP -- approximate via greedy
        # mass-transfer along centroid distance order.
        cost = _l2_cost_matrix(centroids, centroids)
        diff = mass_1 - mass_2  # +ve = excess truth mass in cluster
        idx = np.argsort(np.abs(diff))[::-1]
        inter = 0.0
        rem = diff.copy()
        for i in idx:
            if rem[i] <= 1e-9:
                continue
            order = np.argsort(cost[i])
            for j in order:
                if i == j or rem[j] >= -1e-9:
                    continue
                send = min(rem[i], -rem[j])
                inter += send * cost[i, j]
                rem[i] -= send
                rem[j] += send
                if rem[i] <= 1e-9:
                    break
    else:
        inter = 0.0

    result = intra + inter
    if return_breakdown:
        return result, {
            "intra": intra,
            "inter": inter,
            "regions": regions,
            "centroids": centroids,
            "mass_1": mass_1,
            "mass_2": mass_2,
        }
    return result


def hausdorff_distance(g1, g2) -> float:
    """Symmetric Hausdorff distance on graph node-coordinate sets.

    d_H(A, B) = max( sup_{a in A} d(a, B), sup_{b in B} d(b, A) ).

    Provided alongside ``node_w1_distance`` for completeness; the
    paper uses W1 because Hausdorff is dominated by single
    outlier nodes and is therefore brittle.
    """
    pts1 = np.array(
        [list(g1.nodes[n].get("px", n)) for n in g1.nodes()],
        dtype=np.float64,
    )
    pts2 = np.array(
        [list(g2.nodes[n].get("px", n)) for n in g2.nodes()],
        dtype=np.float64,
    )
    if pts1.size == 0 or pts2.size == 0:
        return float("nan")
    if pts1.ndim == 1:
        pts1 = pts1.reshape(1, -1)
    if pts2.ndim == 1:
        pts2 = pts2.reshape(1, -1)
    cost = _l2_cost_matrix(pts1, pts2)
    fwd = float(cost.min(axis=1).max())
    bwd = float(cost.min(axis=0).max())
    return max(fwd, bwd)

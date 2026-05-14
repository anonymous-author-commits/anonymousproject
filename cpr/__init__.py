"""Climate Planning Regret (CPR) — decision-theoretic evaluation.

Public surface:
  - graphs.extract_graph(mask) → networkx.MultiGraph
  - payoff.allocation_payoff(action, theta) → float
  - payoff.lipschitz_constant(payoff_fn, samples) → float
  - transport.w1_distance(mu_a, mu_b) → float
  - regret.cpr(model_samples, reference_graph, payoff_fn) → float
  - bound.verify_bound(...) → BoundResult

The W₁ bound on CPR (Theorem 1 in the paper):
    CPR(G) ≤ L · W₁(μ_G, μ*)
with L the Lipschitz constant of the allocation payoff in θ.
"""

__all__ = [
    "graphs",
    "payoff",
    "transport",
    "regret",
    "bound",
]

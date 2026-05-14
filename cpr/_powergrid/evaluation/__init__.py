"""DC power-flow evaluator vendored from tessera.powergrid.

Only the components used by cpr.payoff_powerflow are re-exported here;
the full evaluation suite (contingency analysis, reliability metrics,
cross-city tooling) lives upstream.
"""

from .powerflow import DCPowerFlow, run_dc_powerflow

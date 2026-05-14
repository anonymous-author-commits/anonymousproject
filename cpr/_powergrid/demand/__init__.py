"""Endogenous demand modeling subset used by the vendored DC power-flow solver.

Only LoadProfile is required by cpr.payoff_powerflow; broader scenario
construction lives upstream in the full tessera.powergrid distribution.
"""

from .profiles import LoadProfile, generate_load_profiles

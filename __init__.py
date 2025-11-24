"""
Piper Mission Planner (PMP)
===========================

Top-level package for the PMP mission-planning framework.

Subpackages (intended layout):
- pmp.trajectory  : trajectory, transfers, ADCS
- pmp.power       : EPS, probe env/power/thermal
- pmp.edl         : entry–descent–landing kernels
- pmp.comms       : link budgets and radios
- pmp.seasons     : seasonal / ephemeris helpers
- pmp.missions    : mission-specific orchestration (e.g., MAGPIE)
- pmp.legacy      : original MAGPIE scripts, largely unmodified
"""

__all__ = [
    "trajectory",
    "power",
    "edl",
    "comms",
    "seasons",
    "missions",
    "legacy",
]

__version__ = "0.0.1"

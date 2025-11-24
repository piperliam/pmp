"""
Legacy MAGPIE Scripts
=====================

This package contains original MAGPIE scripts, largely unmodified:
- ADCS_0.2a.py
- antenna_0.4.py
- CFD_0.2.py
- comm1d.py
- hex_v1.2a.py
- orbit_1.4c.py
- orbiter_EPS_0.1.py
- probe_sim_v0.3a.py
- reentry_2.1c.py
- season_calc.py

These are kept for reference. Framework-level code should NOT depend
on them directly; instead, use the cleaned wrappers in other pmp
subpackages (e.g., pmp.trajectory.adcs, pmp.power.probe_env, etc.).

If you want to import from a legacy script, it is recommended to:
- Rename it to a valid module name (e.g., `magpie_adcs_v0_2a.py`)
- Then add a corresponding import here.
"""

__all__ = []

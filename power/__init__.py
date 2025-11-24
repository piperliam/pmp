"""
Power & Thermal Subsystems
==========================

Includes:
- Probe environment + power + thermal simulations.
- Orbiter EPS and battery/SOC models.
"""

# Probe surface env/power/thermal

try:
    from .probe_env import (
        ProbeEnvConfig,
        ProbePanelConfig,
        ProbeBatteryConfig,
        ProbeThermalConfig,
        ProbeEnvSimConfig,
        ProbeEnvSimResult,
        simulate_probe_env_power,
    )
except Exception:  # pragma: no cover
    pass

# Orbiter EPS

try:
    from .orbiter_eps import (
        OrbitLightingConfig,
        ArrayConfig,
        BatteryConfig,
        LoadConfig,
        OrbiterEPSSimConfig,
        OrbiterEPSSimResult,
        simulate_orbiter_eps,
        magpie_default_eps_config,
    )
except Exception:  # pragma: no cover
    pass

__all__ = [
    # probe env/power
    "ProbeEnvConfig",
    "ProbePanelConfig",
    "ProbeBatteryConfig",
    "ProbeThermalConfig",
    "ProbeEnvSimConfig",
    "ProbeEnvSimResult",
    "simulate_probe_env_power",
    # orbiter eps
    "OrbitLightingConfig",
    "ArrayConfig",
    "BatteryConfig",
    "LoadConfig",
    "OrbiterEPSSimConfig",
    "OrbiterEPSSimResult",
    "simulate_orbiter_eps",
    "magpie_default_eps_config",
]

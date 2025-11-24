# pmp/power/orbiter_eps.py
"""
Orbiter EPS Simulator (Framework Version)
=========================================

Generic 0D EPS model for an orbiting spacecraft with:
- 1 battery
- 1 solar array
- Sunlit + eclipse portions of each orbit
- Simple constant or piecewise-constant loads

This is inspired by the original MAGPIE `orbiter_EPS_0.1.py` but
rewritten *from scratch* as a clean PMP framework module because the
uploaded legacy script is truncated and cannot be safely wrapped.

Key ideas
---------
- Orbit is represented by:
    * period_s
    * eclipse_fraction (0–1)
- Solar array is represented by:
    * nameplate power at reference irradiance (e.g., 900 W at 1 AU)
    * efficiency and degradation factors
    * pointing efficiency
- Battery is represented by:
    * capacity (Wh)
    * charge/discharge efficiencies
    * SOC limits

The main API is:

    simulate_orbiter_eps(cfg: OrbiterEPSSimConfig) -> OrbiterEPSSimResult

which returns:
- A pandas DataFrame of time history (SOC, powers, flags).
- Per-orbit energy balance summary.

This is mission-agnostic: it works for any central body / orbit as long
as you can specify period and eclipse fraction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class OrbitLightingConfig:
    """
    Simple orbit + lighting configuration.

    period_s        : orbital period [s]
    eclipse_fraction: fraction of period spent in eclipse (0-1)
    """
    period_s: float
    eclipse_fraction: float

    def __post_init__(self):
        if not (0.0 <= self.eclipse_fraction <= 1.0):
            raise ValueError("eclipse_fraction must be between 0 and 1.")


@dataclass
class ArrayConfig:
    """
    Solar array configuration.

    nameplate_power_W  : power at reference irradiance (e.g., 1 AU, normal incidence).
    eff_scale          : generic scaling factor (e.g., 0.9 for degradation).
    incidence_eff      : average pointing efficiency (0-1).
    irradiance_scale   : scale factor for irradiance vs reference (e.g., (1 AU / r_AU)^2).
    """
    nameplate_power_W: float = 900.0
    eff_scale: float = 1.0
    incidence_eff: float = 0.9
    irradiance_scale: float = 1.0

    @property
    def effective_power_W(self) -> float:
        return (
            self.nameplate_power_W
            * self.eff_scale
            * self.incidence_eff
            * self.irradiance_scale
        )


@dataclass
class BatteryConfig:
    """
    Orbiter battery configuration.

    cap_Wh        : nominal capacity [Wh]
    soc_min       : lower SOC bound (0-1)
    soc_max       : upper SOC bound (0-1)
    eta_chg       : charge efficiency (0-1)
    eta_dchg      : discharge efficiency (0-1)
    """
    cap_Wh: float = 2500.0
    soc_min: float = 0.2
    soc_max: float = 1.0
    eta_chg: float = 0.95
    eta_dchg: float = 0.95


@dataclass
class LoadConfig:
    """
    Simple load model.

    You can either:
    - Use constant loads (sunlit and eclipse), or
    - Provide a time-varying load function `P_load_func`.

    Constant loads are in Watts. If `P_load_func` is provided, it
    overrides the constant loads.

    P_load_func signature:
        P_load_func(t_s: float, in_sun: bool, orbit_index: int) -> float
    """
    P_sun_W: float = 600.0
    P_eclipse_W: float = 400.0
    P_load_func: Optional[Callable[[float, bool, int], float]] = None


@dataclass
class OrbiterEPSSimConfig:
    """
    Top-level EPS simulation configuration.

    total_orbits : number of orbits to simulate
    dt_s         : timestep [s]
    orbit        : OrbitLightingConfig
    array        : ArrayConfig
    battery      : BatteryConfig
    loads        : LoadConfig
    soc0         : initial state-of-charge (0-1)
    """
    total_orbits: int = 20
    dt_s: float = 10.0
    orbit: OrbitLightingConfig = OrbitLightingConfig(period_s=2 * 3600.0, eclipse_fraction=0.35)
    array: ArrayConfig = ArrayConfig()
    battery: BatteryConfig = BatteryConfig()
    loads: LoadConfig = LoadConfig()
    soc0: float = 0.8


@dataclass
class OrbiterEPSSimResult:
    """
    EPS simulation result.

    df          : time history DataFrame
    orbit_table : per-orbit summary (DataFrame)
    meta        : metadata dict (config echo, flags)
    """
    df: pd.DataFrame
    orbit_table: pd.DataFrame
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary (without full time series)."""
        return {
            "meta": self.meta,
            "n_steps": int(len(self.df)),
            "n_orbits": int(len(self.orbit_table)),
            "columns": list(self.df.columns),
            "orbit_columns": list(self.orbit_table.columns),
        }


# ============================================================
# Core simulator
# ============================================================

def simulate_orbiter_eps(cfg: OrbiterEPSSimConfig) -> OrbiterEPSSimResult:
    """
    Run a 0D EPS simulation for an orbiter.

    States:
        - Battery energy E_batt [Wh]
        - Derived SOC = E_batt / C_batt_Wh

    At each timestep:
        - Determine if in sun or eclipse from orbit phase.
        - Compute P_gen (from array) and P_load.
        - Net power P_net = P_gen - P_load.
        - Battery energy update:
            dE = P_net * dt / 3600, modified by charge/discharge efficiency.
        - Enforce SOC limits [soc_min, soc_max].

    Returns:
        OrbiterEPSSimResult with:
            - df: time history (t, orbit_idx, phase, in_sun, powers, SOC).
            - orbit_table: per-orbit energy generated/consumed and min SOC.
    """

    # Unpack configs
    period = cfg.orbit.period_s
    f_ecl = cfg.orbit.eclipse_fraction
    dt = cfg.dt_s
    n_orbits = cfg.total_orbits

    C_batt = cfg.battery.cap_Wh
    soc_min = cfg.battery.soc_min
    soc_max = cfg.battery.soc_max
    eta_chg = cfg.battery.eta_chg
    eta_dchg = cfg.battery.eta_dchg

    P_array_eff = cfg.array.effective_power_W

    # Time grid
    total_time = n_orbits * period
    n_steps = int(np.floor(total_time / dt)) + 1
    t = np.linspace(0.0, total_time, n_steps)

    # Allocate logs
    orbit_idx = np.zeros(n_steps, dtype=int)
    phase = np.zeros(n_steps)
    in_sun = np.zeros(n_steps, dtype=bool)
    P_gen = np.zeros(n_steps)
    P_ld = np.zeros(n_steps)
    P_net = np.zeros(n_steps)
    soc = np.zeros(n_steps)
    dE_batt = np.zeros(n_steps)

    E_batt = cfg.soc0 * C_batt

    # Per-orbit accumulators
    E_gen_orbit = np.zeros(n_orbits)
    E_load_orbit = np.zeros(n_orbits)
    soc_min_orbit = np.ones(n_orbits) * 1.0

    hit_min_soc_global = False

    for k, tk in enumerate(t):
        # Where in orbit are we?
        this_orbit = int(tk // period)
        if this_orbit >= n_orbits:
            this_orbit = n_orbits - 1  # clamp final
        orbit_idx[k] = this_orbit

        phi = (tk % period) / period  # 0..1
        phase[k] = phi

        # Eclipse if phase in [0, f_ecl)
        sun_flag = phi >= f_ecl
        in_sun[k] = sun_flag

        # Generation
        P_gen_k = P_array_eff if sun_flag else 0.0

        # Loads
        if cfg.loads.P_load_func is not None:
            P_ld_k = cfg.loads.P_load_func(tk, bool(sun_flag), this_orbit)
        else:
            P_ld_k = cfg.loads.P_sun_W if sun_flag else cfg.loads.P_eclipse_W

        P_gen[k] = P_gen_k
        P_ld[k] = P_ld_k

        # Net power
        P_net_k = P_gen_k - P_ld_k
        P_net[k] = P_net_k

        # Battery update
        dE = P_net_k * dt / 3600.0  # Wh
        if dE >= 0.0:
            # charging
            dE_batt_eff = dE * eta_chg
        else:
            # discharging
            dE_batt_eff = dE / eta_dchg

        E_batt = np.clip(E_batt + dE_batt_eff, soc_min * C_batt, soc_max * C_batt)
        dE_batt[k] = dE_batt_eff
        soc_k = E_batt / C_batt
        soc[k] = soc_k

        # Per-orbit accumulation
        E_gen_orbit[this_orbit] += max(P_gen_k, 0.0) * dt / 3600.0
        E_load_orbit[this_orbit] += P_ld_k * dt / 3600.0
        soc_min_orbit[this_orbit] = min(soc_min_orbit[this_orbit], soc_k)

        if soc_k <= soc_min + 1e-6:
            hit_min_soc_global = True

    # Build time history DataFrame
    df = pd.DataFrame(
        {
            "t_s": t,
            "orbit_idx": orbit_idx,
            "phase": phase,
            "in_sun": in_sun,
            "P_gen_W": P_gen,
            "P_load_W": P_ld,
            "P_net_W": P_net,
            "dE_batt_Wh": dE_batt,
            "SOC": soc,
        }
    )

    # Per-orbit table
    orbit_ids = np.arange(n_orbits)
    orbit_table = pd.DataFrame(
        {
            "orbit_idx": orbit_ids,
            "E_gen_Wh": E_gen_orbit,
            "E_load_Wh": E_load_orbit,
            "E_net_Wh": E_gen_orbit - E_load_orbit,
            "SOC_min": soc_min_orbit,
        }
    )

    meta: Dict[str, Any] = {
        "config": {
            "orbit": asdict(cfg.orbit),
            "array": asdict(cfg.array),
            "battery": asdict(cfg.battery),
            "loads": {
                "P_sun_W": cfg.loads.P_sun_W,
                "P_eclipse_W": cfg.loads.P_eclipse_W,
                "has_func": cfg.loads.P_load_func is not None,
            },
            "dt_s": cfg.dt_s,
            "total_orbits": cfg.total_orbits,
            "soc0": cfg.soc0,
        },
        "flags": {
            "hit_min_soc_global": hit_min_soc_global,
        },
    }

    return OrbiterEPSSimResult(df=df, orbit_table=orbit_table, meta=meta)


# ============================================================
# MAGPIE-style convenience preset
# ============================================================

def magpie_default_eps_config() -> OrbiterEPSSimConfig:
    """
    Convenience constructor that approximates the original MAGPIE orbiter EPS:
    - 2 hr orbit with ~35% eclipse
    - 900 W array (effective ~810 W with incidence_eff=0.9)
    - ~2.5 kWh NiMH battery
    - Loads: ~600 W in sun, ~400 W in eclipse
    - SOC0 = 0.8
    """
    orbit = OrbitLightingConfig(period_s=2 * 3600.0, eclipse_fraction=0.35)
    array = ArrayConfig(
        nameplate_power_W=900.0,
        eff_scale=1.0,
        incidence_eff=0.9,
        irradiance_scale=1.0,
    )
    battery = BatteryConfig(
        cap_Wh=2500.0,
        soc_min=0.25,
        soc_max=1.0,
        eta_chg=0.95,
        eta_dchg=0.95,
    )
    loads = LoadConfig(
        P_sun_W=600.0,
        P_eclipse_W=400.0,
        P_load_func=None,
    )
    return OrbiterEPSSimConfig(
        total_orbits=30,
        dt_s=10.0,
        orbit=orbit,
        array=array,
        battery=battery,
        loads=loads,
        soc0=0.8,
    )


__all__ = [
    "OrbitLightingConfig",
    "ArrayConfig",
    "BatteryConfig",
    "LoadConfig",
    "OrbiterEPSSimConfig",
    "OrbiterEPSSimResult",
    "simulate_orbiter_eps",
    "magpie_default_eps_config",
]

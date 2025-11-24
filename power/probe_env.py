# pmp/power/probe_env.py
"""
Probe Environment + Power + Thermal (Framework Wrapper)
=======================================================

This module wraps the MAGPIE probe environment/power/thermal simulator
(`probe_sim_v0.3a.py`) into a clean, mission-agnostic PMP API.

Design goals
------------
- Keep the original, detailed Mars model and `run_sim(...)` *unchanged*,
  but isolate it in `pmp.legacy.magpie_probe_sim_v0_3a`.
- Provide a small number of dataclasses that describe a probe's
  surface environment + power/thermal configs in a way that works for
  *any* body (Mars by default via the underlying EnvConfig).
- Expose a single high-level function:

    simulate_probe_env_power(sim: ProbeEnvSimConfig) -> ProbeEnvSimResult

  which returns:
    - a pandas DataFrame of the full time series, and
    - a summary dict (energy, heater duty, lint messages, etc.).

Generalization
--------------
Right now, the underlying physics is still tuned for Mars:
CO2 atmosphere, Martian sol length, dust optical depth (tau), etc.

However, the framework interface is body-agnostic:

- You can supply any:
    * latitude, elevation
    * day length, solar constant
    * ground albedo/emissivity
- Later we can add a `BodyEnvModel` abstraction to support Earth,
  Moon, Titan, etc., while keeping the same top-level API.

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import pandas as pd

# Import *only* what we need from the legacy MAGPIE sim.
# The idea is: legacy module keeps all detailed internals; this wrapper
# defines the "public" PMP-facing API.
from pmp.legacy import magpie_probe_sim_v0_3a as legacy


# ============================================================
# Public-facing config + result dataclasses
# ============================================================

@dataclass
class ProbeEnvConfig:
    """
    High-level environment configuration for a landed probe.

    This maps directly onto legacy.EnvConfig, but is phrased in
    mission-agnostic terms so you can use it for other bodies
    (with appropriate parameter values).

    For Mars, the default values mirror those in v0.3a.
    """
    lat_deg: float = legacy.LAT_DEG
    lon_deg: float = legacy.LON_DEG
    elev_m: float = legacy.ELEV_M

    # Atmosphere / sky / dust
    tau: float = legacy.TAU_DUST
    albedo: float = legacy.ALBEDO_GROUND
    emissivity_sky: float = legacy.EMISSIVITY_SKY

    # Air temperature profile
    t_mean_K: float = legacy.MARS_MEAN_AIR_TEMP_K
    daily_swing_K: float = legacy.DAILY_TEMP_SWING_K
    wind_mps: float = legacy.WIND_MPS

    # Solar geometry
    solar_const: float = legacy.SOLAR_CONSTANT_MARS
    Ls_deg: float = legacy.Ls_DEG   # areocentric solar longitude / "season"


@dataclass
class ProbePanelConfig:
    """
    Solar panel configuration for the probe.

    Internally this maps onto legacy.PanelConfig plus module-level
    geometry toggles (e.g., GEOMETRY_MODE, fold-out panels).
    Right now we only wrap the electrical efficiency properties.
    """
    eff: float = legacy.PANEL_EFF
    mppt_eff: float = legacy.MPPT_EFF


@dataclass
class ProbeBatteryConfig:
    """
    Main rechargeable battery configuration.
    """
    cap_Wh: float = legacy.BATT_CAP_WH
    v_nom: float = legacy.BATT_V_NOM
    eta_chg: float = legacy.BATT_CHG_EFF
    eta_dchg: float = legacy.BATT_DCHG_EFF
    min_soc: float = legacy.BATT_MIN_SOC


@dataclass
class ProbeThermalConfig:
    """
    Thermal network configuration for the probe + battery.

    These fields map directly to legacy.ThermalConfig. We expose only
    the most mission-relevant knobs; you can always dig deeper into
    the legacy.ThermalConfig if needed.
    """
    emissivity_skin: float = legacy.EMISSIVITY_SKIN
    absorptivity_skin: float = legacy.ABSORPTIVITY_SKIN
    area_out_m2: float = legacy.PROBE_OUTER_AREA_M2
    area_cond_m2: float = legacy.AREA_CONDUCTION_M2

    cork_d_m: float = legacy.CORK_THICK_M
    cork_k_W_mK: float = legacy.CORK_K_W_MK

    mli_internal: bool = False
    mli_batt: bool = False


@dataclass
class ProbeHeaterConfig:
    """
    Heater control logic configuration.

    This wraps the legacy HeaterController defaults. For now, we just
    expose on/off as a simple toggle and delegate the detailed control
    logic (deadbands, failure modes) to the legacy module.
    """
    enable_heater: bool = legacy.ENABLE_HEATER


@dataclass
class ProbeEnvSimConfig:
    """
    Top-level simulation configuration for a landed probe.

    This is the *only* thing user code needs to construct to run a
    probe environment + power + thermal simulation.

    Internally, we map this onto the legacy dataclasses and call
    legacy.run_sim(...).
    """
    # Simulation horizon
    sim_hours: float = legacy.SIM_DURATION_H
    dt_s: float = legacy.SIM_DT_S

    # Configs
    env: ProbeEnvConfig = ProbeEnvConfig()
    panel: ProbePanelConfig = ProbePanelConfig()
    battery: ProbeBatteryConfig = ProbeBatteryConfig()
    thermal: ProbeThermalConfig = ProbeThermalConfig()
    heater: ProbeHeaterConfig = ProbeHeaterConfig()

    # Initial conditions
    init_skin_K: float = 240.0
    init_int_K: float = 245.0
    init_batt_K: float = 248.0
    start_soc: float = 0.80  # fraction of nominal battery capacity

    # Mode schedule: keep default from legacy unless overridden.
    # This is a list of (mode_name, dwell_minutes) tuples, e.g.:
    # [("survival", 720), ("science", 120), ("uplink", 30)]
    mode_schedule: Optional[list[tuple[str, float]]] = None

    # Single-use backup battery enable override (None = keep legacy default)
    enable_single_use_batt: Optional[bool] = None


@dataclass
class ProbeEnvSimResult:
    """
    Output from a probe environment + power + thermal simulation.

    df       : full timeseries (pandas DataFrame)
    summary  : dict of scalar metrics + config echo + lint messages
    """
    df: pd.DataFrame
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary (without the full DataFrame)."""
        meta = dict(self.summary)
        meta["columns"] = list(self.df.columns)
        meta["n_rows"] = int(len(self.df))
        return meta


# ============================================================
# Public API: run a sim in a framework-friendly way
# ============================================================

def simulate_probe_env_power(cfg: ProbeEnvSimConfig) -> ProbeEnvSimResult:
    """
    Run a unified probe environment + power + thermal simulation.

    This is the **framework-facing** API.

    It:
    - Builds legacy EnvConfig, PanelConfig, BatteryConfig, ThermalConfig.
    - Optionally overrides the mode schedule and single-use backup toggle.
    - Calls legacy.run_sim(...).
    - Returns ProbeEnvSimResult with (DataFrame, summary dict).

    It does **not**:
    - Plot anything.
    - Write CSV/JSON/manifest files.
    - Do multi-latitude sweeps or survival stats; those belong in a
      mission-specific layer (e.g., pmp.missions.magpie.probe_sweeps).
    """

    # --- Map high-level configs -> legacy dataclasses ---

    # EnvConfig
    env_legacy = legacy.EnvConfig(
        lat_deg=cfg.env.lat_deg,
        lon_deg=cfg.env.lon_deg,
        elev_m=cfg.env.elev_m,
        tau=cfg.env.tau,
        albedo=cfg.env.albedo,
        emissivity_sky=cfg.env.emissivity_sky,
        t_mean_K=cfg.env.t_mean_K,
        daily_swing_K=cfg.env.daily_swing_K,
        wind_mps=cfg.env.wind_mps,
        solar_const=cfg.env.solar_const,
        Ls_deg=cfg.env.Ls_deg,
    )

    # PanelConfig
    panel_legacy = legacy.PanelConfig(
        eff=cfg.panel.eff,
        mppt_eff=cfg.panel.mppt_eff,
    )

    # BatteryConfig
    batt_legacy = legacy.BatteryConfig(
        cap_Wh=cfg.battery.cap_Wh,
        v_nom=cfg.battery.v_nom,
        eta_chg=cfg.battery.eta_chg,
        eta_dchg=cfg.battery.eta_dchg,
        min_soc=cfg.battery.min_soc,
    )

    # ThermalConfig
    therm_legacy = legacy.ThermalConfig(
        emissivity_skin=cfg.thermal.emissivity_skin,
        absorptivity_skin=cfg.thermal.absorptivity_skin,
        area_out_m2=cfg.thermal.area_out_m2,
        area_cond_m2=cfg.thermal.area_cond_m2,
        cork_d=cfg.thermal.cork_d_m,
        cork_k=cfg.thermal.cork_k_W_mK,
        mli_internal=cfg.thermal.mli_internal,
        mli_batt=cfg.thermal.mli_batt,
    )

    # --- Optional global overrides (mode schedule, backup batt) ---

    # Mode schedule:
    # The legacy code uses a module-level MODE_SCHEDULE. For now,
    # we allow overriding it here in a controlled way. This is a
    # small "impurity" we'll clean up in a future refactor.
    if cfg.mode_schedule is not None:
        legacy.MODE_SCHEDULE = list(cfg.mode_schedule)

    # Single-use backup battery override
    if cfg.enable_single_use_batt is not None:
        legacy.ENABLE_SINGLE_USE_BATT = bool(cfg.enable_single_use_batt)

    # --- Call legacy core integrator ---

    df, summary = legacy.run_sim(
        sim_hours=cfg.sim_hours,
        dt_s=cfg.dt_s,
        panel=panel_legacy,
        batt=batt_legacy,
        therm=therm_legacy,
        env=env_legacy,
        init_skin_K=cfg.init_skin_K,
        init_int_K=cfg.init_int_K,
        init_batt_K=cfg.init_batt_K,
        start_soc=cfg.start_soc,
    )

    return ProbeEnvSimResult(df=df, summary=summary)


__all__ = [
    "ProbeEnvConfig",
    "ProbePanelConfig",
    "ProbeBatteryConfig",
    "ProbeThermalConfig",
    "ProbeHeaterConfig",
    "ProbeEnvSimConfig",
    "ProbeEnvSimResult",
    "simulate_probe_env_power",
]

"""
MAGPIE Unified Environmental + Power + Thermal Simulator (v0.3a)

2025

Liam Piper

"""

from __future__ import annotations
import os, json, math, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# ==================== USER TOGGLES (TOP) ====================
# ============================================================
# --- PLOTS ---
ENABLE_PLOTS_DEFAULT = True
PLOTS_SAVE_DEFAULT   = True
PLOTS_SHOW_DEFAULT   = True

# --- OUTPUTS ---
OUTDIR_DEFAULT         = "."
CSV_NAME_DEFAULT       = "magpie_env_power_thermal_timeseries.csv"
JSON_NAME_DEFAULT      = "magpie_env_power_thermal_summary.json"
WRITE_MANIFEST_DEFAULT = False
SCHEMA_VERSION         = "0.3"

# --- PHYSICS SWITCHES ---
ENABLE_STORMS_DEFAULT  = False   # enables Markov dust storms that raise tau and dust the panels
ENABLE_HEATER_DEFAULT  = True
ENABLE_TILT_DEFAULT    = False   # placeholder (tilt behavior via geometry)
ENABLE_DUST_STORM      = False
ENABLE_CONVECTION_LOSS = True
ENABLE_SOLAR_DIURNAL   = True
ENABLE_HEATER          = True
ENABLE_POWER_CAPPING   = False

# --- DUST/STORM PARAMS ---
TAU0_DEFAULT           = 0.5     # background atmospheric optical depth
K_DUST_DEFAULT         = 0.005   # fractional per-sol panel fouling rate outside storms
CLEAN_PROB_DEFAULT     = 0.05    # probability per sol of a natural cleaning
EFF_RESET_DEFAULT      = 0.95    # efficiency recovery fraction on cleaning
STORM_TAU_DEFAULT      = 1.8     # tau level when in a storm
P_ENTER_DEFAULT        = 0.02    # daily P(storm starts)
P_EXIT_DEFAULT         = 0.25    # daily P(storm ends)
TAU_JITTER_DEFAULT     = 0.03    # ± jitter

# --- HEATER PARAMS ---
HEATER_ON_C_DEFAULT    = -30.0
HEATER_OFF_C_DEFAULT   = 0.0
HEATER_POWER_W_DEFAULT = 2.0

# --- SIM HORIZON ---
SIM_DT_S               = 30.0
SIM_DURATION_H         = 72.0 # real max is 700, set for 72 for ease of verification rn
RANDOM_SEED            = 42

#///////////////////////////////////////////////////////////////////////////////////////////////////
#---------------------------------------------------------------------------------------------------
# --- Environment ---
#---------------------------------------------------------------------------------------------------
#///////////////////////////////////////////////////////////////////////////////////////////////////

# LAT_DEG and LON_DEG define the probe’s surface location on Mars:
#   LAT_DEG  → latitude in degrees (positive = north, negative = south)
#               affects solar angle, day length, and thermal cycles
#               e.g. -90 = south pole, 0 = equator, +90 = north pole
#   LON_DEG  → longitude in degrees (0–360 east-positive convention)
#               only affects local solar time reference (rotation phase)
# Use these together with Ls_DEG to set season and lighting for a given landing site.
LAT_DEG                = -42.2  # -66.32914255 PB: 87.98,15
LON_DEG                = 70.5

ELEV_M                 = 0.0
TAU_DUST               = 0.4
ALBEDO_GROUND          = 0.25
EMISSIVITY_SKY         = 0.92
MARS_MEAN_AIR_TEMP_K   = 210.0
DAILY_TEMP_SWING_K     = 40.0
WIND_MPS               = 5.0
# Ls_DEG controls the "season" on Mars via areocentric solar longitude:
#   0°   = northern spring equinox
#   90°  = northern summer solstice
#   180° = northern autumn equinox
#   270° = northern winter solstice
# Remember: in the **southern hemisphere** (negative latitudes), these are flipped
# (e.g., northern summer ≈ southern winter). Change this if you want to sweep seasons
# at a fixed landing site / latitude.
Ls_DEG                 = 173.746  # areocentric solar longitude [deg]; 90 ≈ north summer peak


# --- Solar / panels ---
SOLAR_CONSTANT_MARS    = 590.0     # W/m² at 1.52 AU nominal
PANEL_EFF              = 0.375      # STC efficiency at 25 °C
MPPT_EFF               = 0.975
TEMP_COEFF_PERCpC      = -0.35     # %/°C power temperature coefficient
NOCT_C                 = 45.0      # Nominal Operating Cell Temp @ 800 W/m², 20 °C, 1 m/s
NOCT_REF_IRR           = 800.0

# ==== Geometry toggles (solar acceptance) ====
GEOMETRY_MODE          = "foldout_top"   # "cone" | "cylinder" | "hybrid" | "foldout" | "foldout_top"
TOP_PANEL_AREA_M2      = 0.275       # top deck area
SIDE_PANEL_AREA_M2     = 0.20       # wrap area
CYLINDER_BANDS         = 2          # number of vertical bands (if cylinder)
CONE_GAIN_BONUS        = 0.35       # diffuse bonus for cone proxy (dimensionless)
CHAR_LEN_M             = 0.35       # characteristic length for convection correlations

# --- Fold-out solar arrays ---
FOLDOUT_DEPLOYED        = True
FOLDOUT_COUNT           = 12 # really 3x but each folds out twice
FOLDOUT_PANEL_AREA_M2   = 0.2     # per fold-out panel area
FOLDOUT_TILT_DEG        = 18.0     # tilt relative to horizontal (0=flat top)

# --- Battery (main) ---
BATT_CAP_WH            = 137.5 # 110 for 4x, 137.5 for 5x
BATT_V_NOM             = 7.2
BATT_CHG_EFF           = 0.98
BATT_DCHG_EFF          = 0.98
BATT_MIN_SOC           = 0.0025

# --- Single-use (non-rechargeable) backup battery ---
ENABLE_SINGLE_USE_BATT  = True
SINGLE_USE_CAP_WH       = 18.0      # total usable energy (Wh), one-shot, no recharge
SINGLE_USE_MAX_W        = 12.0     # optional delivery cap (W); set high to effectively disable

# --- Thermal geometry/materials ---
EMISSIVITY_SKIN        = 0.85
ABSORPTIVITY_SKIN      = 0.65
PROBE_OUTER_AREA_M2    = 0.45
CORK_THICKNESS_M       = 0.03 # 30mm
CORK_K_W_MK            = 0.03 
CROSS_AREA_COND_M2     = 0.19 
C_THERM_SKIN_JK        = 800.0
C_THERM_INTERNAL_JK    = 2200.0
C_THERM_BATT_JK        = 1500.0

# Heater control
HEATER_MAX_W           = 5.0
HEATER_SETPOINT_BATT_K = 258.15   # -15 °C
HEATER_HYSTERESIS_K    = 2.0
HEATER_INTERNAL_BIAS   = 0.21

# --- Multi-Layer Insulation (MLI) toggles ---
ENABLE_MLI_INTERNAL    = True
ENABLE_MLI_BATT        = True
MLI_K_W_MK             = 0.003    # effective radiative-equivalent k for MLI stack
MLI_INTERNAL_THICK_M   = 0.005    # 5 mm internal lining
MLI_BATT_THICK_M       = 0.0075    # 7.5 mm battery wrap

# --- Mode loads ---
POWER_MODES: Dict[str, Dict[str, float]] = {
    "SURVIVE": {"base": 1.4, "avionics": 0.0, "sensors": 0.1, "comms": 0.1},
    "IDLE":    {"base": 1.8, "avionics": 0.1, "sensors": 0.1, "comms": 0.0},
    "SCIENCE": {"base": 4.5, "avionics": 0.4, "sensors": 0.45, "comms": 0.0},
    "UHF_TX":  {"base": 2.0, "avionics": 0.0, "sensors": 0.2, "comms": 3.0},
    "LORA":    {"base": 2.0, "avionics": 0.0, "sensors": 0.1, "comms": 2.0},
    "CAM":     {"base": 6.0, "avionics": 0.1, "sensors": 0.5, "comms": 0.0},
}
MODE_SCHEDULE: List[Tuple[str, float]] = [
    ("SURVIVE", 180), ("SCIENCE", 60), ("CAM", 12), ("IDLE", 40),
    ("LORA", 10), ("SURVIVE", 125), ("UHF_TX", 6),
    # tot = 443
]

# ============================================================
# =================== CONSTANTS / HELPERS ====================
# ============================================================
SB_SIGMA = 5.670374419e-8
MARS_SOL_S = 88775.0
MARS_DAY_HOURS = 24.6597
DEG2RAD = math.pi/180.0
rng = np.random.default_rng(RANDOM_SEED)

# Mars standard atmos (coarse): scale height H≈10.8 km, p0≈610 Pa (0 elev)
MARS_P0_PA = 610.0
MARS_H_M   = 10800.0
R_CO2      = 188.9     # J/(kg·K) specific gas constant for CO₂
MU0_CO2    = 1.37e-5   # Pa·s at ~300 K
T0_CO2     = 300.0
S_CO2      = 240.0     # Sutherland constant (approx)
K_CO2      = 0.010     # W/mK (order of magnitude at ~200 K)
PR_CO2     = 0.71

# ============================================================
# ======================== DATA CLASSES ======================
# ============================================================
@dataclass
class PanelConfig:
    eff: float = PANEL_EFF
    mppt_eff: float = MPPT_EFF

@dataclass
class BatteryConfig:
    cap_Wh: float = BATT_CAP_WH
    v_nom: float = BATT_V_NOM
    eta_chg: float = BATT_CHG_EFF
    eta_dchg: float = BATT_DCHG_EFF
    min_soc: float = BATT_MIN_SOC
    def cap_J(self) -> float: return self.cap_Wh * 3600.0 

@dataclass
class SingleUseBatteryConfig:
    cap_Wh: float = SINGLE_USE_CAP_WH
    max_W: float  = SINGLE_USE_MAX_W
    def cap_J(self) -> float: return self.cap_Wh * 3600.0

@dataclass
class ThermalConfig:
    emissivity_skin: float = EMISSIVITY_SKIN
    absorptivity_skin: float = ABSORPTIVITY_SKIN
    area_out_m2: float = PROBE_OUTER_AREA_M2
    area_cond_m2: float = CROSS_AREA_COND_M2
    cork_k: float = CORK_K_W_MK
    cork_d: float = CORK_THICKNESS_M
    c_skin: float = C_THERM_SKIN_JK
    c_int: float = C_THERM_INTERNAL_JK
    c_batt: float = C_THERM_BATT_JK
    # MLI configuration
    enable_mli_internal: bool = ENABLE_MLI_INTERNAL
    enable_mli_batt: bool = ENABLE_MLI_BATT
    mli_k: float = MLI_K_W_MK
    mli_internal_d: float = MLI_INTERNAL_THICK_M
    mli_batt_d: float = MLI_BATT_THICK_M

@dataclass
class EnvConfig:
    lat_deg: float = LAT_DEG
    lon_deg: float = LON_DEG
    elev_m: float = ELEV_M
    tau: float = TAU_DUST
    albedo: float = ALBEDO_GROUND
    emissivity_sky: float = EMISSIVITY_SKY
    t_mean_K: float = MARS_MEAN_AIR_TEMP_K
    daily_swing_K: float = DAILY_TEMP_SWING_K
    wind_mps: float = WIND_MPS
    solar_const: float = SOLAR_CONSTANT_MARS
    Ls_deg: float = Ls_DEG

# ============================================================
# ===================== ENVIRONMENT MODELS ===================
# ============================================================
def mars_pressure_pa(alt_m: float) -> float:
    return MARS_P0_PA * math.exp(-max(0.0, alt_m)/MARS_H_M)

def co2_dynamic_viscosity_Pas(T: float) -> float:
    return MU0_CO2 * ((T/T0_CO2)**1.5) * (T0_CO2 + S_CO2)/(T + S_CO2)

# ---- Solar geometry
def solar_declination_deg(Ls_deg: float) -> float:
    return 25.19 * math.sin(Ls_deg*DEG2RAD)

def solar_hour_angle_deg(t: float) -> float:
    hours = (t % MARS_SOL_S) * (MARS_DAY_HOURS / (MARS_SOL_S))
    return (hours - MARS_DAY_HOURS/2.0) * (360.0 / MARS_DAY_HOURS)

def sun_elevation_deg(t: float, env: EnvConfig) -> float:
    dec = solar_declination_deg(env.Ls_deg)
    ha  = solar_hour_angle_deg(t)
    lat = env.lat_deg
    lat_r, dec_r, ha_r = map(lambda x: x*DEG2RAD, (lat, dec, ha))
    sin_el = math.sin(lat_r)*math.sin(dec_r) + math.cos(lat_r)*math.cos(dec_r)*math.cos(ha_r)
    return max(-90.0, min(90.0, math.degrees(math.asin(sin_el))))

# ---- Air temperature diurnal
def mars_air_temp_K(t: float, env: EnvConfig) -> float:
    if not ENABLE_SOLAR_DIURNAL:
        return env.t_mean_K
    hours = (t % MARS_SOL_S) * (MARS_DAY_HOURS / (MARS_SOL_S))
    phase = 2*math.pi*(hours/MARS_DAY_HOURS)
    return env.t_mean_K + 0.5*env.daily_swing_K * math.cos(phase - 0.4*math.pi)

# ---- Irradiance at surface
def mars_irradiance_Wm2(t: float, env: EnvConfig) -> Tuple[float, float, float]:
    el = sun_elevation_deg(t, env)
    if el <= 0.0:
        return 0.0, 0.0, 0.0
    mu = max(1e-3, math.sin(el*DEG2RAD))
    tau = max(0.0, env.tau + rng.normal(0.0, TAU_JITTER_DEFAULT))
    direct = env.solar_const * mu * math.exp(-tau/mu)
    diffuse = 0.12 * env.solar_const * (1 - math.exp(-tau))
    albedo  = env.albedo * env.solar_const * mu * (1 - math.exp(-tau))
    return max(0.0, direct), max(0.0, diffuse), max(0.0, albedo)

# ============================================================
# ====================== GEOMETRY GAINS ======================
# ============================================================
def geometry_gain(el_deg: float) -> Tuple[float, str]:
    """
    Return (effective_area_m2, label) per geometry mode.

    Modes:
      - cone         : side behaves like tilted facets + diffuse bonus
      - cylinder     : side bands -> average 2/pi * sin(el)
      - hybrid       : top disk + side band
      - foldout      : hybrid + N tilted fold-out panels
      - foldout_top  : top disk + N tilted fold-out panels (no side wrap)
    """
    el = max(0.0, el_deg)
    proj = math.sin(el*DEG2RAD)
    mode = GEOMETRY_MODE.lower()

    if mode == "cone":
        side = SIDE_PANEL_AREA_M2
        eff_area = side * (0.6*proj + CONE_GAIN_BONUS)
        return max(0.0, eff_area), "cone"

    if mode == "cylinder":
        side = SIDE_PANEL_AREA_M2 * max(1, CYLINDER_BANDS)
        eff_area = side * (2.0/math.pi) * proj
        return max(0.0, eff_area), "cylinder"

    # base "hybrid": top + side band
    top = TOP_PANEL_AREA_M2 * proj
    side = SIDE_PANEL_AREA_M2 * (2.0/math.pi) * proj
    base = max(0.0, top + side)

    if mode == "foldout":
        if FOLDOUT_DEPLOYED and FOLDOUT_COUNT > 0 and FOLDOUT_PANEL_AREA_M2 > 0.0:
            proj_fold = math.sin(max(0.0, el + FOLDOUT_TILT_DEG) * DEG2RAD)
            fold_area = FOLDOUT_COUNT * FOLDOUT_PANEL_AREA_M2 * max(0.0, proj_fold)
            return base + fold_area, "foldout"
        return base, "foldout"

    if mode == "foldout_top":
        # only top disk + foldouts; ignore side wrap
        base_top = max(0.0, top)
        if FOLDOUT_DEPLOYED and FOLDOUT_COUNT > 0 and FOLDOUT_PANEL_AREA_M2 > 0.0:
            proj_fold = math.sin(max(0.0, el + FOLDOUT_TILT_DEG) * DEG2RAD)
            fold_area = FOLDOUT_COUNT * FOLDOUT_PANEL_AREA_M2 * max(0.0, proj_fold)
            return base_top + fold_area, "foldout_top"
        return base_top, "foldout_top"

    return base, "hybrid"

# ============================================================
# =========== PANEL TEMP / EFFICIENCY DERATING ===============
# ============================================================
def panel_cell_temp_C(irr_Wm2: float, T_air_C: float, wind_mps: float) -> float:
    wind_factor = 1.0/(1.0 + 0.02*max(0.0, wind_mps-1.0))
    return T_air_C + (NOCT_C - 20.0) * (irr_Wm2/NOCT_REF_IRR) * wind_factor

def eff_temp_derated(eff_STC: float, cell_T_C: float) -> float:
    dT = cell_T_C - 25.0
    return max(0.0, eff_STC * (1.0 + (TEMP_COEFF_PERCpC/100.0)*dT))

# ============================================================
# ====================== POWER MODELS ========================
# ============================================================
def power_breakdown_W(mode: str) -> Dict[str, float]:
    d = POWER_MODES.get(mode, {})
    return {k: float(d.get(k, 0.0)) for k in ("base","avionics","sensors","comms")}

@dataclass
class DustState:
    eff_mult: float = 1.0   # multiplicative hit to panel efficiency (≤1)
    in_storm: bool = False
    last_sol_idx: int = -1

def update_dust_state(ds: DustState, t_s: float, env: EnvConfig) -> DustState:
    sol_idx = int(t_s // MARS_SOL_S)
    if sol_idx == ds.last_sol_idx:
        return ds
    ds.last_sol_idx = sol_idx
    if ENABLE_STORMS_DEFAULT or ENABLE_DUST_STORM:
        if ds.in_storm:
            if rng.random() < P_EXIT_DEFAULT:
                ds.in_storm = False
                env.tau = max(0.1, TAU0_DEFAULT + rng.normal(0.0, TAU_JITTER_DEFAULT))
            else:
                env.tau = STORM_TAU_DEFAULT + rng.normal(0.0, 0.1)
        else:
            if rng.random() < P_ENTER_DEFAULT:
                ds.in_storm = True
                env.tau = STORM_TAU_DEFAULT + rng.normal(0.0, 0.1)
            else:
                env.tau = TAU0_DEFAULT + rng.normal(0.0, TAU_JITTER_DEFAULT)
    if ds.in_storm:
        ds.eff_mult *= (1.0 - 0.5*K_DUST_DEFAULT)
    else:
        ds.eff_mult *= (1.0 - K_DUST_DEFAULT)
    if rng.random() < CLEAN_PROB_DEFAULT:
        ds.eff_mult = max(EFF_RESET_DEFAULT, ds.eff_mult)
    ds.eff_mult = float(np.clip(ds.eff_mult, 0.5, 1.0))
    return ds

def panel_power_W(t: float, panel: PanelConfig, env: EnvConfig, ds: DustState) -> Tuple[float, Dict[str, float]]:
    el = sun_elevation_deg(t, env)
    direct, diffuse, albedo = mars_irradiance_Wm2(t, env)
    area_eff, _ = geometry_gain(el)
    irr_total = direct + 0.35*diffuse + 0.5*albedo  # tilt-agnostic shares
    T_air_C = mars_air_temp_K(t, env) - 273.15
    Tcell_C = panel_cell_temp_C(irr_total, T_air_C, env.wind_mps)
    eff_use = eff_temp_derated(panel.eff, Tcell_C) * ds.eff_mult
    p_out = irr_total * area_eff * eff_use * panel.mppt_eff
    return max(0.0, p_out), {
        "sun_el_deg": el,
        "irr_direct": direct,
        "irr_diffuse": diffuse,
        "irr_albedo": albedo,
        "irr_total": irr_total,
        "area_eff": area_eff,
        "cell_T_C": Tcell_C,
        "eff_use": eff_use,
    }

# ============================================================
# ===================== THERMAL NETWORK ======================
# ============================================================
@dataclass
class ThermalState:
    T_skin: float
    T_int: float
    T_batt: float
    heater_on: bool

@dataclass
class ThermalOutputs:
    q_rad_out_W: float
    q_cond_skin_int_W: float
    q_cond_int_batt_W: float
    q_solar_abs_W: float
    q_conv_W: float
    heater_W: float

class HeaterController:
    def __init__(self, setpoint_K=HEATER_SETPOINT_BATT_K, hysteresis_K=HEATER_HYSTERESIS_K):
        self.setpoint = setpoint_K
        self.hyst = hysteresis_K
        self._state = False
    def step(self, T_batt_K: float) -> bool:
        if self._state and T_batt_K >= self.setpoint + self.hyst:
            self._state = False
        elif (not self._state) and T_batt_K <= self.setpoint - self.hyst:
            self._state = True
        return self._state

def skin_sky_radiative_exchange_W(T_skin: float, T_air: float, area: float, eps_skin: float, eps_sky: float) -> float:
    T_sky = (eps_sky**0.25) * T_air
    return eps_skin * SB_SIGMA * area * (T_skin**4 - T_sky**4)

def forced_convection_W(T_surface: float, T_air: float, V: float, L: float, area: float) -> float:
    Tfilm = 0.5*(T_surface + T_air)
    mu = co2_dynamic_viscosity_Pas(Tfilm)
    p = mars_pressure_pa(ELEV_M)
    rho = p/(R_CO2*Tfilm)
    k = K_CO2
    Re = max(1.0, rho*V*L/max(1e-6, mu))
    Pr = PR_CO2
    Nu = 0.664*(Re**0.5)*(Pr**(1.0/3.0))
    h = Nu * k / max(1e-3, L)
    return h * area * (T_surface - T_air)

def thermal_step(dt_s: float, t_s: float, state: ThermalState, therm: ThermalConfig,
                 env: EnvConfig, panel_W: float, internal_diss_W: float,
                 heater_ctrl: HeaterController) -> Tuple[ThermalState, ThermalOutputs]:
    T_air = mars_air_temp_K(t_s, env)
    q_rad_out = skin_sky_radiative_exchange_W(state.T_skin, T_air, therm.area_out_m2, therm.emissivity_skin, env.emissivity_sky)
    direct, diffuse, albedo = mars_irradiance_Wm2(t_s, env)
    q_solar_abs = therm.absorptivity_skin * (direct + diffuse + albedo) * therm.area_out_m2

    q_conv = 0.0
    if ENABLE_CONVECTION_LOSS:
        q_conv = forced_convection_W(state.T_skin, T_air, env.wind_mps, CHAR_LEN_M, therm.area_out_m2)

    # Conduction: outer skin → interior through cork (+ optional internal MLI in series)
    R_cork = therm.cork_d / max(1e-9, therm.cork_k * therm.area_cond_m2)
    R_mli_int = (therm.mli_internal_d / max(1e-9, therm.mli_k * therm.area_cond_m2)) if therm.enable_mli_internal else 0.0
    R_skin_int = R_cork + R_mli_int
    q_cond_skin_int = (state.T_skin - state.T_int) / max(1e-6, R_skin_int)

    heater_on = heater_ctrl.step(state.T_batt) if ENABLE_HEATER else False
    heater_W = HEATER_MAX_W if heater_on else 0.0
    q_heater_int = heater_W * HEATER_INTERNAL_BIAS
    q_heater_batt = heater_W * (1.0 - HEATER_INTERNAL_BIAS)

    # Conduction: interior ↔ battery (strap in series with optional battery MLI)
    k_strap = 0.6  # W/K baseline
    R_strap = 1.0 / max(1e-6, k_strap)
    R_mli_batt = (therm.mli_batt_d / max(1e-9, therm.mli_k * therm.area_cond_m2)) if therm.enable_mli_batt else 0.0
    R_int_batt = R_strap + R_mli_batt
    q_cond_int_batt = (state.T_int - state.T_batt) / max(1e-6, R_int_batt)

    # Temperature rates
    dT_skin = ( - q_rad_out - q_conv - q_cond_skin_int + q_solar_abs ) * dt_s / therm.c_skin
    dT_int  = ( + q_cond_skin_int - q_cond_int_batt + q_heater_int + 0.8*internal_diss_W ) * dt_s / therm.c_int
    dT_batt = ( + q_cond_int_batt + q_heater_batt + 0.2*internal_diss_W ) * dt_s / therm.c_batt

    new = ThermalState(
        T_skin = state.T_skin + dT_skin,
        T_int  = state.T_int  + dT_int,
        T_batt = state.T_batt + dT_batt,
        heater_on = heater_on,
    )
    out = ThermalOutputs(
        q_rad_out_W = q_rad_out,
        q_cond_skin_int_W = q_cond_skin_int,
        q_cond_int_batt_W = q_cond_int_batt,
        q_solar_abs_W = q_solar_abs,
        q_conv_W = q_conv,
        heater_W = heater_W,
    )
    return new, out

# ============================================================
# ================ POWER FLOW / BATTERY SOC ==================
# ============================================================
@dataclass
class PowerState:
    soc: float
    energy_J: float
    single_use_J: float = SINGLE_USE_CAP_WH * 3600.0  # start full if enabled

@dataclass
class PowerOutputs:
    mode: str
    load_W: float
    panel_W: float
    heater_W: float
    net_W: float            # panel - load (legacy)
    single_use_W: float     # extra supplied by one-shot battery (≥0)
    net_W_total: float      # panel + single_use - load (true electrical balance)
    capped: bool

def mode_load_W(mode: str) -> float:
    return float(sum(POWER_MODES.get(mode, {}).values()))

def power_step(dt_s: float, mode: str, panel_W: float, heater_W: float,
               batt: BatteryConfig, su: SingleUseBatteryConfig,
               state: PowerState) -> Tuple[PowerState, PowerOutputs]:
    base_load_W = mode_load_W(mode)
    total_load_W = base_load_W + heater_W

    capped = False
    if ENABLE_POWER_CAPPING and state.soc <= batt.min_soc and (panel_W < total_load_W):
        d = POWER_MODES.get(mode, {}).copy()
        base = d.pop("base", 0.0)
        essentials = base + d.get("avionics", 0.0)
        nonessential = sum(v for k,v in d.items() if k != "avionics")
        reduced = nonessential * 0.5
        total_load_W = essentials + reduced + heater_W
        capped = True

    net_W_primary = panel_W - total_load_W
    single_use_W = 0.0

    if net_W_primary >= 0:
        # Charge primary (with efficiency)
        state.energy_J = min(state.energy_J + net_W_primary * dt_s * batt.eta_chg, batt.cap_J())
    else:
        # Energy deficit
        J_need = -net_W_primary * dt_s
        # Use primary down to min SoC (discharge efficiency)
        E_min = batt.cap_J() * batt.min_soc
        if state.energy_J > E_min:
            J_from_primary = min(J_need / max(1e-6, batt.eta_dchg), state.energy_J - E_min)
            state.energy_J -= J_from_primary
            J_need -= J_from_primary * max(1e-6, batt.eta_dchg)
        # Use single-use if enabled
        if ENABLE_SINGLE_USE_BATT and J_need > 1e-12 and state.single_use_J > 0.0:
            W_cap = su.max_W if su.max_W is not None else float("inf")
            J_cap = W_cap * dt_s
            J_from_su = min(J_need, state.single_use_J, J_cap)
            state.single_use_J -= J_from_su
            single_use_W = J_from_su / dt_s
            J_need -= J_from_su
        # Any remaining J_need is unmet (brownout), reflected in net_W_total

    state.energy_J = float(np.clip(state.energy_J, batt.cap_J()*batt.min_soc, batt.cap_J()))
    state.soc = state.energy_J / batt.cap_J()
    net_W_total = panel_W + single_use_W - total_load_W

    po = PowerOutputs(
        mode=mode, load_W=total_load_W, panel_W=panel_W,
        heater_W=heater_W, net_W=panel_W - total_load_W,
        single_use_W=single_use_W, net_W_total=net_W_total, capped=capped
    )
    return state, po

# ============================================================
# ===================== MODE SCHEDULER =======================
# ============================================================
def build_mode_timeline(schedule: List[Tuple[str, float]], total_seconds: float, dt_s: float) -> List[str]:
    seq: List[str] = []
    i = 0
    t_left = 0.0
    mode = schedule[0][0]
    while len(seq) * dt_s < total_seconds:
        if t_left <= 0.0:
            mode, minutes = schedule[i % len(schedule)]
            t_left = minutes * 60.0
            i += 1
        seq.append(mode)
        t_left -= dt_s
    return seq

# ============================================================
# ===================== MAIN SIMULATION ======================
# ============================================================
def run_sim(sim_hours: float, dt_s: float,
            panel: PanelConfig, batt: BatteryConfig,
            therm: ThermalConfig, env: EnvConfig,
            init_skin_K=240.0, init_int_K=245.0, init_batt_K=248.0,
            start_soc=0.80) -> Tuple[pd.DataFrame, Dict]:

    steps = int(round((sim_hours*3600.0)/dt_s))
    timeline = build_mode_timeline(MODE_SCHEDULE, sim_hours*3600.0, dt_s)

    pstate = PowerState(soc=start_soc, energy_J=start_soc*batt.cap_J(),
                        single_use_J=SINGLE_USE_CAP_WH*3600.0 if ENABLE_SINGLE_USE_BATT else 0.0)
    sucfg = SingleUseBatteryConfig()
    tstate = ThermalState(T_skin=init_skin_K, T_int=init_int_K, T_batt=init_batt_K, heater_on=False)
    hctrl = HeaterController()
    dstate = DustState()

    rows = []
    heater_energy_J = 0.0
    capped_count = 0
    cum_E_harv_J = 0.0
    cum_E_load_J = 0.0
    cum_E_heater_J = 0.0
    cum_E_singleuse_J = 0.0

    peak_panel_W = 0.0

    # Print insulation resistances at start for verification
    def _series_R(d, k, A): return d / max(1e-9, k*A)
    R_cork = _series_R(therm.cork_d, therm.cork_k, therm.area_cond_m2)
    R_mli_int = _series_R(therm.mli_internal_d, therm.mli_k, therm.area_cond_m2) if therm.enable_mli_internal else 0.0
    R_int_total = R_cork + R_mli_int
    R_strap = 1.0 / max(1e-6, 0.6)
    R_mli_b = _series_R(therm.mli_batt_d, therm.mli_k, therm.area_cond_m2) if therm.enable_mli_batt else 0.0
    R_batt_link = R_strap + R_mli_b
    print(f"[insul] R_skin→int = {R_int_total:.4f} K/W   (cork {R_cork:.4f}, mli {R_mli_int:.4f})")
    print(f"[insul] R_int↔batt = {R_batt_link:.4f} K/W   (strap {R_strap:.4f}, mli {R_mli_b:.4f})")

    for k in range(steps):
        t_s = k*dt_s
        mode = timeline[k]

        dstate = update_dust_state(dstate, t_s, env)

        p_panel, pmeta = panel_power_W(t_s, panel, env, dstate)
        brk = power_breakdown_W(mode)
        total_wo_heater_W = brk["base"] + brk["avionics"] + brk["sensors"] + brk["comms"]

        internal_diss = 0.7 * total_wo_heater_W

        tstate, tout = thermal_step(dt_s, t_s, tstate, therm, env, p_panel, internal_diss, hctrl)
        pstate, pout = power_step(dt_s, mode, p_panel, tout.heater_W, batt, sucfg, pstate)

        if pout.capped: capped_count += 1
        heater_energy_J += tout.heater_W * dt_s

        cum_E_harv_J += max(0.0, p_panel) * dt_s
        cum_E_load_J += max(0.0, total_wo_heater_W) * dt_s
        cum_E_heater_J += max(0.0, tout.heater_W) * dt_s
        cum_E_singleuse_J += max(0.0, pout.single_use_W) * dt_s

        peak_panel_W = max(peak_panel_W, p_panel)

        rows.append({
            "t_s": t_s,
            "sol": t_s/MARS_SOL_S,
            "mode": mode,
            "panel_W": p_panel,
            "single_use_W": pout.single_use_W,
            "load_W": pout.load_W,
            "heater_W": tout.heater_W,
            "net_W": pout.net_W,
            "net_W_total": pout.net_W_total,
            "soc": pstate.soc,
            "singleuse_soc": (pstate.single_use_J / sucfg.cap_J()) if (ENABLE_SINGLE_USE_BATT and sucfg.cap_J() > 0) else None,
            "base_W": brk["base"],
            "avionics_W": brk["avionics"],
            "sensors_W": brk["sensors"],
            "comms_W": brk["comms"],
            "load_no_heater_W": total_wo_heater_W,
            "E_harv_J": cum_E_harv_J,
            "E_load_J": cum_E_load_J,
            "E_heater_J": cum_E_heater_J,
            "E_singleuse_J": cum_E_singleuse_J,
            "T_air_K": mars_air_temp_K(t_s, env),
            "T_skin_K": tstate.T_skin,
            "T_int_K": tstate.T_int,
            "T_batt_K": tstate.T_batt,
            "heater_on": int(tstate.heater_on),
            "q_rad_out_W": tout.q_rad_out_W,
            "q_conv_W": tout.q_conv_W,
            "q_solar_abs_W": tout.q_solar_abs_W,
            "q_cond_skin_int_W": tout.q_cond_skin_int_W,
            "q_cond_int_batt_W": tout.q_cond_int_batt_W,
            # meta for verification/plots
            "sun_el_deg": pmeta["sun_el_deg"],
            "irr_direct": pmeta["irr_direct"],
            "irr_diffuse": pmeta["irr_diffuse"],
            "irr_albedo": pmeta["irr_albedo"],
            "irr_total": pmeta["irr_total"],
            "panel_area_eff_m2": pmeta["area_eff"],
            "panel_cell_T_C": pmeta["cell_T_C"],
            "panel_eff_used": pmeta["eff_use"],
            "dust_eff_mult": dstate.eff_mult,
            "storm": int(dstate.in_storm),
            "tau": env.tau,
            "capped": int(pout.capped),
        })

    df = pd.DataFrame(rows)

    duty = df["heater_on"].mean() if len(df) else 0.0
    heater_Wh = heater_energy_J/3600.0
    capped_pct = 100.0*(capped_count/len(df)) if len(df) else 0.0

    mode_dwell_h = df.groupby("mode")["t_s"].count() * (dt_s/3600.0)
    mode_energy_Wh = (df.groupby("mode")["load_no_heater_W"].mean() * mode_dwell_h).to_dict()

    summary = {
        "sim": {"duration_h": sim_hours, "dt_s": dt_s, "steps": steps},
        "env": asdict(env),
        "panel": asdict(panel),
        "battery": asdict(batt),
        "single_use": {
            "enabled": ENABLE_SINGLE_USE_BATT,
            "cap_Wh": SINGLE_USE_CAP_WH,
            "max_W": SINGLE_USE_MAX_W
        },
        "thermal": asdict(therm),
        "geometry": {
            "mode": GEOMETRY_MODE,
            "top_area_m2": TOP_PANEL_AREA_M2,
            "side_area_m2": SIDE_PANEL_AREA_M2,
            "cylinder_bands": CYLINDER_BANDS,
            "char_len_m": CHAR_LEN_M,
            "foldout": {
                "deployed": FOLDOUT_DEPLOYED,
                "count": FOLDOUT_COUNT,
                "panel_area_m2": FOLDOUT_PANEL_AREA_M2,
                "tilt_deg": FOLDOUT_TILT_DEG,
            }
        },
        "results": {
            "final_soc": float(df["soc"].iloc[-1]) if len(df) else None,
            "min_soc": float(df["soc"].min()) if len(df) else None,
            "heater_duty": duty,
            "heater_energy_Wh": heater_Wh,
            "brownout_capped_pct": capped_pct,
            "min_T_batt_K": float(df["T_batt_K"].min()) if len(df) else None,
            "min_T_int_K": float(df["T_int_K"].min()) if len(df) else None,
            "min_T_skin_K": float(df["T_skin_K"].min()) if len(df) else None,
            "harvested_Wh": float(df["E_harv_J"].iloc[-1]/3600.0) if len(df) else 0.0,
            "consumed_Wh_no_heater": float(df["E_load_J"].iloc[-1]/3600.0) if len(df) else 0.0,
            "heater_Wh_cum": float(df["E_heater_J"].iloc[-1]/3600.0) if len(df) else 0.0,
            "single_use_Wh_used": float(df["E_singleuse_J"].iloc[-1]/3600.0) if len(df) else 0.0,
            "single_use_Wh_remaining": float(pstate.single_use_J/3600.0),
            "mode_energy_Wh": mode_energy_Wh,
            "peak_panel_W": float(peak_panel_W),
            "storm_time_h": float(df["storm"].sum()*dt_s/3600.0) if "storm" in df else 0.0,
            "avg_tau": float(df["tau"].mean()) if "tau" in df else env.tau,
        }
    }

    # Lint checks
    _lint_messages: List[str] = []
    night = df["sun_el_deg"] <= 0.0
    if len(df[night]) > 0 and (df.loc[night, "panel_W"].max() > 1e-2):
        _lint_messages.append("Nonzero panel_W at night; check geometry/irradiance.")
    if not np.isfinite(df[["panel_W","soc","T_batt_K","T_int_K","T_skin_K"]].to_numpy()).all():
        _lint_messages.append("Found non-finite values in core timeseries.")
    if not ENABLE_HEATER and duty > 0:
        _lint_messages.append("Heater produced energy with ENABLE_HEATER=False.")
    summary["lint"] = _lint_messages

    return df, summary

# ============================================================
# =========================== PLOTS ==========================
# ============================================================
def plot_timeseries(df: pd.DataFrame, outdir: str, show: bool, save: bool):
    if df.empty: return
    t_h = df["t_s"]/3600.0

    # Power & SOC
    plt.figure(figsize=(11,4.8))
    plt.title("Power Flow & SOC")
    plt.plot(t_h, df["panel_W"], label="Panel W")
    if "single_use_W" in df: plt.plot(t_h, df["single_use_W"], label="Single-use W")
    plt.plot(t_h, df["load_W"], label="Total Load W (incl heater)")
    plt.plot(t_h, df.get("net_W_total", df["net_W"]), label="Net W (incl single-use)")
    plt.plot(t_h, df["soc"], label="SOC (0..1)")
    plt.xlabel("Time [h]"); plt.grid(True, alpha=0.3); plt.legend()
    if save: plt.savefig(os.path.join(outdir, "power_soc.png"), dpi=160, bbox_inches='tight')
    plt.show() if show else plt.close()

    # Subsystem stacked
    plt.figure(figsize=(11,4.8))
    plt.title("Subsystem Loads (stacked)")
    plt.stackplot(t_h, df["base_W"], df["avionics_W"], df["sensors_W"], df["comms_W"], labels=["base","avionics","sensors","comms"])
    plt.plot(t_h, df["heater_W"], lw=1.2, label="Heater W")
    plt.xlabel("Time [h]"); plt.grid(True, alpha=0.25); plt.legend(loc="upper right")
    if save: plt.savefig(os.path.join(outdir, "subsystems_stacked.png"), dpi=160, bbox_inches='tight')
    plt.show() if show else plt.close()

    # Mode timeline
    modes = {m:i for i,m in enumerate(sorted(df["mode"].unique()))}
    y = df["mode"].map(modes)
    plt.figure(figsize=(11,2.8))
    plt.title("Mode Timeline")
    plt.step(t_h, y, where='post')
    plt.yticks(list(modes.values()), list(modes.keys()))
    plt.xlabel("Time [h]"); plt.grid(True, axis='x', alpha=0.3)
    if save: plt.savefig(os.path.join(outdir, "mode_timeline.png"), dpi=160, bbox_inches='tight')
    plt.show() if show else plt.close()

    # Cumulative energy
    plt.figure(figsize=(11,4.8))
    plt.title("Cumulative Energy Budget")
    plt.plot(t_h, df["E_harv_J"]/3600.0, label="Harvested Wh")
    plt.plot(t_h, df["E_load_J"]/3600.0, label="Consumed Wh (no heater)")
    plt.plot(t_h, df["E_heater_J"]/3600.0, label="Heater Wh")
    if "E_singleuse_J" in df: plt.plot(t_h, df["E_singleuse_J"]/3600.0, label="Single-use Wh")
    plt.xlabel("Time [h]"); plt.grid(True, alpha=0.3); plt.legend()
    if save: plt.savefig(os.path.join(outdir, "energy_cumulative.png"), dpi=160, bbox_inches='tight')
    plt.show() if show else plt.close()

    # Temperatures
    plt.figure(figsize=(11,4.8))
    plt.title("Temperatures")
    plt.plot(t_h, df["T_air_K"]-273.15, label="Air °C")
    plt.plot(t_h, df["T_skin_K"]-273.15, label="Skin °C")
    plt.plot(t_h, df["T_int_K"]-273.15, label="Internal °C")
    plt.plot(t_h, df["T_batt_K"]-273.15, label="Battery °C")
    plt.xlabel("Time [h]"); plt.grid(True, alpha=0.3); plt.legend()
    if save: plt.savefig(os.path.join(outdir, "temps.png"), dpi=160, bbox_inches='tight')
    plt.show() if show else plt.close()

    # Heater & brownout & storms
    plt.figure(figsize=(11,4.0))
    plt.title("Heater / Brownout / Storms")
    plt.plot(t_h, df["heater_on"], label="Heater On (0/1)")
    plt.plot(t_h, df["capped"], label="Capped (0/1)")
    if "storm" in df: plt.plot(t_h, df["storm"], label="Storm (0/1)")
    plt.xlabel("Time [h]"); plt.grid(True, alpha=0.3); plt.legend()
    if save: plt.savefig(os.path.join(outdir, "heater_brownout.png"), dpi=160, bbox_inches='tight')
    plt.show() if show else plt.close()

# ---- v0.2-style adapter plots (from new df)
def _storm_spans(ax, t: np.ndarray, storm_flags: np.ndarray):
    if storm_flags is None or not storm_flags.any(): return
    in_reg = False; start = None
    for k, flag in enumerate(storm_flags):
        if flag and not in_reg: in_reg = True; start = t[k]
        if in_reg and (k == len(storm_flags)-1 or not storm_flags[k+1]):
            stop = t[k]
            ax.axvspan(start, stop, alpha=0.12, color="gray", label="Dust storm"); in_reg = False

def plot_environment_style_v02(df: pd.DataFrame, outdir: str, show: bool, save: bool):
    if df.empty: return
    stride = max(1, int(len(df)/2000)); df_hr = df.iloc[::stride]
    t = df_hr["sol"].to_numpy()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t, df_hr["T_air_K"]-273.15, label="Surface/air")
    ax.plot(t, df_hr["T_int_K"]-273.15, label="Internal")
    ax.plot(t, df_hr["T_batt_K"]-273.15, label="Battery")
    ax.set_xlabel("Sol"); ax.set_ylabel("Temp (°C)"); ax.set_title("Temperature profile")
    ax.grid(True); ax.legend(); fig.tight_layout()
    if save: fig.savefig(os.path.join(outdir, "temp_profile_v02.png"), dpi=150)
    if show: plt.show(); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(t, df_hr["panel_W"], label="Panel W")
    ax.plot(t, df_hr["soc"]*100.0, label="Battery SoC (%)")
    ax.set_xlabel("Sol"); ax.set_ylabel("Power (W) / SoC (%)"); ax.set_title("Panel Output and Battery State")
    ax.grid(True); ax.legend(); fig.tight_layout()
    if save: fig.savefig(os.path.join(outdir, "power_battery_v02.png"), dpi=150)
    if show: plt.show(); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,4))
    storms = df_hr["storm"].to_numpy(dtype=bool) if "storm" in df_hr else np.zeros_like(t, dtype=bool)
    tau_series = df_hr["tau"].to_numpy() if "tau" in df_hr else np.full_like(t, TAU_DUST)
    ax.plot(t, tau_series*100.0, label="Tau ×100")
    _storm_spans(ax, t, storms)
    ax.set_xlabel("Sol"); ax.set_ylabel("Tau ×100"); ax.set_title("Dust / Storm Proxy")
    ax.grid(True); ax.legend(); fig.tight_layout()
    if save: fig.savefig(os.path.join(outdir, "dust_eff_v02.png"), dpi=150)
    if show: plt.show(); plt.close(fig)

# ---- Seasonal daily curves & totals (v0.2-style, SoC physics-fixed)
def daily_curves_and_tables_v02(outdir: str, show: bool, save: bool,
                                latitudes: List[int] = [90,45,0,-45,-90],
                                seasons: List[str] = ["Spring","Summer","Autumn","Winter"],
                                selected_season: str = "Autumn"):
    TIME_HOURS = np.linspace(0.0, MARS_DAY_HOURS, 100)

    def season_to_Ls(season: str) -> float:
        return {"Spring": 0.0, "Summer": 90.0, "Autumn": 180.0, "Winter": 270.0}.get(season, 0.0)

    # How many sols to "spin up" before plotting the last one
    N_EQUIV_SOLS = 5

    season_results = {s: {} for s in seasons}
    total_daily_energy = {s: {} for s in seasons}
    batt_pct_profiles = {}

    for lat in latitudes:
        for season in seasons:
            env_tmp = EnvConfig(
                lat_deg=lat,
                Ls_deg=season_to_Ls(season),
                tau=TAU_DUST,
                albedo=ALBEDO_GROUND,
                emissivity_sky=EMISSIVITY_SKY,
                t_mean_K=MARS_MEAN_AIR_TEMP_K,
                daily_swing_K=DAILY_TEMP_SWING_K,
                wind_mps=WIND_MPS,
                solar_const=SOLAR_CONSTANT_MARS,
            )

            power_trace: List[float] = []
            batt_pct: List[float] = []

            # Use a nominal capacity for these proxy curves (v0.2 legacy style)
            cap_nom = 120.0  # Wh
            dt = TIME_HOURS[1] - TIME_HOURS[0]  # hours

            # Start at full charge (physics-realistic initial condition)
            soc = cap_nom

            # Run several sols in a row; record only the last sol for plotting
            for sol_idx in range(N_EQUIV_SOLS):
                power_trace_day: List[float] = []
                batt_pct_day: List[float] = []

                for hr in TIME_HOURS:
                    t_s = hr * MARS_SOL_S / MARS_DAY_HOURS
                    direct, diffuse, albedo = mars_irradiance_Wm2(t_s, env_tmp)
                    area_eff, _ = geometry_gain(sun_elevation_deg(t_s, env_tmp))
                    irr_total = direct + 0.35*diffuse + 0.5*albedo
                    pwr = area_eff * PANEL_EFF * MPPT_EFF * irr_total

                    # Simple representative draw profile (v0.2-style)
                    if hr < 2.5:
                        draw = 6.0
                    elif hr < 5.5:
                        draw = 5.5
                    else:
                        draw = 2.9

                    # soc tracked in Wh
                    soc = np.clip(soc + (pwr - draw) * dt, 0.0, cap_nom)
                    power_trace_day.append(pwr)
                    batt_pct_day.append(100.0 * soc / cap_nom)

                # Only keep the last sol's profile for plots / tables
                if sol_idx == N_EQUIV_SOLS - 1:
                    power_trace = power_trace_day
                    batt_pct = batt_pct_day

            season_results[season][lat] = power_trace
            total_daily_energy[season][lat] = float(np.trapz(power_trace, TIME_HOURS))
            if season == selected_season:
                batt_pct_profiles[lat] = batt_pct

    # --- Plots (v0.2 look, unchanged externally) ---
    for season in seasons:
        fig, ax = plt.subplots(figsize=(9,5))
        for lat in latitudes:
            ax.plot(TIME_HOURS, season_results[season][lat], label=f"Lat {lat}")
        ax.set_title(f"Generated Power vs Sol Time – {season}")
        ax.set_xlabel("Time of Sol (h)")
        ax.set_ylabel("Power (W)")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        if save:
            fig.savefig(os.path.join(outdir, f"daily_power_{season.lower()}_v02.png"), dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9,5))
    for lat in latitudes:
        ax.plot(TIME_HOURS, batt_pct_profiles[lat], label=f"Lat {lat}")
    ax.set_title(f"Battery SoC vs Time – {selected_season}")
    ax.set_xlabel("Time of Sol (h)")
    ax.set_ylabel("SoC (%)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(outdir, f"soc_vs_time_{selected_season.lower()}_v02.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    width = 0.18
    x = np.arange(len(latitudes))
    for i, season in enumerate(seasons):
        vals = [total_daily_energy[season][lat] for lat in latitudes]
        ax.bar(x + i*width, vals, width, label=season)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels([str(l) for l in latitudes])
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Energy (Wh)")
    ax.set_title("Total Daily Generated Energy – All Seasons (v0.2 style)")
    ax.legend()
    ax.grid(axis='y')
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(outdir, "daily_energy_bar_v02.png"), dpi=150)
    if show:
        plt.show()
    plt.close(fig)

    return total_daily_energy

# ---- Long-trend plots (v0.2-style)
def plot_long_trends_v02(df: pd.DataFrame, outdir: str, show: bool, save: bool):
    if df.empty: return
    t = df["sol"].to_numpy(); soc_pct = df["soc"].to_numpy()*100.0; net_Wt = df.get("net_W_total", df["net_W"]).to_numpy()
    def finish(fig, fname):
        if save: fig.savefig(os.path.join(outdir, fname), dpi=160, bbox_inches='tight')
        if show: plt.show()
        plt.close(fig)
    mask28 = (t - t.min()) <= 28.0
    fig, ax = plt.subplots(figsize=(11,5)); ax.plot(t[mask28], pd.Series(soc_pct).rolling(101, min_periods=1, center=True).mean()[mask28])
    ax.set_title("Battery SoC over 28 Sols (smoothed)"); ax.set_xlabel("Sol"); ax.set_ylabel("SoC (%)"); ax.grid(True); finish(fig, "long_SoC_28.png")
    fig, ax = plt.subplots(figsize=(11,5)); ax.plot(t[mask28], pd.Series(net_Wt).rolling(101, min_periods=1, center=True).mean()[mask28])
    ax.set_title("Net Power over 28 Sols (smoothed)"); ax.set_xlabel("Sol"); ax.set_ylabel("Net Power (W)"); ax.grid(True); finish(fig, "long_Net_28.png")
    mask7 = (t - t.min()) <= 7.0
    fig, ax = plt.subplots(figsize=(11,5)); ax.plot(t[mask7], soc_pct[mask7])
    ax.set_title("Battery SoC – First 7 Sols"); ax.set_xlabel("Sol"); ax.set_ylabel("SoC (%)"); ax.grid(True); finish(fig, "long_SoC_7.png")
    fig, ax = plt.subplots(figsize=(11,5)); ax.plot(t[mask7], net_Wt[mask7])
    ax.set_title("Net Power – First 7 Sols"); ax.set_xlabel("Sol"); ax.set_ylabel("Net Power (W)"); ax.grid(True); finish(fig, "long_Net_7.png")

# ---- Long-trend printed tables (v0.2-style)
def long_trend_tables_v02(latitudes: List[int], selected_season: str, outdir: str, show: bool, save: bool):
    hours_per_sol = int(round(MARS_DAY_HOURS)); dt = 1.0
    stat_death, stat_surv_first, stat_energy = {}, {}, {}
    stat_surv_entries, stat_emer_times = {}, {}
    def season_to_Ls(season: str) -> float:
        return {"Spring": 0.0, "Summer": 90.0, "Autumn": 180.0, "Winter": 270.0}.get(selected_season, 180.0)
    for lat in latitudes:
        cap = 120.0
        soc = cap
        hrs = 0.0
        in_surv = False
        last_exit = None
        surv_entries = []
        emer_times = []
        total_pwr = []
        env_tmp = EnvConfig(lat_deg=lat, Ls_deg=season_to_Ls(selected_season), tau=TAU_DUST,
                            albedo=ALBEDO_GROUND, emissivity_sky=EMISSIVITY_SKY,
                            t_mean_K=MARS_MEAN_AIR_TEMP_K, daily_swing_K=DAILY_TEMP_SWING_K,
                            wind_mps=WIND_MPS, solar_const=SOLAR_CONSTANT_MARS)
        for sol in range(28):
            for h in range(hours_per_sol):
                hr = h * (MARS_DAY_HOURS / hours_per_sol)
                t_s = hr * MARS_SOL_S / MARS_DAY_HOURS
                direct, diffuse, albedo = mars_irradiance_Wm2(t_s, env_tmp)
                area_eff, _ = geometry_gain(sun_elevation_deg(t_s, env_tmp))
                pwr = area_eff * PANEL_EFF * MPPT_EFF * (direct + 0.35*diffuse + 0.5*albedo)
                total_pwr.append(pwr)
                soc_pct = 100.0 * soc / cap
                if not in_surv and soc_pct <= 30.0:
                    in_surv = True
                    if lat not in stat_surv_first: stat_surv_first[lat] = hrs
                    surv_entries.append(hrs)
                    if last_exit is not None:
                        pass
                elif in_surv and soc_pct > 30.0:
                    in_surv = False; last_exit = hrs
                if soc_pct <= 5.0 and not emer_times:
                    soc = min(soc + 18.0, cap)
                    emer_times.append(hrs)
                draw = 5.7 if hr < 2.5 else 5.6 if hr < 5.5 else 2.87
                net = pwr - draw
                soc = np.clip(soc + net * dt, 0.0, cap)
                hrs += dt
                if soc <= 0 and lat not in stat_death:
                    stat_death[lat] = hrs
            if soc <= 0: break
        stat_death.setdefault(lat, hrs)
        stat_energy[lat] = float(np.trapz(total_pwr, dx=dt))
        stat_surv_entries[lat] = surv_entries
        stat_emer_times[lat] = emer_times
    print("\n==============================")
    print("Unified Estimated Probe Death and Survival Times:")
    print("Latitude (deg) | Time to Death (h) | Time to Survival Mode (h)")
    print("--------------------------------------------------------------")
    for lat in latitudes:
        t_d = stat_death[lat]
        t_s = stat_surv_first.get(lat, None)
        print(f"{lat:>14} | {t_d:>16.2f} | {t_s if t_s is not None else 'Never':>24}")
    print("\n==============================")
    print(f"Total Energy Generated Over 28 Sols – {selected_season}:")
    print("Latitude (deg) | Total Energy (Wh)")
    print("----------------------------------")
    for lat in latitudes:
        print(f"{lat:>14} | {stat_energy[lat]:>14.2f}")
    print("\n==============================")
    print("Survival Mode Statistics:")
    print("Latitude (deg) | # Entries")
    print("--------------------------")
    for lat in latitudes:
        entries = len(stat_surv_entries[lat])
        print(f"{lat:>14} | {entries:>9}")
    print("\n==============================")
    print("Emergency Battery Usage:")
    print("Latitude (deg) | Activation Times (h)")
    print("-------------------------------------")
    for lat in latitudes:
        acts = stat_emer_times[lat]
        acts_s = ', '.join(f"{a:.2f}" for a in acts) if acts else "Never Used"
        print(f"{lat:>14} | {acts_s}")

# ---- Environment summary (v0.2 look)
def print_environment_summary_v02(df: pd.DataFrame, num_hours: float, dt_s: float):
    if df.empty: return
    print("="*30)
    print("ENVIRONMENT SUMMARY  (proxy)")
    print("="*30)
    print(f"Min/Max surface temp : {df.T_air_K.min()-273.15:.1f} / {df.T_air_K.max()-273.15:.1f} °C")
    print(f"Min battery SoC      : {df.soc.min()*100:.1f} %")
    capped_hours = float(df['capped'].sum()) * (dt_s/3600.0)
    print(f"Capped/Brownout hrs  : {capped_hours:.1f} h")
    tau_avg = df.get('tau', pd.Series([TAU_DUST]*len(df))).mean()
    print(f"Tau (avg)            : {tau_avg:.2f}")
    storm_h = float(df.get('storm', pd.Series(0, index=df.index)).sum()) * (dt_s/3600.0)
    print(f"Storm time           : {storm_h:.1f} h")
    print("="*30)

# ============================================================
# ====================== PRINT SUMMARY =======================
# ============================================================
def print_summary(summary: Dict):
    print("\n=== MAGPIE Probe Env + Power + Thermal — Summary ===")
    sim = summary["sim"]; res = summary["results"]
    print(f"Duration: {sim['duration_h']:.1f} h  |  dt: {sim['dt_s']:.1f} s  |  steps: {sim['steps']}")
    print(f"Final SOC: {res['final_soc']:.3f}  |  Min SOC: {res['min_soc']:.3f}")
    print(f"Heater duty: {100.0*res['heater_duty']:.1f}%  |  Heater energy: {res['heater_energy_Wh']:.1f} Wh")
    print(f"Min temps (°C): skin {res['min_T_skin_K']-273.15:.1f}, int {res['min_T_int_K']-273.15:.1f}, batt {res['min_T_batt_K']-273.15:.1f}")
    print(f"Brownout caps: {res['brownout_capped_pct']:.1f}% of timesteps")
    print(f"Harvested: {res['harvested_Wh']:.1f} Wh  |  Consumed (no heater): {res['consumed_Wh_no_heater']:.1f} Wh  |  Heater: {res['heater_Wh_cum']:.1f} Wh")
    print(f"Single-use batt: used {res['single_use_Wh_used']:.2f} Wh, remaining {res['single_use_Wh_remaining']:.2f} Wh (enabled={summary['single_use']['enabled']})")
    print(f"Peak panel W: {res['peak_panel_W']:.2f}  | Storm time: {res['storm_time_h']:.1f} h  | Avg tau: {res['avg_tau']:.2f}")
    if summary.get('lint'):
        for m in summary['lint']:
            print(f"[lint] {m}")
    print("\n-- Per-Mode Energy (Wh, no heater) --")
    for m, val in res.get("mode_energy_Wh", {}).items():
        print(f"  {m:>8}: {val:.1f}")
    # MLI echo
    therm = summary["thermal"]
    print("\n-- Thermal insulation --")
    print(f"  Internal MLI: enabled={therm['enable_mli_internal']}  d={therm['mli_internal_d']*1e3:.1f} mm  k={therm['mli_k']}")
    print(f"  Battery  MLI: enabled={therm['enable_mli_batt']}      d={therm['mli_batt_d']*1e3:.1f} mm  k={therm['mli_k']}")

# ============================================================
# ========================= I/O ==============================
# ============================================================
def save_outputs(df: pd.DataFrame, summary: Dict, outdir: str, csv_name: str, json_name: str, write_manifest: bool):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, csv_name)
    json_path = os.path.join(outdir, json_name)
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as f: json.dump(summary, f, indent=2)
    print(f"Saved CSV → {csv_path}")
    print(f"Saved JSON → {json_path}")
    if write_manifest:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "geometry": summary.get("geometry", {}),
            "sim": summary.get("sim", {}),
            "panel": summary.get("panel", {}),
            "battery": summary.get("battery", {}),
            "single_use": summary.get("single_use", {}),
            "thermal": summary.get("thermal", {}),
            "artifacts": {"csv": csv_path, "json": json_path}
        }
        mpath = os.path.join(outdir, "run_manifest.json")
        with open(mpath, 'w') as f: json.dump(manifest, f, indent=2)
        print(f"Manifest saved → {mpath}")

# ============================================================
# ========================== CLI =============================
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MAGPIE Probe Env + Power + Thermal Simulator v0.3")
    parser.add_argument("--hours", type=float, default=SIM_DURATION_H, help="Simulation duration (hours)")
    parser.add_argument("--dt", type=float, default=SIM_DT_S, help="Timestep (seconds)")
    parser.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT, help="Output directory")
    parser.add_argument("--plots", action="store_true", help="Enable plots")
    parser.add_argument("--saveplots", action="store_true", help="Save plots to files")
    parser.add_argument("--showplots", action="store_true", help="Show plot windows")
    parser.add_argument("--csv", type=str, default=CSV_NAME_DEFAULT, help="CSV output filename")
    parser.add_argument("--json", type=str, default=JSON_NAME_DEFAULT, help="JSON summary filename")
    parser.add_argument("--manifest", action="store_true", help="Write manifest file")
    parser.add_argument("--mli-int", action="store_true", help="Enable internal MLI insulation")
    parser.add_argument("--mli-batt", action="store_true", help="Enable battery MLI insulation")
    parser.add_argument("--foldout", action="store_true", help="Print note that fold-out panels are assumed deployed")
    parser.add_argument("--backup", action="store_true", help="Enable single-use backup battery (override if disabled globally)")
    parser.add_argument("--geom", type=str,
                        choices=["cone","cylinder","hybrid","foldout","foldout_top"],
                        help="Override solar geometry mode")
    # NOTE: currently using parse_args([]) so it behaves like a module; swap to parser.parse_args()
    # for real CLI use.
    args = parser.parse_args([])

    # --- Initialize configs ---
    panel = PanelConfig()
    batt = BatteryConfig()
    therm = ThermalConfig(
        enable_mli_internal=args.mli_int,
        enable_mli_batt=args.mli_batt
    )
    env = EnvConfig()

    # --- Apply runtime toggles ---
    if args.foldout:
        print("[Fold-out panels assumed deployed for geometry modes that use them]")

    # Geometry override (non-breaking; default remains GEOMETRY_MODE from top)
    global GEOMETRY_MODE
    if args.geom:
        GEOMETRY_MODE = args.geom
        print(f"[Geometry mode override] GEOMETRY_MODE = '{GEOMETRY_MODE}'")

    # Backup battery override (only forces ON; default True is preserved otherwise)
    global ENABLE_SINGLE_USE_BATT
    if args.backup:
        ENABLE_SINGLE_USE_BATT = True
        print("[Backup single-use battery enabled]")

    # --- Run simulation ---
    df, summary = run_sim(
        sim_hours=args.hours,
        dt_s=args.dt,
        panel=panel,
        batt=batt,
        therm=therm,
        env=env
    )

    # --- Print results and save ---
    print_summary(summary)
    save_outputs(df, summary, args.outdir, args.csv, args.json, write_manifest=args.manifest)

    # --- Plot logic ---
    if args.plots or ENABLE_PLOTS_DEFAULT:
        plot_timeseries(df, args.outdir, show=args.showplots or PLOTS_SHOW_DEFAULT,
                        save=args.saveplots or PLOTS_SAVE_DEFAULT)
        plot_environment_style_v02(df, args.outdir, show=args.showplots or PLOTS_SHOW_DEFAULT,
                                   save=args.saveplots or PLOTS_SAVE_DEFAULT)
        daily_curves_and_tables_v02(args.outdir, show=args.showplots or PLOTS_SHOW_DEFAULT,
                                    save=args.saveplots or PLOTS_SAVE_DEFAULT)
        plot_long_trends_v02(df, args.outdir, show=args.showplots or PLOTS_SHOW_DEFAULT,
                             save=args.saveplots or PLOTS_SAVE_DEFAULT)
        print_environment_summary_v02(df, num_hours=args.hours, dt_s=args.dt)
        long_trend_tables_v02([90,45,0,-45,-90], selected_season="Autumn",
                              outdir=args.outdir, show=args.showplots or PLOTS_SHOW_DEFAULT,
                              save=args.saveplots or PLOTS_SAVE_DEFAULT)

if __name__ == "__main__":
    main()

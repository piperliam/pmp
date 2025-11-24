# pmp/edl/edl_1d.py
"""
Generic 1D EDL (Entry–Descent–Landing) Framework
================================================

Mission-agnostic point-mass EDL integrator for downward (vertical)
trajectories through an atmosphere with:

- Gravity (from central body mu & radius).
- Exponential atmosphere (rho = rho0 * exp(-h / H)).
- Aero drag with constant Cd, reference area.
- Simple parachute deployment logic (altitude + q + Mach windows).
- Optional retro-propulsive burn near the ground.

This is **not** a full port of MAGPIE `reentry_2.1c.py`, which is
truncated in the uploaded version and cannot be safely wrapped.
Instead, this provides a clean PMP EDL kernel that can be used for:

- Mars ballistic / chute / retro EDL.
- Other bodies (Earth, Titan, etc.) by changing parameters.

High-level API:
    simulate_edl(cfg: EDLSimConfig) -> EDLSimResult

which returns:
    - df: time history DataFrame
    - events: list of (t, label, extra) dicts

All mission-specific plotting, CSV/JSON export, and advanced
aerothermal / honeycomb modeling should live in mission code
(e.g., pmp.missions.magpie.edl).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional

import math
import numpy as np
import pandas as pd


# ============================================================
# Dataclasses – configuration
# ============================================================

@dataclass
class BodyAtmosphere:
    """
    Simple exponential atmosphere + gravity model tied to a body.

    mu_m3_s2    : GM of the body [m^3/s^2]
    radius_m    : reference radius [m]
    rho0_kg_m3  : reference density at h=0 [kg/m^3]
    hscale_m    : density scale height [m]
    gamma       : heat capacity ratio (for Mach estimate)
    R_spec      : specific gas constant [J/(kg*K)] (for Mach estimate)
    T0_K        : reference temperature [K] (assumed constant)
    """

    mu_m3_s2: float
    radius_m: float
    rho0_kg_m3: float
    hscale_m: float
    gamma: float = 1.3
    R_spec: float = 190.0
    T0_K: float = 210.0

    def gravity(self, alt_m: float) -> float:
        r = self.radius_m + max(alt_m, 0.0)
        return self.mu_m3_s2 / (r * r)

    def density(self, alt_m: float) -> float:
        if alt_m < 0.0:
            alt_m = 0.0
        return self.rho0_kg_m3 * math.exp(-alt_m / self.hscale_m)

    def speed_of_sound(self) -> float:
        return math.sqrt(self.gamma * self.R_spec * self.T0_K)


@dataclass
class VehicleAeroshell:
    """
    Basic aeroshell / descent vehicle configuration.

    mass_kg : initial mass at EI [kg]
    Cd      : drag coefficient
    area_m2 : reference cross-sectional area [m^2]
    """
    mass_kg: float
    Cd: float
    area_m2: float


@dataclass
class ParachuteConfig:
    """
    Simple single-stage main chute model.

    Cd        : chute drag coefficient
    area_m2   : full canopy area [m^2]
    deploy_alt_m : altitude at/below which chute is allowed to deploy
    q_min_Pa : minimum dynamic pressure [Pa] for safe deploy
    q_max_Pa : maximum dynamic pressure [Pa] for safe deploy
    mach_min : minimum Mach number for corridor
    mach_max : maximum Mach number for corridor

    If all corridor bounds are None, only deploy_alt_m is used.
    """
    Cd: float
    area_m2: float
    deploy_alt_m: float
    q_min_Pa: Optional[float] = None
    q_max_Pa: Optional[float] = None
    mach_min: Optional[float] = None
    mach_max: Optional[float] = None


@dataclass
class RetroConfig:
    """
    Simple retro-propulsive burn model.

    thrust_N  : constant thrust [N] during burn
    Isp_s     : specific impulse [s]
    burn_alt_m: altitude below which burn can start [m]
    cutoff_v_mps: stop burn if speed falls below this [m/s]
    """
    thrust_N: float
    Isp_s: float
    burn_alt_m: float
    cutoff_v_mps: float


@dataclass
class EDLSimConfig:
    """
    Top-level EDL simulation config.

    body         : BodyAtmosphere
    vehicle      : VehicleAeroshell
    t_max_s      : max sim time [s]
    dt_s         : time step [s]
    h0_m         : initial altitude above reference [m]
    v0_mps       : initial vertical velocity [m/s] (positive downward)
    parachute    : optional main chute config
    retro        : optional retro-burn config
    enable_retro : whether retro is allowed
    """
    body: BodyAtmosphere
    vehicle: VehicleAeroshell
    t_max_s: float
    dt_s: float
    h0_m: float
    v0_mps: float
    parachute: Optional[ParachuteConfig] = None
    retro: Optional[RetroConfig] = None
    enable_retro: bool = True


# ============================================================
# Dataclasses – results
# ============================================================

@dataclass
class EDLEvent:
    t_s: float
    label: str
    details: Dict[str, Any]


@dataclass
class EDLSimResult:
    df: pd.DataFrame
    events: List[EDLEvent]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Summary without full timeseries."""
        return {
            "n_steps": int(len(self.df)),
            "events": [e.__dict__ for e in self.events],
            "meta": self.meta,
            "columns": list(self.df.columns),
        }


# ============================================================
# Core integrator (simple explicit Euler)
# ============================================================

def simulate_edl(cfg: EDLSimConfig) -> EDLSimResult:
    """
    Run a simple 1D EDL simulation.

    State:
        h [m] : altitude above reference surface (downward positive vel)
        v [m/s] : vertical speed (positive downward)
        m [kg] : vehicle mass

    Forces:
        + gravity (downward)
        - drag (upward)
        + retro thrust (upward) if enabled and active

    Integration is explicit Euler for simplicity; dt should be small.

    Returns:
        EDLSimResult with:
          - df: time history
          - events: list of EDL events (deploy, retro start, touchdown)
    """
    body = cfg.body
    veh = cfg.vehicle

    dt = cfg.dt_s
    n_steps = int(cfg.t_max_s / dt) + 1

    # Allocate arrays
    t_arr = np.zeros(n_steps)
    h_arr = np.zeros(n_steps)
    v_arr = np.zeros(n_steps)
    m_arr = np.zeros(n_steps)
    rho_arr = np.zeros(n_steps)
    q_arr = np.zeros(n_steps)
    a_arr = np.zeros(n_steps)
    mach_arr = np.zeros(n_steps)
    drag_arr = np.zeros(n_steps)
    thrust_arr = np.zeros(n_steps)
    chute_deployed = np.zeros(n_steps, dtype=bool)
    retro_active = np.zeros(n_steps, dtype=bool)

    # Initial state
    t = 0.0
    h = cfg.h0_m
    v = cfg.v0_mps
    m = veh.mass_kg

    a_sound = body.speed_of_sound()

    events: List[EDLEvent] = []

    # Helper to check chute corridor
    def chute_corridor_ok(q, mach) -> bool:
        pc = cfg.parachute
        if pc is None:
            return False
        if pc.q_min_Pa is not None and q < pc.q_min_Pa:
            return False
        if pc.q_max_Pa is not None and q > pc.q_max_Pa:
            return False
        if pc.mach_min is not None and mach < pc.mach_min:
            return False
        if pc.mach_max is not None and mach > pc.mach_max:
            return False
        return True

    # Simulation loop
    for i in range(n_steps):
        t_arr[i] = t
        h_arr[i] = h
        v_arr[i] = v
        m_arr[i] = m

        # Atmosphere + aero
        rho = body.density(h)
        rho_arr[i] = rho
        q = 0.5 * rho * v * v
        q_arr[i] = q
        mach = v / max(a_sound, 1e-3)
        mach_arr[i] = mach

        # Base drag
        A_eff = veh.area_m2
        Cd_eff = veh.Cd

        # Chute deployment check
        if cfg.parachute is not None:
            pc = cfg.parachute
            if (not chute_deployed[i - 1] if i > 0 else True):
                if h <= pc.deploy_alt_m:
                    if (pc.q_min_Pa is None and pc.q_max_Pa is None and
                        pc.mach_min is None and pc.mach_max is None):
                        # altitude-only trigger
                        deployed_now = True
                    else:
                        deployed_now = chute_corridor_ok(q, mach)
                    if deployed_now:
                        events.append(EDLEvent(t, "MAIN_CHUTE_DEPLOY", {
                            "h_m": h,
                            "q_Pa": q,
                            "mach": mach,
                        }))
                        chute_deployed[i] = True

        # If chute already deployed in prior step, carry flag forward
        if i > 0 and chute_deployed[i] is False:
            chute_deployed[i] = chute_deployed[i - 1]

        # Adjust drag if chute deployed
        if cfg.parachute is not None and chute_deployed[i]:
            pc = cfg.parachute
            A_eff = veh.area_m2 + pc.area_m2
            # Simple Cd blend: treat canopy as separate surface
            # Here we just use canopy Cd and ignore aeroshell drag
            Cd_eff = pc.Cd

        # Drag magnitude (always opposing motion)
        D = Cd_eff * A_eff * q
        drag_arr[i] = D

        # Gravity (downward positive)
        g = body.gravity(h)

        # Retro thrust
        T = 0.0
        if cfg.enable_retro and cfg.retro is not None:
            rc = cfg.retro
            if h <= rc.burn_alt_m and v > rc.cutoff_v_mps and m > 0.0:
                T = rc.thrust_N
                retro_active[i] = True
                # Mass flow from rocket eq: mdot = T / (Isp * g0)
                g0 = 9.80665
                mdot = T / (rc.Isp_s * g0)
                # reduce mass over this step
                m -= mdot * dt
                if m < 0.0:
                    m = 0.0
                if not any(e.label == "RETRO_START" for e in events):
                    events.append(EDLEvent(t, "RETRO_START", {
                        "h_m": h,
                        "v_mps": v,
                    }))
        thrust_arr[i] = T

        # Net force (downward positive): F = m*g - D - T
        F = m * g - D - T
        a = F / max(m, 1e-6)
        a_arr[i] = a

        # Integrate (explicit Euler)
        v = v + a * dt
        h = h - v * dt  # v is downward; decrease altitude as we move down

        # Touchdown condition
        if h <= 0.0:
            events.append(EDLEvent(t, "TOUCHDOWN", {
                "impact_v_mps": abs(v),
                "mass_kg": m,
            }))
            # clamp last state
            h_arr[i] = 0.0
            v_arr[i] = v
            # cut off arrays at touchdown index
            t_arr = t_arr[: i + 1]
            h_arr = h_arr[: i + 1]
            v_arr = v_arr[: i + 1]
            m_arr = m_arr[: i + 1]
            rho_arr = rho_arr[: i + 1]
            q_arr = q_arr[: i + 1]
            a_arr = a_arr[: i + 1]
            mach_arr = mach_arr[: i + 1]
            drag_arr = drag_arr[: i + 1]
            thrust_arr = thrust_arr[: i + 1]
            chute_deployed = chute_deployed[: i + 1]
            retro_active = retro_active[: i + 1]
            break

        # Advance time
        t += dt

    # Build DataFrame
    df = pd.DataFrame(
        {
            "t_s": t_arr,
            "h_m": h_arr,
            "v_mps": v_arr,
            "m_kg": m_arr,
            "rho_kg_m3": rho_arr,
            "q_Pa": q_arr,
            "a_mps2": a_arr,
            "mach": mach_arr,
            "D_N": drag_arr,
            "T_N": thrust_arr,
            "chute_deployed": chute_deployed,
            "retro_active": retro_active,
        }
    )

    meta: Dict[str, Any] = {
        "body": asdict(cfg.body),
        "vehicle": asdict(cfg.vehicle),
        "has_parachute": cfg.parachute is not None,
        "has_retro": cfg.retro is not None and cfg.enable_retro,
    }

    return EDLSimResult(df=df, events=events, meta=meta)


# ============================================================
# MAGPIE-style Mars presets (approximate)
# ============================================================

def magpie_mars_body_default() -> BodyAtmosphere:
    """
    Rough Mars atmosphere + gravity model to use with MAGPIE-like probes.
    """
    return BodyAtmosphere(
        mu_m3_s2=4.282837e13,
        radius_m=3389.5e3,
        rho0_kg_m3=0.02,     # ~20 g/m^3 near 0 m (order-of-magnitude)
        hscale_m=10_800.0,
        gamma=1.29,
        R_spec=188.9,
        T0_K=210.0,
    )


def magpie_probe_aeroshell_default() -> VehicleAeroshell:
    """
    Approximate MAGPIE probe aeroshell ballistic properties.
    Adjust these to match your v2.1c mass/area/Cd.
    """
    return VehicleAeroshell(
        mass_kg=41.0,       # bus + aeroshell
        Cd=1.6,             # sphere-cone-ish
        area_m2=0.78,       # 1 m dia -> pi * (0.5)^2
    )


def magpie_main_chute_default() -> ParachuteConfig:
    """
    Approximate main chute for MAGPIE-style EDL.
    """
    return ParachuteConfig(
        Cd=1.6,
        area_m2=8.0,
        deploy_alt_m=9000.0,
        q_min_Pa=200.0,
        q_max_Pa=900.0,
        mach_min=0.3,
        mach_max=2.0,
    )


def magpie_retro_default() -> RetroConfig:
    """
    Simple MAGPIE-esque retro config: low thrust just before touchdown.
    """
    return RetroConfig(
        thrust_N=1400.0,
        Isp_s=230.0,
        burn_alt_m=200.0,
        cutoff_v_mps=1.0,
    )


__all__ = [
    "BodyAtmosphere",
    "VehicleAeroshell",
    "ParachuteConfig",
    "RetroConfig",
    "EDLSimConfig",
    "EDLEvent",
    "EDLSimResult",
    "simulate_edl",
    # Presets
    "magpie_mars_body_default",
    "magpie_probe_aeroshell_default",
    "magpie_main_chute_default",
    "magpie_retro_default",
]

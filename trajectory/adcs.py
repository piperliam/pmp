# pmp/trajectory/adcs.py
"""
PMP ADCS Framework Wrapper
==========================

Framework-style wrappers around the MAGPIE ADCS v0.3 code.

The original script `ADCS_0.2a.py` is assumed to be moved into:

    pmp/legacy/magpie_adcs_v0_3.py

and left otherwise unchanged.

This module exposes:
- Dataclasses to describe a rigid body + RW cluster.
- A generic PD slew simulation:
    simulate_pd_slew(...) -> ADCSSlewResult

Internally, it uses the legacy functions:
- ReactionWheelCluster
- RigidBodyWithRWs
- simulate
- compute_rw_power
- estimate_settling_time
- quat_from_axis_angle
- attitude_error_vec

The goal is to make ADCS *mission-agnostic*:
any inertia, any wheel configuration, any axis/angle.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

import numpy as np

# Import the legacy MAGPIE ADCS module
from pmp.legacy import magpie_adcs_v0_3 as legacy


# ============================================================
# Dataclasses: configs
# ============================================================

@dataclass
class RWClusterConfig:
    """
    Reaction wheel cluster configuration (framework-level).

    axes      : 3xN array-like (each column or row is a unit axis in body frame)
    Iw        : rotor inertia [kg·m²]
    tau_max   : max wheel torque magnitude [N·m]
    w_max     : max wheel speed magnitude [rad/s]
    eta       : electrical efficiency used for power calc (0-1)
    """
    axes: np.ndarray
    Iw: float
    tau_max: float
    w_max: float
    eta: float = 0.7

    def as_legacy_cluster(self) -> legacy.ReactionWheelCluster:
        """
        Build the legacy ReactionWheelCluster from this config.
        """
        axes_arr = np.array(self.axes, dtype=float)
        # Allow both 3xN and Nx3 input; convert to 3xN
        if axes_arr.shape[0] != 3 and axes_arr.shape[1] == 3:
            axes_arr = axes_arr.T
        return legacy.ReactionWheelCluster(
            axes=axes_arr,
            Iw=self.Iw,
            tau_max=self.tau_max,
            w_max=self.w_max,
        )


@dataclass
class RigidBodyConfig:
    """
    Rigid body inertia configuration.

    I_body : 3x3 inertia matrix [kg·m²] in body frame.
    name   : optional label for logging/plots.
    """
    I_body: np.ndarray
    name: str = "vehicle"


@dataclass
class SlewPDConfig:
    """
    PD attitude controller configuration for slews.

    Kp, Kd          : PD gains in body frame
    axis_body       : axis of initial attitude offset (body frame)
    angle_deg       : initial slew angle [deg]; commanded target is identity.

    t_final         : final time for simulation [s]
    dt              : time step [s]
    w_thresh_deg    : body-rate magnitude threshold for settling [deg/s]
    ang_err_thresh_deg : attitude error threshold vs final [deg]
    """
    Kp: float
    Kd: float
    axis_body: np.ndarray
    angle_deg: float
    t_final: float
    dt: float
    w_thresh_deg: float = 0.01
    ang_err_thresh_deg: float = 0.1


@dataclass
class ADCSSlewSimConfig:
    """
    Top-level config for a single PD slew simulation.

    body    : inertia + name
    rw      : reaction wheel cluster config
    control : PD and slew specification
    """
    body: RigidBodyConfig
    rw: RWClusterConfig
    control: SlewPDConfig


# ============================================================
# Dataclasses: results
# ============================================================

@dataclass
class ADCSSlewResult:
    """
    Result of a single PD slew simulation.

    history:
        t         : [steps] time array [s]
        state     : [steps x (4+3+Nw)] array
                    [q0, q1, q2, q3, wx, wy, wz, w_rw...]

    power:
        dict from legacy.compute_rw_power:
            tau_rw       : [steps x Nw] wheel torques
            P_el         : [steps x Nw] electrical powers
            P_el_tot     : [steps] total RW power [W]
            E_J          : scalar total energy [J]
            E_Wh         : scalar total energy [Wh]

    metrics:
        t_settle   : estimated settling time [s]
        max_slew   : max |w_body| [deg/s]
        E_J        : total RW energy [J]
        E_Wh       : total RW energy [Wh]
        P_avg      : average RW power over sim [W]
        P_peak     : peak RW power over sim [W]
    """
    history: Dict[str, np.ndarray]
    power: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable summary (numerics only, no big arrays)."""
        return {
            "metrics": self.metrics,
            "history_meta": {
                "n_steps": int(len(self.history["t"])),
                "state_dim": int(self.history["state"].shape[1]),
            },
        }


# ============================================================
# Core wrapper function
# ============================================================

def simulate_pd_slew(cfg: ADCSSlewSimConfig) -> ADCSSlewResult:
    """
    Simulate a PD-controlled slew for a rigid body with RWs.

    This is a framework-friendly wrapper around the legacy MAGPIE ADCS
    code. It reproduces the style of the tables from:

        run_orbiter_slew_tables_all_axes()
        run_probe_slew_table()

    but instead of printing, it returns structured data.

    Pattern:
    - Initial attitude q0 = rotation about cfg.control.axis_body
      by cfg.control.angle_deg.
    - Desired attitude q_des is identity quaternion [1, 0, 0, 0].
    - Control law:
          tau_cmd = -Kp * attitude_error_vec(q, q_des) - Kd * w_body
    - Dynamics are run via:
          RigidBodyWithRWs.dynamics()
          simulate(...)

    Notes
    -----
    This function does not:
    - Plot,
    - Write tables,
    - Assume any specific vehicle (orbiter/probe).
    """
    # Unpack configs
    I_body = np.array(cfg.body.I_body, dtype=float)
    name = cfg.body.name

    rw_cfg = cfg.rw
    ctrl = cfg.control

    # Build legacy RW cluster
    rw_cluster = rw_cfg.as_legacy_cluster()

    # Build legacy rigid body + RWs
    vehicle = legacy.RigidBodyWithRWs(I_body, rw_cluster, name=name)

    # Normalize axis and build initial quaternion q0
    axis_vec = np.asarray(ctrl.axis_body, dtype=float)
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm == 0:
        raise ValueError("axis_body must be non-zero.")
    axis_vec /= axis_norm

    ang_rad = np.deg2rad(ctrl.angle_deg)
    q0 = legacy.quat_from_axis_angle(axis_vec, ang_rad)

    # Initial state: [q, w_body, w_rw]
    Nw = rw_cluster.A.shape[1]
    w0 = np.zeros(3)
    w_rw0 = np.zeros(Nw)
    state0 = np.concatenate((q0, w0, w_rw0))

    # Desired attitude: identity quaternion
    q_des = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Control law (PD in body frame)
    Kp = float(ctrl.Kp)
    Kd = float(ctrl.Kd)

    def tau_body_cmd(t, q, w_body, q_desired):
        e = legacy.attitude_error_vec(q, q_desired)
        return -Kp * e - Kd * w_body

    # Wrap into the RigidBodyWithRWs dynamics interface
    def dyn(t, state):
        return vehicle.dynamics(t, state, tau_body_cmd, q_des)

    # Run integration
    hist = legacy.simulate(
        state0,
        dyn_func=dyn,
        t_final=ctrl.t_final,
        dt=ctrl.dt,
    )

    # RW power
    power = legacy.compute_rw_power(hist, Iw=rw_cfg.Iw, eta=rw_cfg.eta)

    # Body rate metrics
    t_arr = hist["t"]
    state_arr = hist["state"]
    w_body = state_arr[:, 4:7]
    w_mag_deg = np.rad2deg(np.linalg.norm(w_body, axis=1))
    max_slew = float(np.max(w_mag_deg))

    # Settling time
    t_settle = float(
        legacy.estimate_settling_time(
            hist,
            w_thresh_deg=ctrl.w_thresh_deg,
            ang_err_thresh_deg=ctrl.ang_err_thresh_deg,
        )
    )

    # Energy / power metrics
    E_J = float(power["E_J"])
    E_Wh = float(power["E_Wh"])
    P_avg = E_J / t_arr[-1] if t_arr[-1] > 0 else 0.0
    P_peak = float(np.max(power["P_el_tot"]))

    metrics = {
        "t_settle_s": t_settle,
        "max_slew_deg_per_s": max_slew,
        "E_J": E_J,
        "E_Wh": E_Wh,
        "P_avg_W": P_avg,
        "P_peak_W": P_peak,
    }

    return ADCSSlewResult(history=hist, power=power, metrics=metrics)


# ============================================================
# MAGPIE-style convenience presets
# ============================================================

def magpie_default_orbiter_rw_cluster() -> RWClusterConfig:
    """
    Approximate MAGPIE orbiter RW250 3-axis cluster:
    - 3 wheels along body axes (identity matrix)
    - Uses IW_RW250, TAU_MAX_RW250, W_MAX_RW250 from legacy module.
    """
    axes = np.eye(3)
    return RWClusterConfig(
        axes=axes,
        Iw=legacy.IW_RW250,
        tau_max=legacy.TAU_MAX_RW250,
        w_max=legacy.W_MAX_RW250,
        eta=legacy.ETA_RW250 if hasattr(legacy, "ETA_RW250") else 0.7,
    )


def magpie_default_probe_rw_cluster() -> RWClusterConfig:
    """
    Approximate MAGPIE probe RW25 3-axis cluster.
    """
    axes = np.eye(3)
    return RWClusterConfig(
        axes=axes,
        Iw=legacy.IW_RW25,
        tau_max=legacy.TAU_MAX_RW25,
        w_max=legacy.W_MAX_RW25,
        eta=legacy.ETA_RW25 if hasattr(legacy, "ETA_RW25") else 0.6,
    )


def magpie_default_orbiter_body() -> RigidBodyConfig:
    """
    Default orbiter inertia used in the legacy tables:
    diag([250, 220, 300]) kg·m²
    """
    I_orbiter = np.diag([250.0, 220.0, 300.0])
    return RigidBodyConfig(I_body=I_orbiter, name="MAGPIE_Orbiter")


def magpie_default_probe_body() -> RigidBodyConfig:
    """
    Default probe + aeroshell inertia used in run_probe_slew_table.
    """
    Ixx_gmm2 = 3.9596118029e10
    Iyy_gmm2 = 3.9596118029e10
    Izz_gmm2 = 5.652922409e9
    I_probe = np.diag([
        Ixx_gmm2 * 1e-9,
        Iyy_gmm2 * 1e-9,
        Izz_gmm2 * 1e-9,
    ])
    return RigidBodyConfig(I_body=I_probe, name="MAGPIE_Probe")


__all__ = [
    "RWClusterConfig",
    "RigidBodyConfig",
    "SlewPDConfig",
    "ADCSSlewSimConfig",
    "ADCSSlewResult",
    "simulate_pd_slew",
    # MAGPIE presets
    "magpie_default_orbiter_rw_cluster",
    "magpie_default_probe_rw_cluster",
    "magpie_default_orbiter_body",
    "magpie_default_probe_body",
]

"""
PMP Generic Transfer Module
---------------------------

Generic Lambert-transfer utilities for ANY two bodies around
a central attractor (Sun by default).

This is the mission-agnostic core extracted from the ideas in
MAGPIE's `orbit_1.4c.py`, but *not* hard-coded to Earth/Mars.

Features
--------
- Lambert arc between origin and target bodies.
- Uses poliastro ephemerides for body positions, or user-specified
  state vectors.
- Returns:
  * Heliocentric (or central-body) departure/arrival states
  * v_inf at departure/arrival (relative to each body)
  * Δv at departure/arrival from Lambert solution
  * Time of flight
- Optional MOI Δv estimate for circular capture at a given altitude.

All plotting, CSV/JSON logging, porkchop sweeps, and MAGPIE-specific
logic should live in:
    pmp/missions/<mission_name>/transfer.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from poliastro.iod.izzo import lambert


# =============================================================
# Data classes
# =============================================================

@dataclass
class EndpointConfig:
    """
    Configuration for one endpoint of the transfer.

    You can either:
      - provide a poliastro body object (Earth, Mars, etc.) and let
        PMP pull ephemerides at the given epoch; or
      - provide explicit position/velocity vectors in km and km/s
        in the central body's inertial frame.

    If both `body` and `r/v` are provided, r/v take precedence.
    """
    epoch_utc: str
    body: Optional[Any] = None  # poliastro body or compatible
    r_km: Optional[np.ndarray] = None
    v_km_s: Optional[np.ndarray] = None


@dataclass
class TransferConfig:
    """
    Generic Lambert transfer configuration.

    central_body : poliastro body around which the motion is defined
        (Sun for interplanetary, Earth for Earth–Moon, etc.).

    origin, target : EndpointConfig
        Defines where and when the transfer begins and ends.

    long_way : bool
        Whether to use the long-way Lambert solution instead of the
        default short-way one.

    moi_body_mu_km3_s2 / moi_periapsis_alt_km :
        If both are set, an approximate circular MOI Δv will be computed.
    """
    central_body: Any            # e.g. Sun, Earth, etc.
    origin: EndpointConfig
    target: EndpointConfig
    long_way: bool = False

    # Optional MOI estimate (at arrival)
    moi_body_mu_km3_s2: Optional[float] = None
    moi_periapsis_alt_km: Optional[float] = None


@dataclass
class TransferResult:
    """
    Result of a Lambert transfer between two endpoints.
    """
    # Heliocentric / central-body states
    r1_km: np.ndarray
    v1_km_s: np.ndarray
    r2_km: np.ndarray
    v2_km_s: np.ndarray

    # Lambert departure/arrival velocities
    v1_lambert_km_s: np.ndarray
    v2_lambert_km_s: np.ndarray

    # v_inf at departure/arrival (relative to each body)
    v_inf_depart_km_s: Optional[np.ndarray]
    v_inf_arrive_km_s: Optional[np.ndarray]

    # Magnitudes and Δv
    dv_depart_km_s: float
    dv_arrive_km_s: float
    v_inf_depart_mag_km_s: Optional[float]
    v_inf_arrive_mag_km_s: Optional[float]

    # Time of flight
    tof_days: float

    # Optional MOI Δv (approx circular capture)
    dv_moi_km_s: Optional[float]


# =============================================================
# Helpers
# =============================================================

def _endpoint_state(
    cfg: EndpointConfig,
    central_body: Any,
) -> tuple[np.ndarray, np.ndarray, Orbit | None]:
    """
    Resolve endpoint state.

    Returns:
        r_km, v_km_s, orb (if ephemeris-based Orbit used, else None)
    """
    t = Time(cfg.epoch_utc, scale="tdb")

    if cfg.r_km is not None and cfg.v_km_s is not None:
        r = np.array(cfg.r_km, dtype=float)
        v = np.array(cfg.v_km_s, dtype=float)
        return r, v, None

    if cfg.body is None:
        raise ValueError("EndpointConfig requires either (r, v) or a body with ephemeris.")

    # Use poliastro ephemeris for that body at epoch
    orb = Orbit.from_body_ephem(cfg.body, t)
    r = orb.r.to(u.km).value
    v = orb.v.to(u.km / u.s).value
    return r, v, orb


def _estimate_circular_moi_dv(
    v_inf_mag_km_s: float,
    mu_km3_s2: float,
    rp_km: float,
) -> float:
    """
    Approximate Δv for a single-impulse circular capture at periapsis
    of a v_inf hyperbolic arrival.

    v_inf_mag_km_s : magnitude of arrival excess velocity [km/s]
    mu_km3_s2      : gravitational parameter of target body [km^3/s^2]
    rp_km          : periapsis radius (body radius + altitude) [km]
    """
    # hyperbolic periapsis speed
    v_peri = np.sqrt(v_inf_mag_km_s**2 + 2.0 * mu_km3_s2 / rp_km)
    v_circ = np.sqrt(mu_km3_s2 / rp_km)
    return float(v_peri - v_circ)


# =============================================================
# Main interface
# =============================================================

def compute_lambert_transfer(cfg: TransferConfig) -> TransferResult:
    """
    Compute a generic Lambert transfer for any two endpoints
    around a central body (Sun by default).

    This function does:
    1. Get origin & target state vectors (from ephemeris or direct input).
    2. Run Izzo Lambert solver.
    3. Compute v_inf at departure/arrival (if body ephemerides available).
    4. Compute Δv at departure/arrival.
    5. Optionally estimate MOI Δv for circular capture at arrival.

    It does NOT:
    - Do launch vehicle modeling.
    - Do porkchop sweeps.
    - Plot anything.
    - Write any files.

    That higher-level logic should live in mission-specific code.
    """

    # --------------------
    # Get endpoint states
    # --------------------
    r1_km, v1_km_s, orb1 = _endpoint_state(cfg.origin, cfg.central_body)
    r2_km, v2_km_s, orb2 = _endpoint_state(cfg.target, cfg.central_body)

    # --------------------
    # Time of flight
    # --------------------
    t1 = Time(cfg.origin.epoch_utc, scale="tdb")
    t2 = Time(cfg.target.epoch_utc, scale="tdb")
    tof = (t2 - t1).to(u.day).value

    # --------------------
    # Lambert solution
    # --------------------
    k = cfg.central_body.k if hasattr(cfg.central_body, "k") else cfg.central_body.mu

    r1_q = r1_km * u.km
    r2_q = r2_km * u.km

    # poliastro.iod.izzo.lambert(k, r1, r2, tof)
    v1_lambert_q, v2_lambert_q = lambert(k, r1_q, r2_q, tof * u.day, numiter=35, rtol=1e-8)

    v1_lambert = v1_lambert_q.to(u.km / u.s).value
    v2_lambert = v2_lambert_q.to(u.km / u.s).value

    # Δv at departure & arrival in central-body frame
    dv_depart = float(np.linalg.norm(v1_lambert - v1_km_s))
    dv_arrive = float(np.linalg.norm(v2_km_s - v2_lambert))

    # --------------------
    # v_inf at departure/arrival (if body ephemerides used)
    # --------------------
    v_inf_depart = None
    v_inf_arrive = None
    v_inf_depart_mag = None
    v_inf_arrive_mag = None

    if orb1 is not None:
        # body velocity in central frame = v1_km_s
        v_inf_depart = v1_lambert - v1_km_s
        v_inf_depart_mag = float(np.linalg.norm(v_inf_depart))

    if orb2 is not None:
        v_inf_arrive = v2_lambert - v2_km_s
        v_inf_arrive_mag = float(np.linalg.norm(v_inf_arrive))

    # --------------------
    # Optional MOI Δv estimate
    # --------------------
    dv_moi = None
    if (
        cfg.moi_body_mu_km3_s2 is not None
        and cfg.moi_periapsis_alt_km is not None
        and v_inf_arrive_mag is not None
    ):
        # need radius of target body; if we have orb2.body, use that; else user
        # must have given rp directly including radius.
        if orb2 is not None and hasattr(orb2, "attractor"):
            R_body_km = orb2.attractor.R.to(u.km).value
            rp_km = R_body_km + cfg.moi_periapsis_alt_km
        else:
            # assume user passed "periapsis radius" directly as alt_km; not ideal,
            # but avoids forcing body info.
            rp_km = cfg.moi_periapsis_alt_km

        dv_moi = _estimate_circular_moi_dv(
            v_inf_arrive_mag_km_s=v_inf_arrive_mag,
            mu_km3_s2=cfg.moi_body_mu_km3_s2,
            rp_km=rp_km,
        )

    return TransferResult(
        r1_km=r1_km,
        v1_km_s=v1_km_s,
        r2_km=r2_km,
        v2_km_s=v2_km_s,
        v1_lambert_km_s=v1_lambert,
        v2_lambert_km_s=v2_lambert,
        v_inf_depart_km_s=v_inf_depart,
        v_inf_arrive_km_s=v_inf_arrive,
        dv_depart_km_s=dv_depart,
        dv_arrive_km_s=dv_arrive,
        v_inf_depart_mag_km_s=v_inf_depart_mag,
        v_inf_arrive_mag_km_s=v_inf_arrive_mag,
        tof_days=tof,
        dv_moi_km_s=dv_moi,
    )

"""
PMP Communications – Mars Probe–Orbiter–Earth Links

Refactor of MAGPIE `comm1d.py` into reusable library functions.

This module focuses on:
- Probe–orbiter UHF / LoRa links (surface to low Mars orbiter).
- Deep-space X-band relay from orbiter to an Earth DSN dish.

Notes
-----
This is *not* a 1:1 port of the original script – which also includes:
- Radio occultation inversion and refractivity profiles.
- Latitudinal sweeps and plotting logic.
Those are left for a future `occultation` / `analysis` module.

Here we provide:
- Clean numerical kernels for link budgets.
- Simple `dataclass` containers for configs and results.
- Defaults that reproduce the MAGPIE-style Mars geometry
  (250 km orbiter altitude, 433 / 915 MHz probes, 200 W X-band).

All distances are SI (m) internally; helpers accept km for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def fspl_db(freq_hz: float, distance_m: float) -> float:
    """
    Free-space path loss (FSPL) in dB.

    Parameters
    ----------
    freq_hz : float
        Carrier frequency in Hz.
    distance_m : float
        Distance between transmitter and receiver in meters.

    Returns
    -------
    float
        FSPL in dB.
    """
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(freq_hz) - 147.55


def watts_to_dbm(power_w: float) -> float:
    """Convert Watts -> dBm."""
    if power_w <= 0:
        raise ValueError("Power in Watts must be positive.")
    return 10.0 * np.log10(power_w * 1e3)


def dbm_to_watts(power_dbm: float) -> float:
    """Convert dBm -> Watts."""
    return 10.0 ** (power_dbm / 10.0) / 1e3


# ---------------------------------------------------------------------------
# Probe–Orbiter link configuration & results
# ---------------------------------------------------------------------------

@dataclass
class SurfaceProbeRadio:
    """Surface probe radio / antenna config for a single band."""
    name: str
    frequency_hz: float
    tx_power_w: float
    rx_sensitivity_dbm: float
    antenna_gain_dbi: float = 0.0
    system_losses_db: float = 0.0


@dataclass
class OrbiterRadio:
    """Orbiter radio / antenna config for a single band."""
    name: str
    tx_power_w: float
    antenna_gain_dbi: float
    system_losses_db: float = 0.0


@dataclass
class StaticPassConfig:
    """Simple static pass model: constant bitrate over fixed duration."""
    bitrate_bps: float
    duration_s: float


@dataclass
class StaticLinkResult:
    """Static probe–orbiter link margins and data volume."""
    range_km: float
    uplink_margin_db: float
    downlink_margin_db: float
    uplink_ok: bool
    downlink_ok: bool
    static_data_MB: float


def compute_static_probe_orbiter_link(
    probe: SurfaceProbeRadio,
    orbiter: OrbiterRadio,
    range_km: float,
    static_pass: StaticPassConfig,
) -> StaticLinkResult:
    """
    Compute static probe–orbiter link margins and data volume.

    Mirrors the "10-minute static pass" logic from the original script
    but generalised and parameterised.
    """
    d_m = range_km * 1e3
    fspl = fspl_db(probe.frequency_hz, d_m)

    # Uplink: probe -> orbiter
    tx_probe_dbm = watts_to_dbm(probe.tx_power_w)
    rx_uplink_dbm = (
        tx_probe_dbm
        + probe.antenna_gain_dbi
        + orbiter.antenna_gain_dbi
        - fspl
        - (probe.system_losses_db + orbiter.system_losses_db)
    )
    uplink_margin = rx_uplink_dbm - probe.rx_sensitivity_dbm
    uplink_ok = uplink_margin > 0.0

    # Downlink: orbiter -> probe, re-using same band parameters
    tx_orbiter_dbm = watts_to_dbm(orbiter.tx_power_w)
    rx_downlink_dbm = (
        tx_orbiter_dbm
        + orbiter.antenna_gain_dbi
        + probe.antenna_gain_dbi
        - fspl
        - (probe.system_losses_db + orbiter.system_losses_db)
    )
    downlink_margin = rx_downlink_dbm - probe.rx_sensitivity_dbm
    downlink_ok = downlink_margin > 0.0

    # Static data volume (if link viable and bitrate defined)
    if static_pass.bitrate_bps > 0 and uplink_ok and downlink_ok:
        static_data_bits = static_pass.bitrate_bps * static_pass.duration_s
        static_data_MB = static_data_bits / 8.0 / 1e6
    else:
        static_data_MB = 0.0

    return StaticLinkResult(
        range_km=range_km,
        uplink_margin_db=uplink_margin,
        downlink_margin_db=downlink_margin,
        uplink_ok=uplink_ok,
        downlink_ok=downlink_ok,
        static_data_MB=static_data_MB,
    )


# ---------------------------------------------------------------------------
# Dynamic elevation-arc link
# ---------------------------------------------------------------------------

@dataclass
class DynamicLinkResult:
    """
    Dynamic elevation-arc link integration result.

    The dynamic profile is sampled along a pass with:
    - Elevation(t) = 90° * sin(pi * t / T_total)
    - At each t, a user-supplied profile function defines
      (tx_power_W, bitrate_bps) as a function of elevation.
    """
    range_km: float
    valid_time_s: float
    avg_tx_power_w: float
    dynamic_data_MB: float
    uplink_margin_min_db: float
    downlink_margin_min_db: float


DynamicProfileFn = Callable[[float], Tuple[float, float]]
# signature: elev_deg -> (tx_power_w, bitrate_bps)


def default_dynamic_profile(elev_deg: float) -> Tuple[float, float]:
    """
    Example dynamic profile inspired by the original script.

    Very low elevation -> no transmission.
    Mid elevations -> moderate power/bitrate.
    Near zenith -> higher bitrate but same nominal probe power.

    This is only a *placeholder*; missions should provide their
    own mapping based on regulations and hardware.
    """
    if elev_deg < 10.0:
        return 0.0, 0.0
    elif elev_deg < 30.0:
        return 1.0, 1_000.0    # 1 W, 1 kbps
    elif elev_deg < 60.0:
        return 1.5, 5_000.0    # 1.5 W, 5 kbps
    else:
        return 2.0, 10_000.0   # 2 W, 10 kbps


def compute_dynamic_probe_orbiter_link(
    probe: SurfaceProbeRadio,
    orbiter: OrbiterRadio,
    range_km: float,
    total_pass_duration_s: float,
    time_steps: int = 200,
    profile_fn: DynamicProfileFn = default_dynamic_profile,
) -> DynamicLinkResult:
    """
    Integrate a dynamic elevation-arc pass for a probe–orbiter link.

    Parameters
    ----------
    probe, orbiter : configs
        Radio & antenna configuration for each end.
    range_km : float
        Slant range from probe to orbiter in kilometers.
        For a circular 250 km orbit directly overhead, this
        will be close to 250 km; missions may adjust.
    total_pass_duration_s : float
        Total duration of the pass (rise -> set).
    time_steps : int
        Number of discrete time samples along the pass.
    profile_fn : callable
        Function mapping elevation_deg -> (tx_power_W, bitrate_bps).

    Returns
    -------
    DynamicLinkResult
        Contains integrated data volume, average TX power, and
        worst-case link margins over samples where bitrate > 0.
    """
    d_m = range_km * 1e3
    fspl = fspl_db(probe.frequency_hz, d_m)

    t = np.linspace(0.0, total_pass_duration_s, time_steps)
    elev_deg = 90.0 * np.sin(np.pi * t / total_pass_duration_s)

    dt = total_pass_duration_s / (time_steps - 1)

    total_bits = 0.0
    total_energy = 0.0
    valid_time = 0.0

    uplink_margins: List[float] = []
    downlink_margins: List[float] = []

    for e in elev_deg:
        tx_power_w, bitrate_bps = profile_fn(float(e))
        if bitrate_bps <= 0.0 or tx_power_w <= 0.0:
            # No transmission at this elevation
            continue

        # Uplink margin at this sample
        tx_probe_dbm = watts_to_dbm(tx_power_w)
        rx_uplink_dbm = (
            tx_probe_dbm
            + probe.antenna_gain_dbi
            + orbiter.antenna_gain_dbi
            - fspl
            - (probe.system_losses_db + orbiter.system_losses_db)
        )
        uplink_margin = rx_uplink_dbm - probe.rx_sensitivity_dbm

        # Downlink margin (assuming orbiter TX as configured)
        tx_orbiter_dbm = watts_to_dbm(orbiter.tx_power_w)
        rx_downlink_dbm = (
            tx_orbiter_dbm
            + orbiter.antenna_gain_dbi
            + probe.antenna_gain_dbi
            - fspl
            - (probe.system_losses_db + orbiter.system_losses_db)
        )
        downlink_margin = rx_downlink_dbm - probe.rx_sensitivity_dbm

        if uplink_margin <= 0.0 or downlink_margin <= 0.0:
            # At this elevation the link is not closed in one or both directions
            continue

        uplink_margins.append(float(uplink_margin))
        downlink_margins.append(float(downlink_margin))

        total_bits += bitrate_bps * dt
        total_energy += tx_power_w * dt
        valid_time += dt

    if valid_time > 0.0:
        avg_tx_power_w = total_energy / valid_time
    else:
        avg_tx_power_w = 0.0

    dynamic_data_MB = total_bits / 8.0 / 1e6

    uplink_min = min(uplink_margins) if uplink_margins else float("-inf")
    downlink_min = min(downlink_margins) if downlink_margins else float("-inf")

    return DynamicLinkResult(
        range_km=range_km,
        valid_time_s=valid_time,
        avg_tx_power_w=avg_tx_power_w,
        dynamic_data_MB=dynamic_data_MB,
        uplink_margin_min_db=uplink_min,
        downlink_margin_min_db=downlink_min,
    )


# ---------------------------------------------------------------------------
# Deep-space X-band relay (Orbiter -> DSN)
# ---------------------------------------------------------------------------

@dataclass
class DeepSpaceLinkConfig:
    """
    Configuration for a deep-space X-band link.

    This is intentionally generic: can be used for Mars–Earth,
    but also other bodies with appropriate range_km.
    """
    frequency_hz: float
    tx_power_w: float
    orbiter_dish_diameter_m: float
    orbiter_efficiency: float
    dsn_dish_diameter_m: float
    dsn_efficiency: float
    system_losses_db: float
    bitrate_bps: float
    system_temperature_k: float
    required_ebn0_db: float
    receiver_threshold_dbm: float


@dataclass
class DeepSpaceLinkResult:
    range_km: float
    link_margin_db: float
    ebn0_margin_db: float
    rx_power_dbm: float
    snr_db: float


def parabolic_gain_dbi(diameter_m: float, efficiency: float, wavelength_m: float) -> float:
    """
    Approximate parabolic dish gain in dBi.
    """
    gain_linear = efficiency * (np.pi * diameter_m / wavelength_m) ** 2
    return 10.0 * np.log10(gain_linear)


def compute_deep_space_link(
    range_km: float,
    cfg: DeepSpaceLinkConfig,
) -> DeepSpaceLinkResult:
    """
    Compute a single deep-space X-band link budget at a given range.

    Mirrors the structure of the original script's orbit-to-DSN link
    but packaged around a config dataclass.
    """
    c = 299_792_458.0
    wavelength = c / cfg.frequency_hz

    d_m = range_km * 1e3
    fspl = fspl_db(cfg.frequency_hz, d_m)

    orbiter_gain_dbi = parabolic_gain_dbi(
        cfg.orbiter_dish_diameter_m, cfg.orbiter_efficiency, wavelength
    )
    dsn_gain_dbi = parabolic_gain_dbi(
        cfg.dsn_dish_diameter_m, cfg.dsn_efficiency, wavelength
    )

    tx_dbm = watts_to_dbm(cfg.tx_power_w)

    rx_power_dbm = (
        tx_dbm
        + orbiter_gain_dbi
        + dsn_gain_dbi
        - fspl
        - cfg.system_losses_db
    )
    link_margin_db = rx_power_dbm - cfg.receiver_threshold_dbm

    # Eb/N0 margin
    k = 1.380649e-23  # Boltzmann constant
    rx_power_w = dbm_to_watts(rx_power_dbm)
    noise_w = k * cfg.system_temperature_k * cfg.bitrate_bps
    snr_linear = rx_power_w / noise_w
    snr_db = 10.0 * np.log10(snr_linear)
    ebn0_margin_db = snr_db - cfg.required_ebn0_db

    return DeepSpaceLinkResult(
        range_km=range_km,
        link_margin_db=link_margin_db,
        ebn0_margin_db=ebn0_margin_db,
        rx_power_dbm=rx_power_dbm,
        snr_db=snr_db,
    )


# ---------------------------------------------------------------------------
# MAGPIE-style convenience presets
# ---------------------------------------------------------------------------

def magpie_default_probe_433() -> SurfaceProbeRadio:
    """
    Roughly matches the 433 MHz probe config used in MAGPIE comm1d.py:
    - 3 W TX power
    - modest antenna gain
    - conservative receiver sensitivity
    """
    return SurfaceProbeRadio(
        name="MAGPIE Probe 433 MHz",
        frequency_hz=433e6,
        tx_power_w=3.0,
        rx_sensitivity_dbm=-120.0,
        antenna_gain_dbi=3.0,
        system_losses_db=1.0,
    )


def magpie_default_probe_915() -> SurfaceProbeRadio:
    """
    Roughly matches the 915 MHz LoRa probe config used in MAGPIE comm1d.py:
    - 1 W TX power
    - slightly better sensitivity
    """
    return SurfaceProbeRadio(
        name="MAGPIE Probe 915 MHz",
        frequency_hz=915e6,
        tx_power_w=1.0,
        rx_sensitivity_dbm=-130.0,
        antenna_gain_dbi=3.0,
        system_losses_db=1.0,
    )


def magpie_default_orbiter_radio() -> OrbiterRadio:
    """
    Approximate MAGPIE orbiter UHF/LoRa radio:
    - 200 W shared TX power budget (used via watts_to_dbm in original script),
      but here we set a more modest default for near-surface links and leave
      high-power X-band to the deep-space config.
    """
    return OrbiterRadio(
        name="MAGPIE Orbiter UHF/LoRa",
        tx_power_w=10.0,          # deliberately modest; mission can override
        antenna_gain_dbi=7.0,
        system_losses_db=1.0,
    )


def magpie_default_deep_space_cfg() -> DeepSpaceLinkConfig:
    """
    MAGPIE-style deep-space X-band link, loosely matching the original script:
    - 8.4 GHz carrier
    - 200 W TX
    - 2.0 m orbiter dish (adjust as needed)
    - 70 m DSN dish
    """
    freq_xband = 8.4e9
    return DeepSpaceLinkConfig(
        frequency_hz=freq_xband,
        tx_power_w=200.0,
        orbiter_dish_diameter_m=2.0,
        orbiter_efficiency=0.6,
        dsn_dish_diameter_m=70.0,
        dsn_efficiency=0.6,
        system_losses_db=2.0,
        bitrate_bps=10_000.0,
        system_temperature_k=20.0,
        required_ebn0_db=9.6,
        receiver_threshold_dbm=-150.0,
    )


__all__ = [
    # Utilities
    "fspl_db",
    "watts_to_dbm",
    "dbm_to_watts",
    # Probe–orbiter
    "SurfaceProbeRadio",
    "OrbiterRadio",
    "StaticPassConfig",
    "StaticLinkResult",
    "DynamicLinkResult",
    "DynamicProfileFn",
    "compute_static_probe_orbiter_link",
    "compute_dynamic_probe_orbiter_link",
    # Deep-space
    "DeepSpaceLinkConfig",
    "DeepSpaceLinkResult",
    "parabolic_gain_dbi",
    "compute_deep_space_link",
    # MAGPIE presets
    "magpie_default_probe_433",
    "magpie_default_probe_915",
    "magpie_default_orbiter_radio",
    "magpie_default_deep_space_cfg",
]

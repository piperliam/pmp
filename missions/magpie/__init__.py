"""
MAGPIE Mission
==============

Baseline MAGPIE mission orchestration built on top of PMP subsystems.

Provides:
- MagpieBaselineConfig : high-level knobs for the baseline mission.
- run_magpie_baseline  : runs a simple end-to-end chain (transfer, EDL,
                          probe env/power, orbiter EPS, comms).
"""

from dataclasses import dataclass
from typing import Dict, Any

from pmp.trajectory.transfer import EndpointConfig, TransferConfig, compute_lambert_transfer
from pmp.power.probe_env import (
    ProbeEnvConfig, ProbeThermalConfig, ProbeEnvSimConfig, simulate_probe_env_power
)
from pmp.power.orbiter_eps import magpie_default_eps_config, simulate_orbiter_eps
from pmp.edl.edl_1d import (
    magpie_mars_body_default, magpie_probe_aeroshell_default,
    magpie_main_chute_default, magpie_retro_default,
    EDLSimConfig, simulate_edl,
)
from pmp.comms.link_mars import (
    magpie_default_probe_915,
    magpie_default_orbiter_radio,
    StaticPassConfig,
    compute_static_probe_orbiter_link,
    magpie_default_deep_space_cfg,
    compute_deep_space_link,
)


@dataclass
class MagpieBaselineConfig:
    departure_utc: str = "2028-11-10 00:00:00"
    arrival_utc: str = "2029-08-23 00:00:00"
    landing_lat_deg: float = -45.0
    landing_elev_m: float = -3500.0
    tau: float = 0.4
    sim_hours_surface: float = 700.0


def run_magpie_baseline(cfg: MagpieBaselineConfig) -> Dict[str, Any]:
    """
    Run a minimal end-to-end MAGPIE baseline chain.

    Returns a dictionary of subsystem summaries.
    """
    results: Dict[str, Any] = {}

    # --- 1) Transfer (Earth -> Mars) ---
    from poliastro.bodies import Sun, Earth, Mars

    origin = EndpointConfig(epoch_utc=cfg.departure_utc, body=Earth)
    target = EndpointConfig(epoch_utc=cfg.arrival_utc, body=Mars)
    tcfg = TransferConfig(
        central_body=Sun,
        origin=origin,
        target=target,
        long_way=False,
        moi_body_mu_km3_s2=4.282837e4,
        moi_periapsis_alt_km=250.0,
    )
    t_res = compute_lambert_transfer(tcfg)
    results["transfer"] = {
        "tof_days": t_res.tof_days,
        "dv_depart_km_s": t_res.dv_depart_km_s,
        "dv_arrive_km_s": t_res.dv_arrive_km_s,
        "dv_moi_km_s": t_res.dv_moi_km_s,
    }

    # --- 2) EDL ---
    body = magpie_mars_body_default()
    veh = magpie_probe_aeroshell_default()
    chute = magpie_main_chute_default()
    retro = magpie_retro_default()
    edl_cfg = EDLSimConfig(
        body=body,
        vehicle=veh,
        t_max_s=1200.0,
        dt_s=0.1,
        h0_m=125_000.0,
        v0_mps=5800.0,
        parachute=chute,
        retro=retro,
        enable_retro=True,
    )
    edl_res = simulate_edl(edl_cfg)
    touchdown = [ev for ev in edl_res.events if ev.label == "TOUCHDOWN"]
    impact_v = touchdown[-1].details["impact_v_mps"] if touchdown else None
    results["edl"] = {
        "impact_v_mps": impact_v,
        "n_events": len(edl_res.events),
    }

    # --- 3) Surface probe env/power ---
    env = ProbeEnvConfig(
        lat_deg=cfg.landing_lat_deg,
        elev_m=cfg.landing_elev_m,
        tau=cfg.tau,
    )
    therm = ProbeThermalConfig(mli_internal=False, mli_batt=False)
    ps_cfg = ProbeEnvSimConfig(
        sim_hours=cfg.sim_hours_surface,
        dt_s=30.0,
        env=env,
        thermal=therm,
        start_soc=0.8,
    )
    ps_res = simulate_probe_env_power(ps_cfg)
    results["probe_env"] = ps_res.summary

    # --- 4) Orbiter EPS ---
    eps_cfg = magpie_default_eps_config()
    eps_res = simulate_orbiter_eps(eps_cfg)
    results["eps"] = {
        "hit_min_soc": eps_res.meta["flags"]["hit_min_soc_global"],
        "n_orbits": len(eps_res.orbit_table),
    }

    # --- 5) Probe–orbiter link ---
    probe = magpie_default_probe_915()
    orb = magpie_default_orbiter_radio()
    static_cfg = StaticPassConfig(bitrate_bps=5470.0, duration_s=600.0)
    link_res = compute_static_probe_orbiter_link(probe, orb, range_km=250.0, static_cfg=static_cfg)
    results["probe_orbiter_link"] = {
        "uplink_margin_db": link_res.uplink_margin_db,
        "downlink_margin_db": link_res.downlink_margin_db,
        "static_data_MB": link_res.static_data_MB,
    }

    # --- 6) Deep-space link ---
    ds_cfg = magpie_default_deep_space_cfg()
    ds_res = compute_deep_space_link(range_km=2.5e8, cfg=ds_cfg)
    results["deep_space_link"] = {
        "link_margin_db": ds_res.link_margin_db,
        "ebn0_margin_db": ds_res.ebn0_margin_db,
        "rx_power_dbm": ds_res.rx_power_dbm,
    }

    return results


__all__ = [
    "MagpieBaselineConfig",
    "run_magpie_baseline",
]

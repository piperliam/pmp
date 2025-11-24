"""
Entry–Descent–Landing (EDL) Kernels
===================================

Generic 1D EDL models (ballistic + parachutes + retro).
"""

try:
    from .edl_1d import (
        BodyAtmosphere,
        VehicleAeroshell,
        ParachuteConfig,
        RetroConfig,
        EDLSimConfig,
        EDLEvent,
        EDLSimResult,
        simulate_edl,
        magpie_mars_body_default,
        magpie_probe_aeroshell_default,
        magpie_main_chute_default,
        magpie_retro_default,
    )
except Exception:  # pragma: no cover
    pass

__all__ = [
    "BodyAtmosphere",
    "VehicleAeroshell",
    "ParachuteConfig",
    "RetroConfig",
    "EDLSimConfig",
    "EDLEvent",
    "EDLSimResult",
    "simulate_edl",
    "magpie_mars_body_default",
    "magpie_probe_aeroshell_default",
    "magpie_main_chute_default",
    "magpie_retro_default",
]

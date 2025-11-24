"""
Communications & Link Budgets
=============================

Mars-leaning comms utilities for now (probe–orbiter UHF/LoRa,
deep-space X-band). Will be generalized for arbitrary bodies/links.
"""

try:
    from .link_mars import (
        SurfaceProbeRadio,
        OrbiterRadio,
        StaticPassConfig,
        StaticPassResult,
        compute_static_probe_orbiter_link,
        DeepSpaceLinkConfig,
        DeepSpaceLinkResult,
        compute_deep_space_link,
        magpie_default_probe_433,
        magpie_default_probe_915,
        magpie_default_orbiter_radio,
        magpie_default_deep_space_cfg,
    )
except Exception:  # pragma: no cover
    pass

__all__ = [
    "SurfaceProbeRadio",
    "OrbiterRadio",
    "StaticPassConfig",
    "StaticPassResult",
    "compute_static_probe_orbiter_link",
    "DeepSpaceLinkConfig",
    "DeepSpaceLinkResult",
    "compute_deep_space_link",
    "magpie_default_probe_433",
    "magpie_default_probe_915",
    "magpie_default_orbiter_radio",
    "magpie_default_deep_space_cfg",
]

# pmp/core/bodies.py

from dataclasses import dataclass

@dataclass(frozen=True)
class Body:
    name: str
    mu: float          # gravitational parameter [m^3/s^2]
    radius_m: float    # mean radius [m]
    gm_sun: float | None = None  # optional, for heliocentric stuff

# Known bodies (we’ll expand as needed)
EARTH = Body(
    name="Earth",
    mu=3.986004418e14,
    radius_m=6371e3,
)

MARS = Body(
    name="Mars",
    mu=4.282837e13,
    radius_m=3389.5e3,
)

__all__ = ["Body", "EARTH", "MARS"]
